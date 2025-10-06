# Base code from: https://github.com/VChristlein/icdar24keynote

from sklearn.cluster import MiniBatchKMeans
import scipy
import time
from sklearn.decomposition import PCA
from torchvision import transforms
import torch.nn.functional as F
from collections import defaultdict
from PIL import Image
import numpy as np
import cv2
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from my_dataset import MyDataset 
from itertools import groupby

from attmask_vit import get_teacher
from attmask_vit import get_student

def powernormalize(encs):
    encs = torch.sign(encs) * torch.sqrt(torch.abs(encs))
    norm = torch.linalg.norm(encs)
    if norm > 0:
        encs = encs / norm
    return encs

def assignments(descriptors, clusters):
    """
    compute assignment matrix
    parameters:
        descriptors: TxD descriptor matrix
        clusters: KxD cluster matrix
    returns: TxK assignment matrix
    """
    # compute nearest neighbors
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(descriptors.cpu().numpy().astype(np.float32),
                               clusters.astype(np.float32),
                               k=1)

    # create hard assignment
    assignment = np.zeros((len(descriptors), len(clusters)))
    for e, m in enumerate(matches):
        assignment[e, m[0].trainIdx] = 1

    return assignment


def vlad(desc, cen):
    """
    compute VLAD encoding for one single file
    parameters:
        f:  TxD matrix of image descriptor tokens
        cen: KxD matrix of cluster centers
    returns: K*D encoding vector
    """
    K = cen.shape[0]

    a = assignments(desc, cen)

    T, D = desc.shape
    f_enc = torch.zeros((D * K), dtype=torch.float32, device=desc.device)
    for k in range(K):
        # it's faster to select only those descriptors that have
        # this cluster as nearest neighbor and then compute the
        # difference to the cluster center than computing the differences
        # first and then select

        # get only descriptors that are possible for this cluster
        # => a has size TxK and for each T there is a K map that contains one 1 and the rest is 0
        # => at each t, look if it is >0 at k => true or false for each t
        # => from the T patches we take those that have k as nearest center
        nn = desc[a[:, k] > 0]

        # it can happen that we don't have any descriptors associated for this cluster
        if len(nn) > 0:
            res = nn - cen[k]

            # sum pooling
            f_enc[k * D:(k + 1) * D] = torch.sum(res, axis=0)
    
    f_enc = powernormalize(f_enc)

    return f_enc

def get_image_paths(path):
    for c in os.listdir(path):
        for i in os.listdir(os.path.join(path, c)):
            yield os.path.join(path, c, i)


def get_images_per_author(path, window_size, stride):
    transform = transforms.Compose([transforms.ToTensor()])

    for c in os.listdir(path):
        author_image_windows = []
        for i in os.listdir(os.path.join(path, c)):
            windows = []
            p = os.path.join(path, c, i)
            label = i

            image = Image.open(p).convert('L')
            img_width, img_height = image.size

            for top in range(0, img_height - window_size + 1, stride):
                for left in range(0, img_width - window_size + 1, stride):
                    box = (left, top, left + window_size, top + window_size)
                    windows.append(transform(image.crop(box)).unsqueeze(0)) # add 1 dim as batch dim for the vit
            
            author_image_windows.append((label, windows))
        yield author_image_windows 



def get_dataset_window_len(path, window_size, stride):
    transform = transforms.Compose([transforms.ToTensor()])

    len = 0
    for f in os.listdir(path):
        
        p = os.path.join(path, f)
        label = f #f.split('-')[0]

        image = Image.open(p).convert('RGB')
        img_width, img_height = image.size

        for top in range(0, img_height - window_size + 1, stride):
            for left in range(0, img_width - window_size + 1, stride):
                box = (left, top, left + window_size, top + window_size)
                len += 1
        
    return len
    

def compute_centroids(tokens, centroid_count=100):
    kmeans = MiniBatchKMeans(n_clusters=centroid_count, batch_size=6) #before: batch_size = 6
    return kmeans.fit(tokens).cluster_centers_

"""
    returns variable amount of tokens that will later be accumulated with vlad
"""
def get_tokens_from_vit(batched_windows, vit, patch_size):
    unfolded_patches = F.unfold(batched_windows, kernel_size=patch_size, stride=patch_size) # split window (s_eval x s_eval) into patches (patch_size x patch_size)
    patch_sums = unfolded_patches.sum(dim=1).squeeze()

    foreground_mask = patch_sums > FOREGROUND_THRESHOLD
    foreground_mask = foreground_mask#.cpu().numpy()

    with torch.no_grad():
        vit_output = vit.get_intermediate_layers(batched_windows)[0].cpu()#.detach().numpy()
    
    patch_tokens = vit_output[:, 1:, :]  # Remove cls token
    return patch_tokens, foreground_mask


def get_author_from_doc_label(doc_label):
    return doc_label[:doc_label.index("-")]




"""
Output of the ViT can come from multiple source documents.
This method adds them to the dict based on their label.
"""
def add_doc_tokens(all_tokens_dict, batched_tokens, batched_labels, foreground_mask):
    for label, _ in groupby(batched_labels):
       
        group_tokens = batched_tokens[batched_labels == label]
        group_foreground_mask = foreground_mask[batched_labels == label]
       
        masked_group_tokens = torch.cat([t[m] for t, m in zip(group_tokens, group_foreground_mask)], dim=0)
        label = int(label)
        if label in all_tokens_dict:
            all_tokens_dict[label] = torch.cat((all_tokens_dict[label], masked_group_tokens))
        else:
            all_tokens_dict[label] = masked_group_tokens

def is_empty(d):
    return bool(d)


def safe_concat(t, new):
    if t is None:
        return new
    else:
        return torch.cat((t, new), dim=0)

if __name__ ==  "__main__":
    # CHECKPOINT = "foo-bar" # use some dummy path like this to use random weights for the ViT
    CHECKPOINT = "C/Users/basti/Documents/Uni/5.Semester/Praktukum-Mustererkennung/checkpoints/september/attmask2-my-augmentatino-nonorm-singechannel/checkpoint.pth" # TODO
    TRAIN_PATH = "C:/Users/basti/Documents/Uni/5.Semester/Praktukum-Mustererkennung/datasets/unmodified/icdar2017-training-binary" # TODO 
    TEST_PATH = "C:/Users/basti/Documents/Uni/5.Semester/Praktukum-Mustererkennung/datasets/unmodified/ScriptNet-HistoricalWI-2017-binarized"#"/disks/data1/uh30iwul/icdar2017-test-png" # TODO
    K_TRAIN_EVAL = 224
    S_TRAIN = 224
    S_EVAL = 56
    PCA_COMPONENTS = 384
    FOREGROUND_THRESHOLD = 10
    BATCH_SIZE = 300

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    patch_size = 16
    #vit = get_teacher(patch_size=patch_size, checkpoint_path=CHECKPOINT, device=device) 
    vit, num_channels = get_student(patch_size=patch_size, checkpoint_path=CHECKPOINT, device=device) 
    print("ViT loaded successfully!")

    use_grayscale = num_channels == 1
    if use_grayscale:
        print("Using single-channel weights")
    else:
        print("Using 3-channel weights")

    transform = transforms.Compose([transforms.ToTensor()]) 
    dataset_train = MyDataset(TRAIN_PATH, K_TRAIN_EVAL, S_TRAIN, transform, use_grayscale)
    dataset_test = MyDataset(TEST_PATH, K_TRAIN_EVAL, S_EVAL, transform=transform, use_grayscale)
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=10, shuffle=False, drop_last=False)
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, num_workers=10, shuffle=False, drop_last=False)

    print("train window len: ", len(dataloader_train))
    
    start_time = time.time()
    train_tokens = {}

    
    print("generating train tokens...")
    for batched_windows, batched_labels in tqdm(dataloader_train):

        batched_tokens, foreground_mask = get_tokens_from_vit(batched_windows, vit, patch_size=patch_size)
        #train_tokens.append((class_label, image_tokens))
        add_doc_tokens(train_tokens, batched_tokens, batched_labels, foreground_mask)

    print("train_tokens length: ", len(list(train_tokens.keys())))
    
    concatenated_train_tokens = torch.cat(list(train_tokens.values()), dim=0)
    print(f"concatenated train tokens for k-Means to shape {concatenated_train_tokens.shape}. Generating centroids...")
    
    centroids = compute_centroids(concatenated_train_tokens, centroid_count=100)

    print(f"generated centroids with shape: {centroids.shape}")

    
    print("Generating vlad tokens for PCA...")

    train_encs = []
    for class_label, image_tokens in tqdm(train_tokens.items()):
        # image_tokens are the tokens of only one single image
        f_enc = vlad(image_tokens, centroids)
        train_encs.append(f_enc)
    
    
    pca = PCA(n_components=PCA_COMPONENTS, whiten=True)
    
    print("Fitting PCA...")

    train_encs = np.array(train_encs)
    pca.fit(train_encs)

    print("Generating test tokens...")


    val_doc_encodings = []
    val_doc_labels = []
    i = 0
    
    current_doc_tokens = None
    current_doc_label = None
    for batched_windows, batched_labels in tqdm(dataloader_test):
        batched_tokens, batched_foreground_mask = get_tokens_from_vit(batched_windows, vit, patch_size=patch_size)

        while True:
            if current_doc_label is None:
                current_doc_label = batched_labels[0]

            group_tokens = batched_tokens[batched_labels == current_doc_label]
            group_foreground_mask = batched_foreground_mask[batched_labels == current_doc_label]
            

            if group_tokens.shape[0] > 0:
                masked_group_tokens = torch.cat([t[m] for t, m in zip(group_tokens, group_foreground_mask)], dim=0)
                current_doc_tokens = safe_concat(current_doc_tokens, masked_group_tokens)

            next_group_labels = batched_labels[batched_labels != current_doc_label]
            if next_group_labels.shape[0] > 0:
                # label changed --> current_doc_tokens is finished --> compute vlad
                f_enc = vlad(current_doc_tokens, centroids)
                val_doc_encodings.append(f_enc)
                val_doc_labels.append(int(current_doc_label))

                
                batched_tokens = batched_tokens[batched_labels != current_doc_label]
                batched_foreground_mask = batched_foreground_mask[batched_labels != current_doc_label]
                batched_labels = batched_labels[batched_labels != current_doc_label]
                

                current_doc_label = next_group_labels[0]
                current_doc_tokens = None
            else:
                break
        if i == 200:
            #break
            pass
        i += 1
            

    print("Generating VLAD encodings from test tokens...")



    assert (len(val_doc_encodings) == len(val_doc_labels))

    print("calculated all val tokens from vit and accumulated with vlad successfully")
    
    val_doc_encodings = pca.transform(np.array(val_doc_encodings))

    print("finished transform PCA")

    end_time = time.time()
    elapsed_seconds = end_time - start_time
    hours, remainder = divmod(elapsed_seconds, 60*60)
    minutes, seconds = divmod(remainder, 60)
    print("------------------------------------------------------------------------")
    print(f"Elapsed Time: {hours}h:{minutes}min:{seconds}s")
    print("computing accuracy...")

    labels = torch.tensor([dataset_test.doc_to_author_map[doc_label] for doc_label in val_doc_labels])
    val_doc_encodings = torch.tensor(val_doc_encodings)
    val_doc_encodings = F.normalize(val_doc_encodings) # normalized -> ||x|| == 1 --> no division by length needed
    cosine_sim = val_doc_encodings @ val_doc_encodings.T
    cosine_sim.diagonal().fill_(-2)
    
    average_precisions = []
    top1_count = 0
    for i in range(cosine_sim.shape[0]):
        # note: ranking[i] are the labels of the most similar docs and labels[i] is the current query's label
        query_label = labels[i]
        distances_to_i = cosine_sim[i]
        sorting_indices = torch.argsort(distances_to_i, descending=True)
        sorted_labels = labels[sorting_indices][:-1]
        
        sorted_correct = sorted_labels == query_label
        if sorted_correct[sorted_correct==True].shape[0] == 0:
            continue
        
        top1_count += 1 if sorted_correct[0] == True else 0
        sum_true_positives = torch.cumsum(sorted_correct.float(), dim=0) # is at i the number of true positives until i
        k_range = torch.arange(1, len(sorted_correct) + 1)
        precisions = sum_true_positives / k_range
        num_true_positives = sum_true_positives[-1]

        precisions_at_correct = precisions[sorted_correct]
        average_precisions.append(torch.sum(precisions_at_correct) / len(precisions_at_correct))

    top_1, mAP = top1_count / len(average_precisions), sum(average_precisions) / len(average_precisions)
    

    print("top 1 accuracy:", top_1 * 100)
    print("mAP accuracy:", mAP * 100)
    