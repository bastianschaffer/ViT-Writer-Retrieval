# Base code from: https://github.com/VChristlein/icdar24keynote

from sklearn.cluster import MiniBatchKMeans
import scipy
from sklearn.decomposition import PCA
from torchvision import transforms
import torch.nn.functional as F
from collections import defaultdict
from PIL import Image
import numpy as np
import cv2
import os

from attmask_vit import get_teacher

def powernormalize(encs):
    encs = np.sign(encs) * np.sqrt(np.abs(encs))
    norm = np.linalg.norm(encs)
    if norm > 0:
        encs = encs / norm #normalize(encs, norm='l2')
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
    matches = matcher.knnMatch(descriptors.astype(np.float32),
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
    f_enc = np.zeros((D * K), dtype=np.float32)
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
            f_enc[k * D:(k + 1) * D] = np.sum(res, axis=0)
    
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

def get_train_images(path, window_size, stride):
    transform = transforms.Compose([transforms.ToTensor()])

    for c in os.listdir(path):
        
        for i in os.listdir(os.path.join(path, c)):
            windows = []
            p = os.path.join(path, c, i)
            label = i

            image = Image.open(p).convert('L')
            img_width, img_height = image.size

            for top in range(0, img_height - window_size + 1, stride):
                for left in range(0, img_width - window_size + 1, stride):
                    box = (left, top, left + window_size, top + window_size)
                    windows.append(transform(image.crop(box)).unsqueeze(0).cuda()) # add 1 dim as batch dim for the vit
            
            yield (label, windows) 

def get_train_images_len(path):
    len = 0
    image_paths = get_image_paths(path)
    for p in image_paths:
        len += 1

    return len


def is_foreground_image(image, threshold):
    return np.sum(image) > threshold
    

def compute_centroids(tokens, centroid_count=100):
    kmeans = MiniBatchKMeans(n_clusters=centroid_count, batch_size=6) #before: batch_size = 6
    return kmeans.fit(tokens).cluster_centers_


"""
    returns variable amount of tokens that will later be accumulated with vlad
"""
def get_tokens_from_vit(doc_windows, vit, patch_size):
    doc_tokens = None
    for window in doc_windows:
        unfolded_patches = F.unfold(window, kernel_size=patch_size, stride=patch_size) # split window (s_eval x s_eval) into patches (patch_size x patch_size)
        patch_sums = unfolded_patches.sum(dim=1).squeeze()

        foreground_mask = patch_sums > FOREGROUND_THRESHOLD
        foreground_mask = foreground_mask.cpu().numpy()

        vit_output = vit.get_intermediate_layers(window)[0].cpu().detach().numpy()
        
        patch_tokens = vit_output[0, 1:, :]  # Remove batch dim and cls token
        window_foreground_tokens = patch_tokens[foreground_mask, :]


        if doc_tokens is None:
            doc_tokens = np.copy(window_foreground_tokens)
        else:
            doc_tokens = np.concatenate((doc_tokens, window_foreground_tokens))
    return doc_tokens

def get_author_from_doc_label(doc_label):
    return doc_label[:doc_label.index("-")]

if __name__ ==  "__main__":
    # CHECKPOINT = "foo-bar" # use some dummy path like this if you want to use random weights for the ViT
    CHECKPOINT = "<replace>" # TODO
    TRAIN_PATH = "<replace>" # TODO
    VAL_PATH = "<replace>" # TODO
    K_TRAIN_EVAL = 224 # kernel size for vlad pipeline
    S_TRAIN = 224
    S_EVAL = 56
    PCA_COMPONENTS = 384
    FOREGROUND_THRESHOLD = 10

    patch_size = 16
    vit = get_teacher(patch_size=patch_size, pretrained_weights=CHECKPOINT, architecture="vit_small") 
    print("teacher loaded successfully!")


    train_images = get_train_images(TRAIN_PATH , window_size=K_TRAIN_EVAL, stride=S_TRAIN)
    train_images_len = get_train_images_len(TRAIN_PATH)
    train_tokens = []

    
    print("generating train tokens...")
    for class_label, image_windows in train_images:
        image_tokens = get_tokens_from_vit(image_windows, vit, patch_size=patch_size)
        train_tokens.append((class_label, image_tokens))

    print("generating centroids...")
    centroids = compute_centroids(np.concatenate([tokens for _, tokens in train_tokens]), centroid_count=100)

    print(f"generated centroids with shape: {centroids.shape}")

    pca = PCA(n_components=PCA_COMPONENTS, whiten=True)
    all_encs = []
    for class_label, image_tokens in train_tokens:
        # image_tokens are the tokens of only one single image
        f_enc = vlad(image_tokens, centroids)
        
        all_encs.append(f_enc)

    all_encs = np.array(all_encs)

    all_encs = pca.fit_transform(all_encs)

    val_doc_encodings = []
    val_doc_labels = []

    val_images = get_train_images(VAL_PATH, window_size=K_TRAIN_EVAL, stride=S_EVAL)
    for label, image_windows in val_images:
        image_tokens = get_tokens_from_vit(image_windows, vit, patch_size=patch_size)
        f_enc = vlad(image_tokens, centroids)
        val_doc_encodings.append(f_enc)
        val_doc_labels.append(label)

    assert (len(val_doc_encodings) == len(val_doc_labels))

    print("calculated all val tokens from vit and accumulated with vlad successfully")
    
    val_doc_encodings = pca.transform(np.array(val_doc_encodings))

    print("pca for val encodings successful")
    top_1_results = defaultdict(list)
    top_5_results = defaultdict(list)


    for i in range(len(val_doc_encodings)):
        query = val_doc_encodings[i]

        query_label = val_doc_labels[i]
        query_author = get_author_from_doc_label(query_label)

        distances = []
        for j in range(len(val_doc_encodings)):
            if i == j:
                continue
            distance = scipy.spatial.distance.cosine(query, val_doc_encodings[j])
            distances.append((val_doc_labels[j], distance))
        sorted_distances = sorted(distances, key=lambda x: x[1])
        
        top_1_author = get_author_from_doc_label(sorted_distances[0][0]) # label of smallest distance
        top_1_results[query_author].append(query_author == top_1_author)

        top_5_authors = list(map(lambda x: get_author_from_doc_label(x[0]), sorted_distances[:5]))
        top_5_results[query_author].append(query_author in top_5_authors)


    success_rates_top_1 = []
    for author, results in top_1_results.items():
        success_rate = len([success for success in results if success == True]) / len(results)
        success_rate = round(success_rate * 100, 2)
        success_rates_top_1.append(success_rate)
    
    print("---")
    success_rates_top_5 = []
    for author, results in top_5_results.items():
        success_rate = len([success for success in results if success == True]) / len(results)
        success_rate = round(success_rate * 100, 2)
        success_rates_top_5.append(success_rate)

    avg_success = sum(success_rates_top_1) / len(success_rates_top_1)
    avg_success = round(avg_success, 1)
    print(f"average (per author) top 1 success rate: {avg_success} %")


    top_1_flat = np.concatenate(list(top_1_results.values())).tolist()
    top1_total = len([x for x in top_1_flat if x == True]) / len(top_1_flat)
    top1_total = round(top1_total * 100, 2)
    print(f"top 1 total: {top1_total}")

    avg_success = sum(success_rates_top_5) / len(success_rates_top_5)
    avg_success = round(avg_success, 1)
    print(f"average top 5 success rate: {avg_success} %")
