import os
from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):
    """
    def set_epoch(self, epoch):
        # TODO ?
        pass
    """
    def get_image_paths(self, path):
        paths = [os.path.join(path, p) for p in os.listdir(path)]
        #paths.sort()
        return paths

    """
    def get_pred_ratio(self):
        return random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio
    """

    def build_task_map(self):
        task_map = []
        images = []
        doc_to_author_map = {}
        for i, path in enumerate(self.image_paths):
            author_label = int(str(path).split("/")[-1].split('-')[0])
            document_label = int(str(path).split("/")[-1].split('_')[-1].split(".")[0])

            if self.use_grayscale:
                image = Image.open(path).convert('L')
            else:
                image = Image.open(path).convert('RGB')
            img_width, img_height = image.size

            tasks = [
                (i, document_label, (left, top, left + self.window_size, top + self.window_size))
                for top in range(0, img_height - self.window_size + 1, self.stride)
                for left in range(0, img_width - self.window_size + 1, self.stride)
            ]
            task_map += tasks
            
            images.append(path) #images.append(image)
            if document_label in doc_to_author_map:
                raise Exception("There are multiple documents that have the same label.")
            doc_to_author_map[document_label] = author_label

        return task_map, images, doc_to_author_map


    def __init__(self, root_dir, window_size, stride, transform=None, use_grayscale=True):
        self.transform = transform
        self.image_paths = self.get_image_paths(root_dir)
        self.window_size = window_size
        self.stride = stride
        self.use_grayscale = use_grayscale

        self.task_map, self.images, self.doc_to_author_map = self.build_task_map()
    

    def __len__(self):
        return len(self.task_map)

    def __getitem__(self, idx):
        image_idx, label, box = self.task_map[idx]
        if self.use_grayscale:
            image = Image.open(self.images[image_idx]).convert('L')
        else:
            image = Image.open(self.images[image_idx]).convert('RGB')


        cropped = image.crop(box)
        transformed = self.transform(cropped)
        
        return transformed, label
        
