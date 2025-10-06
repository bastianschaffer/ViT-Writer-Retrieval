import os
from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):
    #"""
    def set_epoch(self, epoch):
        # TODO ?
        pass
    #"""
    def get_image_paths(self, path):
        for p in os.listdir(path):
            yield os.path.join(path, p)

    #"""
    def get_pred_ratio(self):
        return random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio
    #"""

    def build_task_map(self):
        task_map = []
        images = []
        for i, path in enumerate(self.image_paths):
            label = int(str(path).split("/")[-1].split('-')[0])
            image = Image.open(path)
            if self.bGrayscale:
                image = image.convert('L')
            else:
                image = image.convert('RGB')
            img_width, img_height = image.size

            tasks = [
                (i, label, (left, top, left + self.window_size, top + self.window_size))
                for top in range(0, img_height - self.window_size + 1, self.stride)
                for left in range(0, img_width - self.window_size + 1, self.stride)
            ]
            task_map += tasks
            images.append(image)
        return task_map, images


    def __init__(self, root_dir, transform=None, pred_ratio=None, pred_ratio_var=None, bGrayscale=False):
        self.transform = transform
        self.image_paths = list(self.get_image_paths(root_dir))
        self.window_size = 256
        self.stride = 32
        self.bGrayscale = bGrayscale
        
        self.task_map, self.images = self.build_task_map()

        self.pred_ratio = pred_ratio
        self.pred_ratio_var = pred_ratio_var
        
    

    def __len__(self):
        return len(self.task_map)

    def __getitem__(self, idx):
        image_idx, label, box = self.task_map[idx]
        image = self.images[image_idx]    


        cropped = image.crop(box)
        transformed = self.transform(cropped)
        
        return transformed, label
        
