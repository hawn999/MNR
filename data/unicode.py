import os
import glob
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset


sub_folders = {'0': "Ar", 
               '1': "Co", 
               '2': "Di",
               '3': "Pr",
               '4': "Un"}


class Unicode(Dataset):
    def __init__(
        self, dataset_dir, data_split=None, image_size=80, 
        transform=None, subset=None
    ):
        self.dataset_dir = dataset_dir
        self.data_split = data_split
        self.image_size = image_size
        self.transform = transform

        if subset == 'None':
            subsets = os.listdir(self.dataset_dir)
        else:
            subsets = [sub_folders[subset]]
        self.subsets = subsets
        self.file_names = []
        for i in subsets:
            file_names = [f for f in os.listdir(os.path.join(self.dataset_dir, i)) if data_split in f]
            self.file_names += [os.path.join(i, f) for f in file_names]


    def __len__(self):
        return len(self.file_names)

    def _get_data(self, idx):
        data_file = self.file_names[idx]
        
        data_path = os.path.join(self.dataset_dir, data_file)
        data = np.load(data_path)

        image = data["images"].reshape(9, 80, 80)

        if self.image_size != 80:
            resize_image = np.zeros((9, self.image_size, self.image_size))
            for idx in range(0, 9):
                resize_image[idx] = cv2.resize(
                    image[idx], (self.image_size, self.image_size), 
                    interpolation = cv2.INTER_NEAREST
                )
        else:
            resize_image = image

        return resize_image, data, data_file

    def __getitem__(self, idx):
        image, data, data_file = self._get_data(idx)

        # Get additional data
        target = data["target"]
        del data
        
        if self.transform:
            image = torch.from_numpy(image).type(torch.float32)         
            image = self.transform(image)

        target = torch.tensor(target, dtype=torch.long)
        meta_target = torch.tensor(0, dtype=torch.float32)
        structure_encoded = torch.tensor(0, dtype=torch.float32)

        return image, target, meta_target, structure_encoded, data_file

    