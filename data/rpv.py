import os
import glob
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2.functional import resize_image


# sub_folders = 'None'


class RPV(Dataset):
    def __init__(
        self, dataset_dir, data_split=None, image_size=80, 
        transform=None, subset=None
    ):
        self.dataset_dir = dataset_dir
        self.data_split = data_split
        self.image_size = image_size
        self.transform = transform

        # subsets = os.listdir(self.dataset_dir)

        self.file_names = []
        # for i in subsets:
        file_names = [os.path.basename(f) for f in glob.glob(os.path.join(self.dataset_dir, "*_" + self.data_split + ".npz"))]
        self.file_names = file_names

        # """
        # Get file lists
        test_files = sorted(glob.glob(os.path.join(self.dataset_dir, "*_test.npz")))
        train_files = sorted(glob.glob(os.path.join(self.dataset_dir, "*_train.npz")))


        if self.data_split == "train":
            self.file_names = [os.path.basename(f) for f in train_files]
        elif self.data_split in ["val", "test"]:
            total = len(test_files)
            mid = total // 2
            test_files = [os.path.basename(f) for f in test_files]
            if self.data_split == "val":
                self.file_names = test_files#[:mid]
            else:  # "test"
                self.file_names = test_files#[mid:]
        else:
            raise ValueError("data_split must be one of: 'train', 'val', or 'test'")
        # """

        # # Get file lists
        # test_files = sorted(glob.glob(os.path.join(self.dataset_dir, "*_test.npz")))
        # train_files = sorted(glob.glob(os.path.join(self.dataset_dir, "*_train.npz")))
        #
        # if self.data_split == "train":
        #     self.file_names = [os.path.basename(f) for f in train_files]
        # elif self.data_split == "val":
        #     self.file_names = [os.path.basename(f) for f in test_files]  # Use all test files for validation
        # elif self.data_split == "test":
        #     raise ValueError("The 'test' split is unused. All test files are used for validation.")
        # else:
        #     raise ValueError("data_split must be one of: 'train' or 'val'")

    def __len__(self):
        return len(self.file_names)

    def _get_data(self, idx):
        data_file = self.file_names[idx]
        try:
            data_path = os.path.join(self.dataset_dir, data_file)
            data = np.load(data_path)

            # image = data["image"]#.reshape(16, 360, 640, 3)


            # if self.image_size != 160:
            #     resize_image = np.zeros((16, 3, self.image_size, self.image_size))
            #     for idx in range(0, 16):
            #         resize_image[idx] = cv2.resize(
            #             image[idx], (self.image_size, self.image_size),
            #             interpolation = cv2.INTER_NEAREST
            #         ).transpose(2,0,1)
            # else:
            #     resize_image = image
            resize_image=data["image"]
        except Exception as e:
            print(data_file)
            raise
        return resize_image, data, data_file

    def __getitem__(self, idx):
        image, data, data_file = self._get_data(idx)

        # Get additional data
        target = data["target"]
        meta_target = torch.tensor(0)
        structure = torch.tensor(0)
        structure_encoded = torch.tensor(0)
        del data

        if self.transform:
            image = torch.from_numpy(image).type(torch.float32)         
            image = self.transform(image)

        target = torch.tensor(target, dtype=torch.long)

        return image, target, meta_target, structure_encoded, data_file

    