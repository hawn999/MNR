import numpy as np, os

def inspect_npz(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    # 兼容 arr_0 命名
    kimgs = next((k for k in ('images','imgs','arr_0') if k in z), None)
    klabs = next((k for k in ('labels','y','arr_1') if k in z), None)
    imgs = z[kimgs]
    labs = z[klabs]
    print(npz_path)
    print('images shape:', imgs.shape, imgs.dtype)
    print('labels shape:', labs.shape, labs.dtype)
    u, c = np.unique(labs, return_counts=True)
    print('unique labels:', dict(zip(u.tolist(), c.tolist())))

    # 统一成 0/1
    if labs.ndim > 1:                 # one-hot 或列向量
        labs01 = labs.argmax(axis=-1)
    else:
        if set(np.unique(labs)).issubset({-1, 1}):
            labs01 = (labs > 0).astype(np.int64)
        else:
            labs01 = labs.astype(np.int64)   # 假定已是 0/1/2...
    print('mapped (first 10):', labs01[:10])
    return imgs, labs01

# inspect_npz('/home/scxhc1/nvme_data/SVRT/problem_1/train.npz')
# inspect_npz('/home/scxhc1/nvme_data/SVRT/problem_1/test.npz')
# inspect_npz('/home/scxhc1/nvme_data/SVRT/problem_1/val.npz')

import os
import glob
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

sub_folders = {'0': "center_single",
               '1': "in_center_single_out_center_single",
               '2': "up_center_single_down_center_single",
               '3': "left_center_single_right_center_single",
               '4': "distribute_four",
               '5': "distribute_nine",
               '6': "in_distribute_four_out_center_single"}


class SVRT(Dataset):
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

        self.file_names = []
        for i in subsets:
            file_names = [os.path.basename(f) for f in
                          glob.glob(os.path.join(self.dataset_dir, i, "*_" + self.data_split + ".npz"))]
            self.file_names += [os.path.join(i, f) for f in file_names]

    def __len__(self):
        return len(self.file_names)

    def _get_data(self, idx):
        data_file = self.file_names[idx]

        data_path = os.path.join(self.dataset_dir, data_file)
        data = np.load(data_path)

        # image = data["image"].reshape(16, 160, 160)
        # if self.image_size != 160:
        #     resize_image = np.zeros((16, self.image_size, self.image_size))
        #     for idx in range(0, 16):
        #         resize_image[idx] = cv2.resize(
        #             image[idx], (self.image_size, self.image_size),
        #             interpolation = cv2.INTER_NEAREST
        #         )
        # else:
        #     resize_image = data["image"]#image
        resize_image = data["image"]
        return resize_image, data, data_file

    def __getitem__(self, idx):
        image, data, data_file = self._get_data(idx)

        # Get additional data
        target = data["target"]
        meta_target = data["meta_target"]
        structure = data["structure"]
        structure_encoded = data["meta_matrix"]
        del data

        if self.transform:
            image = torch.from_numpy(image).type(torch.float32)
            image = self.transform(image)

        target = torch.tensor(target, dtype=torch.long)
        meta_target = torch.tensor(meta_target, dtype=torch.float32)
        structure_encoded = torch.tensor(structure_encoded, dtype=torch.float32)

        return image, target, meta_target, structure_encoded, data_file


