import numpy as np
import torch
from torch.utils.data import Dataset
import os
import glob
from tqdm import tqdm

class SRAVEN_InMemory(Dataset):
    def __init__(self, dataset_dir, data_split="train", **kwargs):
        file_paths = sorted(glob.glob(os.path.join(dataset_dir, f"sraven_{data_split}_*.npz")))

        all_panels = []
        all_targets = []

        print(f"Loading {data_split} data into memory...")
        for path in tqdm(file_paths):
            data = np.load(path)
            all_panels.append(data['symbolic_panels'])
            all_targets.append(data['target'])

        # 将所有分片数据在内存中拼接成一个大数组
        self.panels = np.concatenate(all_panels, axis=0)
        self.targets = np.concatenate(all_targets, axis=0)

        print(f"Successfully loaded {len(self.targets)} samples for {data_split} split.")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # 直接从内存中索引，速度极快
        panel_data = self.panels[idx]
        target_data = self.targets[idx]

        return (
            torch.from_numpy(panel_data).float(),
            torch.tensor(target_data, dtype=torch.long),
            torch.tensor(0.), # dummy
            torch.tensor(0.), # dummy
            "in_memory_sample"
        )