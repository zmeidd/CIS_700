import psutil
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os


class C2xDataset(Dataset):
    def __init__(self, data_path):
        all_data = np.load(data_path)
        self.pivot = all_data['pivot']
        self.statement = all_data['statement']
        self.label = all_data['label']

    def __len__(self):
        return self.pivot.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pivot = torch.tensor(self.pivot[idx, :])
        statement = torch.tensor(self.statement[idx, :])
        label = torch.tensor([self.label[idx]], dtype=torch.float32)
        return pivot, statement, label
