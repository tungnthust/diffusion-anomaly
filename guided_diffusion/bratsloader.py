import torch 
import torch.nn
import numpy as np
import random
import os
import os.path
import nibabel
from scipy import ndimage
import h5py
from pathlib import Path

class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, mode="train", test_flag=False):
        
        super().__init__()
        
        paths = [p for p in Path(f'{directory}').glob(f'**/*.h5')]
        self.datapaths = []
        for path in paths:
            volume_idx = int(str(paths[0]).split('/')[-1].split('_')[1])
            if mode == "val" and volume_idx >= 295:
                self.datapaths.append(path)
            if mode == "train" and volume_idx < 295:
                self.datapaths.append(path)

    def __getitem__(self, idx):
        data = h5py.File(self.datapaths[idx], 'r')
        image = np.array(data['image'])
        mask = np.array(data['mask'])
        image = np.transpose(image, [2, 0, 1])
        label = 1 if np.sum(mask) > 0 else 0
        padding_image = np.zeros((4,256,256)) + image[0][0][0]
        padding_image[:,8:-8,8:-8] = image
        return np.float32(padding_image), label

    def __len__(self):
        return len(self.datapaths)