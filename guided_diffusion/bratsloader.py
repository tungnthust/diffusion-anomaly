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
    def __init__(self, directory, test_flag=False):
        
        super().__init__()

        self.datapaths = [p for p in Path(f'{image_paths}').glob(f'**/*.h5')]

       
    def __getitem__(self, idx):
        data = h5py.File(self.datapaths[idx], 'r')
        image = np.array(data['image'])
        mask = np.array(data['mask'])
        image = np.transpose(image, [2, 0, 1])
        label = 1 if np.sum(mask) > 0 else 0
        
        return image, label

    def __len__(self):
        return len(self.database)