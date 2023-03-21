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

        self.datapaths = [p for p in Path(f'{directory}').glob(f'**/*.h5')]

       
    def __getitem__(self, idx):
        data = h5py.File(self.datapaths[idx], 'r')
        image = np.array(data['image'])
        mask = np.array(data['mask'])
        image = np.transpose(image, [2, 0, 1])
        label = 1 if np.sum(mask) > 0 else 0
        padding_image = torch.zeros(4,256,256) + image[0][0][0]
        padding_image[:,8:-8,8:-8] = image
        return padding_image, label

    def __len__(self):
        return len(self.datapaths)