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
from sklearn.preprocessing import MinMaxScaler
class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, mode="train", test_flag=False):
        
        super().__init__()
        
        paths = [p for p in Path(f'{directory}').glob(f'**/*.h5')]
        self.datapaths = []
        for path in paths:
            volume_idx = int(str(path).split('/')[-1].split('_')[1])
            slice_idx = int(str(path).split('/')[-1].split('_')[3].split('.')[0])
            if (mode == "val") and (volume_idx > 52 and volume_idx <= 88) and (slice_idx >= 80 and slice_idx <= 128):
                self.datapaths.append(path)
            if (mode == "train") and (volume_idx <= 52 or volume_idx > 88) and (slice_idx >= 80 and slice_idx <= 128):
                self.datapaths.append(path)

    def __getitem__(self, idx):
        scaler = MinMaxScaler()
        data = h5py.File(self.datapaths[idx], 'r')
        image = np.array(data['image'])
        mask = np.array(data['mask'])
        image = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
        image = np.transpose(image, [2, 0, 1])
        label = 1 if np.sum(mask) > 0 else 0
        padding_image = np.zeros((4,256,256)) + np.broadcast_to(image[:, 0, 0][:, np.newaxis, np.newaxis], (4, 256, 256))
        padding_image[:,8:-8,8:-8] = image
        cond = {}
        cond['y'] = label
        return np.float32(padding_image), cond, label

    def __len__(self):
        return len(self.datapaths)