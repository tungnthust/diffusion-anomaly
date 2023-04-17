import torch 
import numpy as np
import torch.nn.functional as F
import pickle
import cv2

class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, mode="train", test_flag=False):
        
        super().__init__()
        self.datapaths = []
        with open(f'/kaggle/working/diffusion-anomaly/data/brats/{mode}_brats20_datapaths.pickle', 'rb') as fp:
            self.datapaths = pickle.load(fp)

    def __getitem__(self, idx):
        data = np.load(self.datapaths[idx])
        image = data['image']
        mask = data['mask']
        label = 1 if np.sum(mask) > 0 else 0
        cond = {}
        cond['y'] = label
        return np.float32(image), cond, label, np.float32(mask)

    def __len__(self):
        return len(self.datapaths)