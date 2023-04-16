import torch 
import numpy as np
import torch.nn.functional as F
import h5py
import pickle
import cv2

class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, mode="train", test_flag=False):
        
        super().__init__()
        self.datapaths = []
        with open(f'/kaggle/working/diffusion-anomaly/data/brats/{mode}_brats20_datapaths.pickle', 'rb') as fp:
            self.datapaths = pickle.load(fp)

    def __getitem__(self, idx):
        data = h5py.File(self.datapaths[idx], 'r')
        image = np.transpose(data['image'], [2, 0, 1])
        mask = np.sum(data['mask'], axis=2, dtype=np.int8)
        image_resized = F.interpolate(torch.Tensor(np.expand_dims(image, axis=0)), mode="bilinear", size=(128, 128))[0]
        mask_resized = F.interpolate(torch.Tensor(np.expand_dims(mask, axis=(0, 1))), mode="bilinear", size=(128, 128))[0][0]
        image = np.array(image_resized)        

        for i in range(image.shape[0]):
            image[i] = cv2.normalize(src=image[i], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            image[i] = image[i] / 255

        mask = np.array(mask_resized)
        mask = np.where(mask > 0, 1, 0)
        image = np.flip(image, 2)
        image = np.rot90(image, k=1, axes=(1, 2))
        mask = np.flip(mask, 1)
        mask = np.rot90(mask, k=1)
        label = 1 if np.sum(mask) > 0 else 0
        cond = {}
        cond['y'] = label
        return np.float32(image), cond, label, np.float32(mask)

    def __len__(self):
        return len(self.datapaths)