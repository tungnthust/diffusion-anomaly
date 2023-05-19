import torch 
import numpy as np
import torch.nn.functional as F
import pickle
import glob

def normalize(image):
    """Basic min max scaler.
    """
    min_ = np.min(image)
    max_ = np.max(image)
    scale = max_ - min_
    image = (image - min_) / scale
    return image

def irm_min_max_preprocess(image, low_perc=1, high_perc=99):
    """Main pre-processing function used for the challenge (seems to work the best).
    Remove outliers voxels first, then min-max scale.
    Warnings
    --------
    This will not do it channel wise!!
    """

    non_zeros = image > 0
    if non_zeros.sum() > 0:
        low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
        image = np.clip(image, low, high)
        image = normalize(image)
    return image

class CheXpertDataset(torch.utils.data.Dataset):
    def __init__(self, directory, mode="train", test_flag=False):
        
        super().__init__()
        self.datapaths = []
        self.mode = mode
        self.datapaths = glob.glob(f'/kaggle/input/chexpert-data/chexpert-data/{mode}/*.npz')
        print(f"Number data: {len(self.datapaths)}")

    def __getitem__(self, idx):
        data = np.load(self.datapaths[idx])
        image = data['image']
        image = np.expand_dims(image, axis=0)
        
        image[0] = irm_min_max_preprocess(image[0])
        mask = None
        label = None
        if self.mode == 'train':
            label = data['label']
        else:
            label = 1
            mask = data['mask']
        cond = {}
        cond['y'] = label
        return np.float32(image), cond, label, np.float32(mask)

    def __len__(self):
        return len(self.datapaths)