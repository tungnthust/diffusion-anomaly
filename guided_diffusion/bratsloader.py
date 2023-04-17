import torch 
import numpy as np
import torch.nn.functional as F
import pickle

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
    low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
    image = np.clip(image, low, high)
    image = normalize(image)
    return image

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