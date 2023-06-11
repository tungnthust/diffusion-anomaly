import torch 
import numpy as np
import torch.nn.functional as F
import pickle
import glob
import pandas as pd

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

class LiTSDataset(torch.utils.data.Dataset):
    def __init__(self, mode="train", fold=1, test_flag=False):
        
        super().__init__()
        self.datapaths = []
        data_split = np.load('/kaggle/working/diffusion-anomaly/data/lits/data_split.npz', allow_pickle=True)
        meta_data_df = pd.read_csv('/kaggle/working/diffusion-anomaly/data/lits/meta_data.csv')
        volume_ids = data_split[f'{mode}_folds'].item()[f'fold_{fold}']
        if not test_flag:
            self.datapaths = meta_data_df[meta_data_df['volume'].isin(volume_ids)]['path'].values
        else:
            self.datapaths = meta_data_df[meta_data_df['volume'].isin(volume_ids) & meta_data_df['label'] == 1]['path'].values
        print(f'Number of {mode} data: {len(self.datapaths)}')

    def __getitem__(self, idx):
        data = np.load(self.datapaths[idx])
        image = data['image']
        image = np.expand_dims(image, axis=0)
        
        image[0] = normalize(np.clip(image[0], -21, 189))
        tumor_mask = data['tumor_mask']
        liver_mask = data['liver_mask']
        liver_mask = np.expand_dims(liver_mask, axis=0)
        if np.sum(tumor_mask) > 0:
            label = 1
        else:
            label = 0
        cond = {}
        cond['y'] = label
        image = image * 2 - 1
        return np.float32(image), cond, label, np.float32(liver_mask), np.float32(tumor_mask)

    def __len__(self):
        return len(self.datapaths)