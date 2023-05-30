import torch 
import numpy as np
import torch.nn.functional as F
import pickle
import glob
import skimage.exposure

def window(data: np.ndarray, lower: float = -125., upper: float = 225., dtype: str = 'float32') -> np.ndarray:
    """ Scales the data between 0..1 based on a window with lower and upper limits as specified. dtype must be a float type.

    Default is a soft tissue window ([-125, 225] ≙ W 350, L50).

    See https://radiopaedia.org/articles/windowing-ct for common width (WW) and center/level (WL) parameters.
    """
    assert 'float' in dtype, 'dtype must be a float type'
    clipped = np.clip(data, lower, upper).astype(dtype)
    # (do not use in_range='image', since this does not yield the desired result if the min/max values do not reach lower/upper)
    return skimage.exposure.rescale_intensity(clipped, in_range=(lower, upper), out_range=(0., 1.))

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
    def __init__(self, directory, mode="train", test_flag=False):
        
        super().__init__()
        self.datapaths = []
        self.mode = mode
        
        with open(f'/kaggle/working/diffusion-anomaly/data/lits/{mode}_lits_data_paths.pkl', 'rb') as f:
            self.datapaths = pickle.load(f)
        print(f"Number data: {len(self.datapaths)}")

    def __getitem__(self, idx):
        data = np.load(self.datapaths[idx])
        image = data['image']
        image = np.expand_dims(image, axis=0)
        
        image[0] = irm_min_max_preprocess(image[0])
        mask = data['tumor_mask']
        if np.sum(mask) > 0:
            label = 1
        else:
            label = 0
        cond = {}
        cond['y'] = label
        return np.float32(image), cond, label, np.float32(mask)

    def __len__(self):
        return len(self.datapaths)