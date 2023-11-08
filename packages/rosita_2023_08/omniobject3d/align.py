import os
import numpy as np

def load_align_data(path):
    p = os.path.join(path,'align.npy')
    if not os.path.exists(p):
        return None
    return np.load(p, allow_pickle=True)
