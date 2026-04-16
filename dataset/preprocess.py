import os
import numpy as np
import pandas as pd

from skimage.filters import threshold_otsu
from tqdm import tqdm

def preprocess_data(path=None, embryo="F11", mode='train'):
    if path is None:
        path = "/home/huangxn/Desktop/Playground/virtual_embryo/heem/preimp_mouse/early_mouse/data/"
    if embryo is None:
        embryo = "F11"

    try:
        if mode == "train":
            data_path = os.path.join(path, mode, embryo)
        elif mode == "test":
            data_path = os.path.join(path, mode, embryo)
        else:
            data_path = os.path.join(path, mode, embryo)
    except Exception as e:
        print(f"Warning: You have to choose one right mode in train, val and test")

    samples = os.listdir(data_path)

    dir_exist('./normalized_data')
    dir_exist('./normalized_data/train')
    dir_exist('./normalized_data/train/data')
    dir_exist('./normalized_data/train/target')

    ### data sample
    for sample in tqdm(samples):
        img_path = os.path.join(data_path,sample,'images',sample+"_image_0001.npy")
        msk_path = os.path.join(data_path,sample,'masks',sample+"_masks_0001.npy")
        img = np.load(img_path)
        msk = np.load(img_path)

        img_norm = normalize_zstack_percentile(img)
        file_name_norm_img = f"{sample}_norm_img.npy"

    ### target sample
        msk_norm      = normalize_zstack_min_max(msk)
        msk_binarized = binary_global_otsu_zstack(msk_norm)
        file_name_norm_msk = f"{sample}_norm_msk.npy"
        
        np.save(f"./normalized_data/train/data/{file_name_norm_img}", img_norm)
        np.save(f"./normalized_data/train/target/{file_name_norm_msk}", msk_binarized)



def normalize_zstack_percentile(data, p_low=0.5, p_high=99.5):
    norm_data = np.zeros_like(data, dtype=np.float32)
    num_slice = data.shape[0]

    for z in range(num_slice):
        slice = data[z]

        bg        = np.percentile(slice, p_low)
        corrected = np.maximum(slice-bg, 0)

        upper = np.percentile(slice, p_high)

        if upper > 0:
            norm_data[z] = np.clip(corrected/upper, 0.0, 1.0)
        else:
            norm_data[z] = corrected / (corrected.max() + 1e-8)

    return norm_data

def normalize_zstack_min_max(data):
    '''
    This is only for data with low range 
    '''
    norm_data = (data - data.min()) / (data.max()).astype(np.float32)
    return norm_data

def binary_global_otsu_zstack(data):
    '''
    '''
    flat_data = data.ravel().astype(np.float32)
    thresh = threshold_otsu(flat_data)
    
    binary = (data > thresh).astype(np.uint8)
    return binary

def dir_exist(dir):
    if os.path.isdir(dir):
        print("Path exists.")
    else:
        os.mkdir(dir)

if __name__ == "__main__":
    preprocess_data()