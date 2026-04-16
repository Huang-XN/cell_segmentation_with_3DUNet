import os
import torch

import numpy as np

import torch.nn.functional as F
#from torchvision.transforms.v2 import functional as F



class EarlyEmbryoDataset(torch.utils.data.Dataset):
    def __init__(self, root,
                 mode="train",
                 patch_size=(32, 256, 256),
                 dtype=torch.float16,
                 random_crop=True):
        self.root        = root
        self.mode        = mode
        self.dtype       = dtype
        self.patch_size  = patch_size
        self.random_crop = random_crop

        self.data_path   = os.path.join(self.root, self.mode, "data")
        self.tgt_path   = os.path.join(self.root, self.mode, "target")
        print(self.tgt_path)

        self.data_sample = list(sorted(os.listdir(self.data_path)))        
        self.tgt_sample = list(sorted(os.listdir(self.tgt_path)))

        print(f"Loaded {len(self.data_sample)} samples for {mode} mode.")

    def _crop_pair(self, img, msk):
        # img, msk shape: (Z, Y, X)
        z, y, x = img.shape
        pz, py, px = self.patch_size

        if pz > z or py > y or px > x:
            raise ValueError(
                f"Patch size {self.patch_size} is larger than image shape {(z, y, x)}"
            )

        if self.random_crop and self.mode == "train":
            z0 = torch.randint(0, z - pz + 1, (1,)).item()
            y0 = torch.randint(0, y - py + 1, (1,)).item()
            x0 = torch.randint(0, x - px + 1, (1,)).item()
        else:
            z0 = (z - pz) // 2
            y0 = (y - py) // 2
            x0 = (x - px) // 2

        img = img[z0:z0 + pz, y0:y0 + py, x0:x0 + px]
        msk = msk[z0:z0 + pz, y0:y0 + py, x0:x0 + px]
        return img, msk
    
    def __getitem__(self, key:int):
        img_path = os.path.join(self.data_path, self.data_sample[key])
        msk_path = os.path.join(self.tgt_path, self.tgt_sample[key])
        img = np.load(img_path)
        msk = np.load(msk_path)

        img, msk = self._crop_pair(img, msk)

        img_tensor  = torch.from_numpy(img).unsqueeze(0).to(self.dtype)
        msk_tensor = torch.from_numpy(msk).unsqueeze(0).to(self.dtype)
        

        return img_tensor, msk_tensor
    
    def __len__(self):
        return len(self.data_sample)


