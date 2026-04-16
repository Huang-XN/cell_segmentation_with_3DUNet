from dataset.datasets import EarlyEmbryoDataset
from model.model import ThreeDimUNet
from train import train

from torch.utils.data import DataLoader, sampler

from torch.nn import functional as F
import torch.optim as optim
import torch

root = "./dataset/normalized_data"

train_data = EarlyEmbryoDataset(root,
                                mode="train",
                                dtype=torch.float32)

train_data_loader = DataLoader(train_data, 
                               batch_size = 1,
                               pin_memory = True,
                               num_workers= 2)


model = ThreeDimUNet(
    in_channels=1,
    out_channels=1,
    channels=(4, 8, 16),   # Much smaller than (16,32,64,128,256)
    kernel_size=3
    )
optimizer = optim.Adam(model.parameters(),
                       lr = 1e-3)
x, y = train_data[0]  # Load one sample
print(f"Single sample shape: {x.shape}")  # Expected: (1, Z, Y, X)
print(f"Approximate volume size: {x.numel() / 1e6:.1f} million voxels")

torch.cuda.empty_cache()
train(train_data_loader,
      model,
      optimizer)