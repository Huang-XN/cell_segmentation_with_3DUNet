import torch
import torch.nn as nn
from torch.nn import functional as F

class ConvBlock(nn.Module):
    def __init__(self, 
                 in_channel:int, 
                 out_channel:int,
                 kernel_size:int = 3):
        super().__init__()

        self.conv_block1 = nn.Sequential(nn.Conv3d(in_channel, out_channel,
                                                   kernel_size=kernel_size,
                                                   padding=kernel_size//2),
                                         nn.BatchNorm3d(out_channel),
                                         nn.ReLU(inplace=True))
        
        self.conv_block2 = nn.Sequential(nn.Conv3d(out_channel, out_channel,
                                                   kernel_size=kernel_size, 
                                                   padding=kernel_size//2),
                                         nn.BatchNorm3d(out_channel),
                                         nn.ReLU(inplace=True))
    
    def forward(self, x:torch.Tensor)-> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)

        return x
    
class DownBlock(nn.Module):
    def __init__(self,
                 in_channel:int, 
                 out_channel:int,
                 kernel_size:int = 3):
        super().__init__()
        self.conv = ConvBlock(in_channel,out_channel,kernel_size)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor):
        feat        = self.conv(x)
        output_down = self.pool(feat)
        return feat, output_down

class UpBlock(nn.Module):
    def __init__(self,
                 in_channel:int, 
                 out_channel:int,
                 skip_channel:int,
                 kernel_size:int = 3):
        super().__init__()
        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(in_channel, out_channel, kernel_size, padding=kernel_size//2)
        )
        self.conv = ConvBlock(out_channel+skip_channel, out_channel, kernel_size)
    
    def forward(self, x, skip):
        x = self.up_sample(x)

        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(
                skip,
                size=x.shape[2:],
                mode='trilinear',
                align_corners=False
            )

        x = torch.cat([x,skip],dim=1)
        x = self.conv(x)
        return x 

class ThreeDimUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: tuple = (16, 32, 64, 128, 256),
        kernel_size: int = 3
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.kernel_size = kernel_size
        
        self.init_conv = nn.Conv3d(in_channels, channels[0],
                                   kernel_size=kernel_size, 
                                   padding=kernel_size//2)

        self.down = nn.ModuleList()

        for i in range(len(channels)-1):
            blk = DownBlock(channels[i],
                            channels[i+1],
                            kernel_size)
            self.down.append(blk)
        
        self.bottleneck = ConvBlock(channels[-1], 
                                    channels[-1], 
                                    kernel_size=kernel_size)

        self.up = nn.ModuleList()
        
        for i in range(len(channels)-2,-1,-1):
            blk = UpBlock(channels[i+1],
                          channels[i],
                          channels[i+1],
                          kernel_size)
            self.up.append(blk)
        

        self.final_conv = nn.Conv3d(channels[0], out_channels, kernel_size=1)
        #self.final_act  = nn.Sigmoid() if out_channels == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        x = self.init_conv(x)
        res = []

        for d in self.down:
            feat, x = d(x)
            res.append(feat)

        x = self.bottleneck(x)
        
        for u in self.up:
            skip = res.pop()
            x = u(x,skip)
        
        x = self.final_conv(x)
        #x = self.final_act(x)

        return x
