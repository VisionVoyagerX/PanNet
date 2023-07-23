import os
import torch.nn as nn
import torch
from torch.nn.functional import interpolate
import kornia

import matplotlib.pyplot as plt

class Residual_Block(nn.Module):
    def __init__(self, channels):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        fea = self.relu(self.bn1(self.conv1(x)))
        fea = self.relu(self.bn2(self.conv2(fea)))
        result = fea + x
        return result
       

class PanNet_model(nn.Module):
    def __init__(self, scale, ms_channels,**kwargs):
        super(PanNet_model, self).__init__()
        self.mslr_mean = kwargs.get('mslr_mean')
        self.mslr_std =  kwargs.get('mslr_std')
        self.pan_mean =  kwargs.get('pan_mean')
        self.pan_std =  kwargs.get('pan_std')
        self.scale = scale

        self.layer_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ms_channels, out_channels=ms_channels, kernel_size=8, stride=4, padding=2, output_padding=0)
        )
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=ms_channels + 1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.layer_2 = nn.Sequential(
            self.make_layer(Residual_Block, 4, 32),
            nn.Conv2d(in_channels=32, out_channels=ms_channels, kernel_size=3, stride=1, padding=1)
        )

        self.interpolate = interpolate

    def make_layer(self, block, num_of_layer, channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channels))
        return nn.Sequential(*layers)

    def forward(self, pan, mslr):
        #channel-wise normalization
        '''pan = (pan - self.pan_mean) / self.pan_std
        mslr = (mslr - self.mslr_mean) / self.mslr_std'''

        lr_up = self.interpolate(mslr, scale_factor=self.scale, mode='bicubic')
        lr_hp = mslr - kornia.filters.BoxBlur((5, 5))(mslr)
        pan_hp = pan - kornia.filters.BoxBlur((5, 5))(pan)
        lr_u_hp = self.layer_0(lr_hp)
        ms = torch.cat([pan_hp, lr_u_hp], dim=1)
        fea = self.layer_1(ms)
        output = self.layer_2(fea) + lr_up



        #channel-wise denormalization
        '''sum(p.numel() for p in self.parameters() if p.requires_grad)'''


        return output


if __name__ == "__main__":
    pan = torch.randn(1, 1, 256, 256)
    lr = torch.randn(1, 8, 64, 64)
    pnn = PanNet_model(scale=4, ms_channels=8)
    print(pnn(pan, lr).shape)
