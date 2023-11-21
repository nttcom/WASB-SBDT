import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    # (Conv2D => [BN] => ReLU) * 2 or (Conv2D => ReLU => [BN]) * 2
    def __init__(self, in_channels, out_channels, mid_channels=None, conv_bias=True, bn_first=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        if bn_first:
            self.double_conv = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=conv_bias),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=conv_bias),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True))
        else:
            self.double_conv = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=conv_bias),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(mid_channels),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=conv_bias),        
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(out_channels))
    def forward(self, x):
        return self.double_conv(x)

class TripleConv(nn.Module):
    # (Conv2D => [ReLU] => BN) * 3 or (Conv2D => BN => ReLU)
    def __init__(self, in_channels, out_channels, mid_channels=None, conv_bias=True, bn_first=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        if bn_first:
            self.triple_conv = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=conv_bias),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=conv_bias),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=conv_bias),        
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True))
        else:
            self.triple_conv = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=conv_bias),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(mid_channels),
                    nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=conv_bias),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(mid_channels),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=conv_bias),        
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return self.triple_conv(x)

class Down(nn.Module):
    # Downscaling with maxpool then double / triple conv
    def __init__(self, n, in_channels, out_channels, bn_first=True):
        super().__init__()
        if not n in [2,3]:
            raise ValueError('n must be 2 or 3')
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, bn_first=bn_first) if n==2 else TripleConv(in_channels, out_channels, bn_first=bn_first)
            )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    # Upscaling then double / triple conv
    def __init__(self, n, in1_channels, in2_channels, out_channels, bilinear=True, mode='nearest', bn_first=True, halve_channel=True):
        super().__init__()
        if not n in [2,3]:
            raise ValueError('n must be 2 or 3')
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            if halve_channel:
                raise ValueError('halve_channel must be False if bilinear is True')
            else:
                if mode=='nearest':
                    self.up = nn.Upsample(scale_factor=2, mode=mode)
                else:
                    self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
                if n==2:
                    self.conv = DoubleConv(in1_channels+in2_channels, out_channels, bn_first=bn_first)
                elif n==3:
                    self.conv = TripleConv(in1_channels+in2_channels, out_channels, bn_first=bn_first) 
        else:
            if halve_channel:
                self.up = nn.ConvTranspose2d(in1_channels, in1_channels//2, kernel_size=2, stride=2)
                if n==2:
                    self.conv = DoubleConv(in1_channels//2+in2_channels, out_channels, mid_channels=out_channels, bn_first=bn_first)
                elif n==3:
                    self.conv = TripleConv(in1_channels//2+in2_channels, out_channels, mid_channels=out_channels, bn_first=bn_first)
            else:
                self.up = nn.ConvTranspose2d(in1_channels, in1_channels, kernel_size=2, stride=2)
                if n==2:
                    self.conv = DoubleConv(in1_channels+in2_channels, out_channels, mid_channels=out_channels, bn_first=bn_first)
                elif n==3:
                    self.conv = TripleConv(in1_channels+in2_channels, out_channels, mid_channels=out_channels, bn_first=bn_first)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

