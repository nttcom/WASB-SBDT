from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class Neck(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, conv_bias=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=conv_bias),
                                   nn.BatchNorm2d(mid_channels),
                                   nn.ReLU(inplace=True) )
        self.conv2 = nn.Sequential(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=conv_bias),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True) )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        return out1+out2

class ResBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels, mid_channels=None, conv_bias=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if not num_convs in [2,3]:
            raise ValueError('num_convs must be 2 or 3')
        if num_convs==2:
            in_list  = [in_channels, mid_channels]
            out_list = [mid_channels, out_channels]
        elif num_convs==3:
            in_list  = [in_channels, mid_channels, mid_channels]
            out_list = [mid_channels, mid_channels, out_channels]

        self.conv0  = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=conv_bias)
        module_dict = OrderedDict()
        for i, (in_c, out_c) in enumerate(zip(in_list, out_list)):
            module_dict['conv{}'.format(i)] = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=conv_bias)
            module_dict['bn{}'.format(i)]   = nn.BatchNorm2d(out_c)
            module_dict['relu{}'.format(i)] = nn.ReLU(inplace=True)
        self.convs = nn.Sequential(module_dict)

    def forward(self, x):
        identity = x
        out      = self.convs(x)
        return out+self.conv0(identity)

class Down(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResBlock(num_convs, in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    # Upscaling then double / triple conv
    def __init__(self, num_convs, in1_channels, in2_channels, out_channels, bilinear=True, halve_channel=False):
        super().__init__()
        if bilinear:
            assert 0
        else:
            if halve_channel:
                mid_channels = in1_channels//2
            else:
                mid_channels = in1_channels
            self.up   = nn.ConvTranspose2d(in1_channels, mid_channels, kernel_size=2, stride=2)
            self.conv = ResBlock(num_convs, mid_channels+in2_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class MonoTrack(nn.Module):
    '''
    https://arxiv.org/abs/2204.01899
    '''
    def __init__(self, n_channels, n_classes, bilinear=False, halve_channel=False):
        super(MonoTrack, self).__init__()
        self.n_channels    = n_channels
        self.n_classes     = n_classes
        self.bilinear      = bilinear
        self.halve_channel = halve_channel
        self.inc   = Neck(n_channels, 32)
        self.down1 = Down(2, 32, 64)
        self.down2 = Down(3, 64, 128)
        self.down3 = Down(3, 128, 256)
        self.up1   = Up(3, 256, 128, 128, bilinear=bilinear, halve_channel=halve_channel)
        self.up2   = Up(2, 128, 64, 64, bilinear=bilinear, halve_channel=halve_channel)
        self.up3   = Up(2, 64, 32, 32, bilinear=bilinear, halve_channel=halve_channel)
        self.outc  = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x  = self.up1(x4, x3)
        x  = self.up2(x, x2)
        x  = self.up3(x, x1)
        logits = self.outc(x)
        return {0: logits}

