import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

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
        return out2

class ResBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, conv_bias=True, stride=1, decoder=False):
        super().__init__()
        self.stride     = stride
        if not self.stride in [1,2]:
            assert ValueError('stride must be 1 or 2')
        self.conv0 = None
        if in_channels!=out_channels:
            self.conv0 = nn.Sequential(nn.MaxPool2d(self.stride),
                                       nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=conv_bias),
                                       nn.BatchNorm2d(out_channels) )
        self.conv1 = nn.Sequential(nn.BatchNorm2d(in_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=conv_bias),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=self.stride, padding=1, bias=conv_bias),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=conv_bias) )
    def forward(self, x):
        y        = self.conv1(x)
        identity = x
        if self.conv0 is not None:
            identity = self.conv0(x)
        #print(identity.shape, y.shape)
        return y + identity

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, blocks):
        super().__init__()
        module_dict = OrderedDict()
        module_dict['block0'] = ResBottleneck(in_channels, out_channels, stride=2)
        for i in range(blocks-1):
            module_dict['block{}'.format(i+1)] = ResBottleneck(out_channels, out_channels)
        self.convs = nn.Sequential(module_dict)
    def forward(self, x):
        return self.convs(x)

class ResBottleneckTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, conv_bias=True, mode='bilinear', kernel_size=2, stride=2):
        super().__init__()
        self.conv0 = nn.Sequential(nn.Upsample(scale_factor=kernel_size, mode=mode),
                                   nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=conv_bias),
                                   nn.BatchNorm2d(out_channels) )
        self.conv1 = nn.Sequential(nn.BatchNorm2d(in_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=conv_bias),
                                   nn.BatchNorm2d(in_channels),
                                   nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=conv_bias) )
    def forward(self, x):
        y        = self.conv1(x)
        identity = self.conv0(x)
        return y+identity

class Up(nn.Module):
    # Upscaling then double / triple conv
    def __init__(self, in1_channels, in2_channels, out_channels, blocks, mode='bilinear', halve_channel=False):
        super().__init__()
        ins, outs = [], []
        for i in range(blocks):
            if i==0:
                if halve_channel:
                    assert 0
                else:
                    ins.append(in1_channels+in2_channels)
                outs.append(out_channels)
            else:
                ins.append(out_channels)
                outs.append(out_channels)

        self.conv0  = ResBottleneckTranspose(in1_channels, in1_channels, mode=mode)
        module_dict = OrderedDict()
        for i, (in_c, out_c) in enumerate(zip(ins, outs)):
            module_dict['block{}'.format(i)] = ResBottleneck(in_c, out_c, decoder=True)
        self.convs = nn.Sequential(module_dict)

    def forward(self, x1, x2):
        x1 = self.conv0(x1)
        x = torch.cat([x2, x1], dim=1)  
        x = self.convs(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, conv_bias=True):
        super(OutConv, self).__init__()
        self.convs = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=conv_bias),
                                   nn.BatchNorm2d(mid_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=conv_bias),
                                   nn.BatchNorm2d(mid_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=conv_bias))

    def forward(self, x):
        return self.convs(x)

class ChangsTrackNet(nn.Module):
    '''
    https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2
    '''
    def __init__(self, n_channels, n_classes, mode='bilinear', halve_channel=False, neck_channels=64, out_mid_channels=64, blocks=[3, 3, 4, 3], channels=[16, 32, 64, 128]):
        super(ChangsTrackNet, self).__init__()
        self.n_channels    = n_channels
        self.n_classes     = n_classes
        self.halve_channel = halve_channel
        self.mode          = mode
        log.info('halve_channel: {}, mode: {}, neck_channels: {}, out_mid_channels: {}'.format(halve_channel, mode, neck_channels, out_mid_channels))
        self.inc   = Neck(n_channels, neck_channels)
        self.down1 = Down(neck_channels, channels[0], blocks[0])
        self.down2 = Down(channels[0], channels[1], blocks[1])
        self.down3 = Down(channels[1], channels[2], blocks[2])
        self.down4 = Down(channels[2], channels[3], blocks[3])

        self.up1   = Up(channels[3], channels[2], channels[2], blocks[2]-1, mode=mode)
        self.up2   = Up(channels[2], channels[1], channels[1], blocks[1]-1, mode=mode)
        self.up3   = Up(channels[1], channels[0], channels[0], blocks[0]-1, mode=mode)
        self.up4   = ResBottleneckTranspose(channels[0], channels[0], mode=mode)
        self.outc  = OutConv(channels[0], out_mid_channels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x, x3)
        x  = self.up3(x, x2)
        x = self.up4(x)
        logits = self.outc(x)
        return {0: logits}

