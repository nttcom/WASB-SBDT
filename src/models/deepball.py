import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, conv_bias=True):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=conv_bias),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True) )
    def forward(self, x):
        return self.conv(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels_list, out_channels_list, kernel_size_list, stride_list, padding_list, with_maxpool=True):
        super().__init__()
        module_dict = OrderedDict()
        for i, (in_channels, out_channels, kernel_size, stride, padding) in enumerate(zip(in_channels_list, out_channels_list, kernel_size_list, stride_list, padding_list)):
            #print('in_channels: {}'.format(in_channels))
            module_dict['conv{}'.format(i+1)] = ConvBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        if with_maxpool:
            module_dict['maxpool1'] = nn.MaxPool2d(2)

        self.module = nn.Sequential(module_dict)
    def forward(self, x):
        y = self.module(x)
        return y

class DeepBall(nn.Module):
    '''
    https://arxiv.org/abs/1902.07304
    '''
    def __init__(self, 
                 n_channels, 
                 n_classes,  
                 bilinear=False,
                 block_channels=[8, 16, 32],
                 block_maxpools=[True, True, True],
                 first_conv_kernel_size=7, 
                 last_conv_kernel_size=3, 
                 first_conv_stride=2,
                 ):
        super(DeepBall, self).__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.bilinear   = bilinear

        if first_conv_kernel_size==7:
            kernel_size_list = [7,3]
            padding_list     = [3,1]
        elif first_conv_kernel_size==3:
            kernel_size_list = [3,3]
            padding_list     = [1,1]
        else:
            raise ValueError('first_conv_kernel_size is 3 or 7')
        self.conv1      = ConvBlock( in_channels_list=[n_channels, block_channels[0]], 
                                     out_channels_list=[block_channels[0], block_channels[0]], 
                                     kernel_size_list=kernel_size_list, stride_list=[first_conv_stride, 1], padding_list=padding_list, with_maxpool=block_maxpools[0])
        self.conv2      = ConvBlock( in_channels_list=[block_channels[0], block_channels[1]], 
                                     out_channels_list=[block_channels[1], block_channels[1]], 
                                     kernel_size_list=[3, 3], stride_list=[1, 1], padding_list=[1, 1], with_maxpool=block_maxpools[1])
        self.conv3      = ConvBlock( in_channels_list=[block_channels[1], block_channels[2]], 
                                     out_channels_list=[block_channels[2], block_channels[2]], 
                                     kernel_size_list=[3, 3], stride_list=[1, 1], padding_list=[1, 1], with_maxpool=block_maxpools[2])
        if bilinear:
            assert 0, 'using nn.Upsample results in non-reproducible results. cf. https://discuss.pytorch.org/t/non-deterministic-behavior-of-pytorch-upsample-interpolate/42842'
            self.upsample2 = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
            self.upsample3 = nn.Upsample(scale_factor=4, mode=mode, align_corners=True)
        else:
            self.upsample2 = nn.ConvTranspose2d(block_channels[1], block_channels[1], kernel_size=2, stride=2)
            self.upsample3 = nn.ConvTranspose2d(block_channels[2], block_channels[2], kernel_size=4, stride=4)

        channels_concat = block_channels[0] + block_channels[1] + block_channels[2]
        self.conv4_1    = ConvBlock( in_channels_list=[channels_concat], 
                                     out_channels_list=[channels_concat], 
                                     kernel_size_list=[3], stride_list=[1], padding_list=[1], with_maxpool=False)
        if last_conv_kernel_size ==3:
            padding = 1
        elif last_conv_kernel_size ==1:
            padding = 0
        else:
            raise ValueError('last_conv_kernel_size is 3 or 1')
        self.conv4_2 = nn.Conv2d( channels_concat, n_classes, last_conv_kernel_size, stride=1, padding=padding, bias=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x2 = self.upsample2(x2)
        x3 = self.upsample3(x3)
        x4 = torch.cat([x1,x2,x3], dim=1)
        x4 = self.conv4_1(x4)
        logits = self.conv4_2(x4)
        return {0: logits}

