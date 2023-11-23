"""Image Cascade Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel

"""
__all__ = ['ICNet', 'get_icnet', 'get_icnet_resnet50_citys',
           'get_icnet_resnet101_citys', 'get_icnet_resnet152_citys']
"""

class BallSeg(SegBaseModel):
    """Image Cascade Network""" 
    def __init__(self, nclass = 19, backbone='resnet50', pretrained_base=False, in_channels=3, scale_factors=[1, 1, 0.5]):
        super(BallSeg, self).__init__(nclass, in_channels=in_channels, backbone=backbone, pretrained_base=pretrained_base)
        
        self._scale_factors = scale_factors

        self.conv_sub1 = nn.Sequential(
            _ConvBNReLU(in_channels, 32, 3, 2),
            _ConvBNReLU(32, 32, 3, 2),
            _ConvBNReLU(32, 64, 3, 2)
        )
        
        self.ppm = PyramidPoolingModule()

        if backbone in ['resnet50', 'resnet101', 'resnet152']:
            nchannels_cff24 = 2048
        elif backbone in ['resnet18', 'resnet34']:
            nchannels_cff24 = 512
        else:
            raise ValueError('unsupported backbone: {}'.format(backbone))

        self.head = _ICHead(nclass, nchannels_cff24=nchannels_cff24, scale_factors=scale_factors)
        self.__setattr__('exclusive', ['conv_sub1', 'head'])
        
    def forward(self, x):
        # sub 1
        x_sub1 = self.conv_sub1(x)

        # sub 2
        x_sub2 = F.interpolate(x, scale_factor=self._scale_factors[1], mode='nearest')
        _, x_sub2, _, _ = self.base_forward(x_sub2)
        
        # sub 4
        x_sub4 = F.interpolate(x, scale_factor=self._scale_factors[2], mode='nearest')
        _, _, _, x_sub4 = self.base_forward(x_sub4)
        
        # add PyramidPoolingModule
        x_sub4 = self.ppm(x_sub4)
        
        outputs = self.head(x_sub1, x_sub2, x_sub4)
        return {0: outputs[0]}

class PyramidPoolingModule(nn.Module):
    def __init__(self, pyramids=[1,2,3,6]):
        super(PyramidPoolingModule, self).__init__()
        self.pyramids = pyramids

    def forward(self, input):
        feat = input
        height, width = input.shape[2:]
        for bin_size in self.pyramids:
            x = F.adaptive_avg_pool2d(input, output_size=bin_size)
            x = F.interpolate(x, size=(height, width), mode='nearest')
            feat  = feat + x
        return feat

class _ICHead(nn.Module):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, nchannels_cff24=512, scale_factors=[1.0, 1.0, 0.5], **kwargs):
        super(_ICHead, self).__init__()
        self.cff_12 = CascadeFeatureFusion(128, 64, 128, nclass, norm_layer, scale_factors[1], **kwargs)
        self.cff_24 = CascadeFeatureFusion(nchannels_cff24, nchannels_cff24//4, 128, nclass, norm_layer, scale_factors[2], **kwargs)

        self.conv_cls = nn.Conv2d(128, nclass, 1, bias=False)
        
        # added to remove F.interpolate
        self.deconv_x2 = nn.ConvTranspose2d(128,128,2,stride=2)
        self.deconv_x8 = nn.ConvTranspose2d(1,1,4,stride=4)

    def forward(self, x_sub1, x_sub2, x_sub4):
        outputs = list()
        x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2)
        outputs.append(x_24_cls)
        x_cff_12, x_12_cls = self.cff_12(x_cff_24, x_sub1)
        outputs.append(x_12_cls)

        up_x2 = self.deconv_x2(x_cff_12)
                
        up_x2 = self.conv_cls(up_x2)
        outputs.append(up_x2)
        up_x8 = self.deconv_x8(up_x2)
        
        outputs.append(up_x8)
        # 1 -> 1/4 -> 1/8 -> 1/16
        outputs.reverse()

        return outputs


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                 groups=1, norm_layer=nn.BatchNorm2d, bias=False, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CascadeFeatureFusion(nn.Module):
    """CFF Unit"""

    def __init__(self, low_channels, high_channels, out_channels, nclass, norm_layer=nn.BatchNorm2d, scale_factor=1.0, **kwargs):
        super(CascadeFeatureFusion, self).__init__()
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            norm_layer(out_channels)
        )
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            norm_layer(out_channels)
        )
        self.conv_low_cls = nn.Conv2d(out_channels, nclass, 1, bias=False)

    def forward(self, x_low, x_high): 
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='nearest')         
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x = x_low + x_high
        x = F.relu(x, inplace=True)
        x_low_cls = self.conv_low_cls(x_low)

        return x, x_low_cls

