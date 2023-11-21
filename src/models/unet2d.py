# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

from .unet2d_parts import *

class TrackNetV2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, mode='nearest', halve_channel=False):
        super(TrackNetV2, self).__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.bilinear   = bilinear
        self.inc   = DoubleConv(n_channels, 64, bn_first=False)
        self.down1 = Down(2, 64, 128, bn_first=False)
        self.down2 = Down(3, 128, 256, bn_first=False)
        self.down3 = Down(3, 256, 512, bn_first=False)
        self.up1   = Up(3, 512, 256, 256, bilinear=bilinear, mode=mode, bn_first=False, halve_channel=halve_channel)
        self.up2   = Up(2, 256, 128, 128, bilinear=bilinear, mode=mode, bn_first=False, halve_channel=halve_channel)
        self.up3   = Up(2, 128, 64, 64, bilinear=bilinear, mode=mode, bn_first=False, halve_channel=halve_channel)
        self.outc  = OutConv(64, n_classes)

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

