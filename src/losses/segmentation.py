from torch import nn
from .ssd_loss import SSDLoss

class SegmentationLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        loss_name = cfg['loss']['sub_name']
        if loss_name=='ssd':
            auto_weight    = cfg['loss']['auto_weight']
            scales         = cfg['loss']['scales']
            neg_factor     = cfg['loss']['neg_factor']
            hnm_batch      = cfg['loss']['hnm_batch']
            background_dim = cfg['loss']['background_dim']
            self._loss     = SSDLoss(scales=scales, auto_weight=auto_weight, neg_factor=neg_factor, hnm_batch=hnm_batch, background_dim=background_dim)
        else:
            raise KeyError('invalid loss: {}'.format(loss_name ))

    def forward(self, inputs, targets):
        #print(inputs.shape, targets.shape)
        loss   = self._loss(inputs, targets)
        return loss

