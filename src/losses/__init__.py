from torch import nn

from .heatmap import HeatmapLoss
from .segmentation import SegmentationLoss

__factory = {
        'heatmap': HeatmapLoss,
        'segmentation': SegmentationLoss
        }

def build_loss_criteria(cfg):
    #print(cfg)
    #print( cfg['loss']['name'] )
    loss_name = cfg['loss']['name']

    if not loss_name in __factory.keys():
        raise KeyError('invalid loss: {}'.format(loss_name ))

    loss = __factory[loss_name](cfg)

    return loss

