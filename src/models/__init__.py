from .unet2d import TrackNetV2
from .resunet2d import ChangsTrackNet
from .monotrack import MonoTrack
from .hrnet import HRNet
from .deepball import DeepBall
from .ballseg import BallSeg

__factory = {
    'tracknetv2': TrackNetV2,
    'monotrack': MonoTrack,
    'restracknetv2': ChangsTrackNet,
    'hrnet': HRNet,
    'deepball': DeepBall,
    'ballseg': BallSeg
        }

def build_model(cfg):
    model_name = cfg['model']['name']
    if not model_name in __factory.keys():
        raise KeyError('invalid model: {}'.format(model_name ))
    if model_name=='tracknetv2' or model_name=='resunet2d' or model_name=='monotrack':
        frames_in  = cfg['model']['frames_in']
        frames_out = cfg['model']['frames_in']
        bilinear   = cfg['model']['bilinear']
        halve_channel = cfg['model']['halve_channel']
        model      = __factory[model_name]( frames_in*3, frames_out, bilinear=bilinear, halve_channel=halve_channel)
    elif model_name=='higher_hrnet' or model_name=='cls_hrnet' or model_name=='hrnet':
        model = __factory[model_name](cfg['model'])
    elif model_name=='restracknetv2':
        frames_in        = cfg['model']['frames_in']
        frames_out       = cfg['model']['frames_out']
        halve_channel    = cfg['model']['halve_channel']
        mode             = cfg['model']['mode']
        neck_channels    = cfg['model']['neck_channels']
        out_mid_channels = cfg['model']['out_mid_channels']
        blocks           = cfg['model']['blocks']
        channels         = cfg['model']['channels']
        model = __factory[model_name]( frames_in*3, 
                                       frames_out, 
                                       mode=mode, 
                                       halve_channel=halve_channel, 
                                       neck_channels=neck_channels, 
                                       out_mid_channels=out_mid_channels, 
                                       blocks=blocks, channels=channels)
    elif model_name=='deepball':
        frames_in              = cfg['model']['frames_in']
        frames_out             = cfg['model']['frames_out']
        class_out              = cfg['model']['class_out']
        block_channels         = cfg['model']['block_channels']
        block_maxpools         = cfg['model']['block_maxpools']
        first_conv_kernel_size = cfg['model']['first_conv_kernel_size']
        last_conv_kernel_size  = cfg['model']['last_conv_kernel_size']
        first_conv_stride      = cfg['model']['first_conv_stride']
        model = __factory[model_name]( frames_in*3, frames_out*class_out,
                                       block_channels=block_channels,
                                       block_maxpools=block_maxpools, 
                                       first_conv_kernel_size=first_conv_kernel_size, 
                                       first_conv_stride=first_conv_stride, 
                                       last_conv_kernel_size=last_conv_kernel_size)
    elif model_name=='ballseg':
        frames_in     = cfg['model']['frames_in']
        frames_out    = cfg['model']['frames_out']
        scale_factors = cfg['model']['scale_factors']
        backbone      = cfg['model']['backbone']
        model = __factory[model_name]( in_channels=frames_in*3, nclass=frames_out*1, backbone=backbone, scale_factors=scale_factors)

    return model

