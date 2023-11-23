from .detector import TracknetV2Detector
from .deepball_detector import DeepBallDetector

__factory = {
    'tracknetv2' : TracknetV2Detector,
    'deepball': DeepBallDetector
    }

def build_detector(cfg, model=None):
    detector_name = cfg['detector']['name'] 
    if not detector_name in __factory.keys():
        raise KeyError('invalid detector: {}'.format(detector_name ))
    return __factory[detector_name](cfg, model=model)

