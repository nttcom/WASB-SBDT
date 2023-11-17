import logging
from omegaconf import DictConfig

from .badminton import Badminton
from .badminton import get_video_clips as get_badminton_clips
from .tennis import Tennis
from .tennis import get_clips as get_tennis_clips
from .volleyball import Volleyball
from .volleyball import get_clips as get_volleyball_clips
from .soccer import Soccer
from .soccer import get_clips as get_soccer_clips
from .basketball import Basketball
from .basketball import get_clips as get_basketball_clips

log = logging.getLogger(__name__)

__dataset_factory = {
    'soccer': Soccer,
    'tennis': Tennis,
    'badminton': Badminton,
    'volleyball': Volleyball,
    'basketball': Basketball,
        }

__video_clip_factory = {
    'badminton': get_badminton_clips,
    'tennis': get_tennis_clips,
    'volleyball': get_volleyball_clips,
    'soccer': get_soccer_clips,
    'basketball': get_basketball_clips,
        }

def select_dataset(
        cfg: DictConfig,
        
):
    dataset_name = cfg['dataset']['name']
    if not dataset_name in __dataset_factory.keys():
        raise KeyError('unknown dataset_name: {}'.format(dataset_name))
    return __dataset_factory[dataset_name](cfg)

def select_video_clips(
        cfg: DictConfig,
        targets,
):
    dataset_name = cfg['dataset']['name']
    if not dataset_name in __video_clip_factory.keys():
        raise KeyError('invalid dataset: {}'.format(dataset_name ))
    if len(targets)==0:
        raise ValueError('targets is empty')
    video_clips = {}
    for target in targets:
        video_clips.update( __video_clip_factory[dataset_name](cfg, train_or_test=target) )
    return video_clips

