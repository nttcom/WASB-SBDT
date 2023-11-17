import logging
from omegaconf import DictConfig

from .samplers import ClipSampler, MatchSampler, RandomSampler

log = logging.getLogger(__name__)

__sampler_factory = {
    'clip'  : ClipSampler,
    'match' : MatchSampler,
    'random': RandomSampler,
        }

def select_sampler(cfg: DictConfig, 
                   dataset,
):
    #print(cfg)
    sampler_name = cfg['name']
    if not sampler_name in __sampler_factory.keys():
        raise KeyError('invalid sampler: {}'.format(sampler_name))
    if sampler_name=='clip':
        train_sampler = __sampler_factory[sampler_name](dataset.train,
                                                        cfg['train_batch_size'],
                                                        shuffle_within_clip=cfg['train_shuffle_within_clip'],
                                                        shuffle_batch=cfg['train_shuffle_batch'],
                                                        drop_last=cfg['train_drop_last'])
        test_sampler = __sampler_factory[sampler_name](dataset.test,
                                                       cfg['test_batch_size'],
                                                       shuffle_within_clip=cfg['test_shuffle_within_clip'],
                                                       shuffle_batch=cfg['test_shuffle_batch'],
                                                       drop_last=cfg['test_drop_last'])
    elif sampler_name=='match':
        train_sampler = __sampler_factory[sampler_name](dataset.train,
                                                        cfg['train_batch_size'],
                                                        shuffle_within_match=cfg['train_shuffle_within_match'],
                                                        shuffle_batch=cfg['train_shuffle_batch'],
                                                        drop_last=cfg['train_drop_last'])
        test_sampler = __sampler_factory[sampler_name](dataset.test,
                                                       cfg['test_batch_size'],
                                                       shuffle_within_match=cfg['test_shuffle_within_match'],
                                                       shuffle_batch=cfg['test_shuffle_batch'],
                                                       drop_last=cfg['test_drop_last'])
    elif sampler_name=='random':
        train_sampler = __sampler_factory[sampler_name](dataset.train,
                                                        cfg['train_batch_size'],
                                                        shuffle_batch=cfg['train_shuffle_batch'],
                                                        drop_last=cfg['train_drop_last'])
        test_sampler = __sampler_factory[sampler_name](dataset.test,
                                                       cfg['test_batch_size'],
                                                       shuffle_batch=cfg['test_shuffle_batch'],
                                                       drop_last=cfg['test_drop_last'])
    
    train_clips         = dataset.train_clips
    train_clip_samplers = {}
    for key, clip in train_clips.items():
        sampler = RandomSampler(clip,
                                cfg['inference_video_batch_size'],
                                shuffle_batch=cfg['inference_video_shuffle_batch'],
                                drop_last=cfg['inference_video_drop_last']
                                )
        train_clip_samplers[key] = sampler

    test_clips         = dataset.test_clips
    test_clip_samplers = {}
    for key, clip in test_clips.items():
        sampler = RandomSampler(clip,
                                cfg['inference_video_batch_size'],
                                shuffle_batch=cfg['inference_video_shuffle_batch'],
                                drop_last=cfg['inference_video_drop_last']
                                )
        test_clip_samplers[key] = sampler

    return train_sampler, test_sampler, train_clip_samplers, test_clip_samplers

