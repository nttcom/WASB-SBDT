import os
import os.path as osp
import logging
from omegaconf import DictConfig
import cv2

from .base import BaseRunner
from utils import mkdir_if_missing

log = logging.getLogger(__name__)

def extract_frame_badminton(cfg):
    root_dir      = cfg['dataset']['root_dir']
    video_dirname = cfg['dataset']['video_dirname']
    frame_dirname = cfg['dataset']['frame_dirname']
    train_matches = cfg['dataset']['train']['matches']
    test_matches  = cfg['dataset']['test']['matches']
    overwrite     = cfg['runner']['overwrite']

    matches = train_matches + test_matches
    for match in matches:
        match_video_dir = osp.join(root_dir, match, video_dirname)
        video_names = os.listdir(match_video_dir)
        video_names.sort()
        for video_name in video_names:
            video_path = osp.join(match_video_dir, video_name)
            frame_dir  = osp.join(root_dir, match, frame_dirname, osp.splitext(video_name)[0])
            if osp.exists(frame_dir) and not overwrite:
                log.info('{} already exists. skip extracting frames'.format(frame_dir))
                continue

            log.info('extract frames in {} to {}'.format(video_path, frame_dir))
            mkdir_if_missing(frame_dir)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                assert 0, '{} cannot opened'.format(video_path)
            cnt = 0
            while True:
                ret, frame = cap.read()
                if ret:
                    frame_path = osp.join(frame_dir, '{:05d}.png'.format(cnt))
                    cv2.imwrite(frame_path, frame)
                    cnt+=1
                else:
                    break

def extract_frame_soccer(cfg: DictConfig):
    root_dir      = cfg['dataset']['root_dir']
    video_dirname = cfg['dataset']['video_dirname']
    frame_dirname = cfg['dataset']['frame_dirname']
    train_videos  = cfg['dataset']['train']['videos']
    test_videos   = cfg['dataset']['test']['videos']
    img_ext       = cfg['dataset']['img_ext']
    video_ext     = cfg['dataset']['video_ext']
    overwrite     = cfg['runner']['overwrite']
    
    videos = train_videos + test_videos
    for video in videos:
        video_path = osp.join(root_dir, video_dirname, '{}{}'.format(video, video_ext) )
        frame_dir  = osp.join(root_dir, frame_dirname, video )
        if osp.exists(frame_dir) and not overwrite:
            log.info('{} already exists. skip extracting frames'.format(frame_dir))
            continue

        log.info('extract frames in {} to {}'.format(video_path, frame_dir))
        mkdir_if_missing(frame_dir)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            assert 0, '{} cannot opened'.format(video_path)
        cnt = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = osp.join(frame_dir, '{:05d}{}'.format(cnt, img_ext))
            cv2.imwrite(frame_path, frame)
            cnt+=1

def extract_frame(cfg: DictConfig):
    dataset_name = cfg['dataset']['name']
    if dataset_name=='badminton':
        extract_frame_badminton(cfg)
    elif dataset_name=='soccer':
        extract_frame_soccer(cfg)
    else:
        raise KeyError('for this dataset extrac_frame is not defined : {}'.format(dataset_name))

class ExtractFrameRunner(BaseRunner):
    def __init__(self,
                 cfg: DictConfig,
    ):
        super().__init__(cfg)
        self._dataset_name = cfg['dataset']['name']
        if not self._dataset_name in ['badminton', 'soccer']:
            raise KeyError('{} does not require frame extraction : {}'.format(dataset_name))
        
    def run(self):
        if self._dataset_name=='badminton':
            extract_frame_badminton(self._cfg)
        elif self._dataset_name=='soccer':
            extract_frame_soccer(self._cfg)
        else:
            raise KeyError('for this dataset extrac_frame is not defined : {}'.format(dataset_name))

