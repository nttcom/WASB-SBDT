import os.path as osp
import logging
from collections import defaultdict
from hydra.core.hydra_config import HydraConfig
import numpy as np
from PIL import Image
import cv2
import torch
from torch import nn

from models import build_model
from dataloaders import read_image, get_transform, build_img_transforms
from utils.image import get_affine_transform, affine_transform
from .postprocessor import TracknetV2Postprocessor
from .deepball_postprocessor import DeepBallPostprocessor

log = logging.getLogger(__name__)

class TracknetV2Detector(object):
    
    __postprocessor_factory = {
            'tracknetv2': TracknetV2Postprocessor,
            'deepball': DeepBallPostprocessor,
            }

    def __init__(self, cfg, model=None):
        self._frames_in  = cfg['model']['frames_in']
        self._frames_out = cfg['model']['frames_out']
        self._input_wh   = (cfg['model']['inp_width'], cfg['model']['inp_height'])
        
        self._scales          = cfg['detector']['postprocessor']['scales']

        self._2d_input = True
        model_name = cfg['model']['name']
        if model_name in ['tracknetv2', 'resunet2d', 'hrnet', 'monotrack', 'changs', 'deepball', 'segball']:
            pass
        else:
            raise ValueError('unknown model_name : {}'.format(model_name))

        _, self._transform = build_img_transforms(cfg)

        self._device = cfg['runner']['device']
        if self._device!='cuda':
            assert 0, 'device=cpu not supported'
        if not torch.cuda.is_available():
            assert 0, 'GPU NOT available'
        self._gpus  = cfg['runner']['gpus']

        if model is None:
            self._model = build_model(cfg)
            model_path = cfg['detector']['model_path']
            if model_path is None:
                output_dir = HydraConfig.get().run.dir
                model_path = osp.join(output_dir, 'best_model.pth.tar')
                log.info('Checkpoint is not specified, so it is set as the best model in {}'.format(output_dir))
                if not osp.exists(model_path):
                    FileNotFoundError('{} not found'.format(model_path))
            checkpoint = torch.load(model_path)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model = self._model.to(self._device)
            self._model = nn.DataParallel(self._model, device_ids=self._gpus)
        else:
            self._model = model

        self._model.eval()

        postprocessor_name = cfg['detector']['postprocessor']['name']
        if not postprocessor_name in self.__postprocessor_factory.keys():
            raise KeyError('invalid dataset: {}'.format(postprocessor_name ))
        self._postprocessor = self.__postprocessor_factory[postprocessor_name](cfg)

    @property
    def frames_in(self):
        return self._frames_in
    
    @property
    def frames_out(self):
        return self._frames_out

    @property
    def input_wh(self):
        return self._input_wh

    def run_tensor(self, imgs, affine_mats):
        imgs  = imgs.to(self._device)
        preds = self._model(imgs)
        pp_results  = self._postprocessor.run(preds, affine_mats)

        results = {}
        hms_vis = {}
        for bid in sorted(pp_results.keys()):
            results[bid] = {}
            hms_vis[bid] = {}
            for eid in sorted(pp_results[bid].keys()):
                results[bid][eid] = []
                hms_vis[bid][eid] = []
                for scale in sorted(pp_results[bid][eid].keys()):
                    scores = pp_results[bid][eid][scale]['scores']
                    xys    = pp_results[bid][eid][scale]['xys']
                    for xy, score in zip(xys, scores):
                        results[bid][eid].append({ 'xy': xy, 'score': score, 'scale': scale})
                    
                    hm    = pp_results[bid][eid][scale]['hm']
                    trans = pp_results[bid][eid][scale]['trans']
                    hms_vis[bid][eid].append({'hm': hm, 'scale': scale, 'trans': trans})

        return results, hms_vis

