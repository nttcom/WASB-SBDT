import os.path as osp
import logging
from hydra.core.hydra_config import HydraConfig
import numpy as np
from PIL import Image
import cv2
import torch
from torch import nn

from models import build_model
from dataloaders import read_image, get_transform, build_img_transforms
from utils.image import get_affine_transform, affine_transform
#from .postprocessor import TracknetV2Postprocessor
from .deepball_postprocessor import DeepBallPostprocessor

log = logging.getLogger(__name__)

class DeepBallDetector(object):
    __supported_2d_models   = ['deepball']
    __supported_3d_models   = []
    __postprocessor_factory = {
            'deepball': DeepBallPostprocessor
            }

    def __init__(self, cfg, model=None):
        self._2d_input = True
        model_name = cfg['model']['name'] 
        if model_name in self.__supported_2d_models:
            pass
        elif model_name in self.__supported_3d_models:
            self._2d_input = False
        else:
            raise ValueError('unsupported model_name : {}'.format(model_name))

        self._frames_in   = cfg['model']['frames_in']
        self._frames_out  = cfg['model']['frames_out']
        self._input_wh    = (cfg['model']['inp_width'], cfg['model']['inp_height'])
        self._classes_out = cfg['model']['class_out']
        if self._classes_out!=2 or self._frames_out!=1:
            raise ValueError('classes_out must be 2 (backgroun & foreground) and frames_out must be 1')
        #print(self._frames_out, self._classes_out)
        
        """
        self._2d_input = True
        model_name = cfg['model']['name']
        if model_name in ['unet2d', 'tracknetv2', 'resunet2d', 'swinunet2d', 'hrnet', 'monotrack', 'changs', 'deepball', 'segball']:
            pass
        elif model_name in ['unet3d', 'resunet3d']:
            self._2d_input = False
        else:
            raise ValueError('unknown model_name : {}'.format(model_name))
        """
        
        _, self._transform = build_img_transforms(cfg)

        self._device = cfg['device']
        if self._device!='cuda':
            assert 0, 'device=cpu not supported'
        if not torch.cuda.is_available():
            assert 0, 'GPU NOT available'
        self._gpus  = cfg['gpus']
        #print(self._device, self._gpus)

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
            #self._model_epoch = checkpoint['epoch']
            self._model = self._model.to(self._device)
            self._model = nn.DataParallel(self._model, device_ids=self._gpus)
            log.info('model is destributed to gpus {}'.format(self._gpus))
        else:
            self._model = model

        self._model.eval()

        postprocessor_name = cfg['detector']['postprocessor']['name']
        #print(self.__postprocessor_factory)
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
        #print(preds.shape)
        xys, visis  = self._postprocessor.run(preds)
        affine_mats = affine_mats.numpy()
        batch_size  = xys.shape[0]
        xys_t       = []
        for b in range(batch_size):
            xys_        = xys[b]
            affine_mat_ = affine_mats[b]
            xys_t_      = []
            for xy_ in xys_:
                #print(xy_, affine_mat_)
                xy_ = affine_transform(xy_, affine_mat_)
                #print(xy_)
                xys_t_.append(xy_)
            xys_t.append(xys_t_)
        return np.array(xys_t), visis

