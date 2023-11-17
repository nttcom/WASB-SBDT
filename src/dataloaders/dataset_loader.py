import os.path as osp
import json
import logging
import random
from collections import defaultdict
import numpy as np
import functools
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from utils import read_image
from utils.image import get_affine_transform, affine_transform
from .heatmaps import select_heatmap_generator

log = logging.getLogger(__name__)

def get_transform(img, input_wh, inv=0):
    h,w,_ = img.shape
    c     = np.array([w / 2., h / 2.], dtype=np.float32)
    s     = max(h, w) * 1.0
    input_w, input_h = input_wh
    trans = get_affine_transform(c, s, 0, [input_w, input_h], inv=inv)
    return trans 

def get_color_jitter_factors(brightness, contrast, saturation, hue):
    brightness_factor = np.random.uniform(max(0,1-brightness), 1+brightness)
    contrast_factor   = np.random.uniform(max(0,1-contrast), 1+contrast)
    saturation_factor = np.random.uniform(max(0,1-saturation), 1+saturation)
    hue_factor        = np.random.uniform(-hue, hue)
    return brightness_factor, contrast_factor, saturation_factor, hue_factor

class ImageDataset(Dataset):
    def __init__(self, 
                 cfg, 
                 dataset, 
                 input_wh, 
                 output_wh=None, 
                 transform=None, 
                 seq_transform=None, 
                 is_train=True,
                 fp1_fpath=None,
    ):
        self._dataset       = dataset
        self._transform     = transform
        self._seq_transform = seq_transform
        self._input_wh      = input_wh
        self._output_wh     = input_wh if output_wh is None else output_wh

        self._fp1_fpath = fp1_fpath

        self._hm_generator = select_heatmap_generator(cfg['dataloader']['heatmap'])

        self._is_train      = is_train

        if is_train:
            self._color_jitter_p          = cfg['transform']['train']['color_jitter']['p']
            self._color_jitter_brightness = cfg['transform']['train']['color_jitter']['brightness']
            self._color_jitter_contrast   = cfg['transform']['train']['color_jitter']['contrast']
            self._color_jitter_saturation = cfg['transform']['train']['color_jitter']['saturation']
            self._color_jitter_hue        = cfg['transform']['train']['color_jitter']['hue']
        else:
            self._color_jitter_p          = cfg['transform']['test']['color_jitter']['p']
            self._color_jitter_brightness = cfg['transform']['test']['color_jitter']['brightness']
            self._color_jitter_contrast   = cfg['transform']['test']['color_jitter']['contrast']
            self._color_jitter_saturation = cfg['transform']['test']['color_jitter']['saturation']
            self._color_jitter_hue        = cfg['transform']['test']['color_jitter']['hue']

        self._rgb_diff   = cfg['model']['rgb_diff']
        self._frames_in  = cfg['model']['frames_in']
        self._out_scales = cfg['model']['out_scales']
        if len(self._out_scales)!=1:
            assert 0
        if self._rgb_diff and self._frames_in!=2:
            raise ValueError('rgb_diff=True supported only with frames_in=2')

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        
        fp1_im_list = []
        if self._fp1_fpath is not None:
            if osp.exists(self._fp1_fpath):
                with open(self._fp1_fpath, 'r') as f:
                    data = f.read().split('\n')
                    fp1_im_list = data[:-1]

        img_paths = self._dataset[index]['frames']
        annos     = self._dataset[index]['annos']
    
        img_paths_out = []
        for anno in annos:
            img_paths_out.append( anno['frame_path'])

        input_w, input_h   = self._input_wh
        output_w, output_h = self._output_wh

        trans_input      = None
        trans_input_inv  = None
        trans_outputs     = defaultdict(list)
        trans_outputs_inv = defaultdict(list)

        apply_color_jitter = True
        if random.uniform(0,1) >= self._color_jitter_p:
            apply_color_jitter = False
        else:
            brightness_factor, contrast_factor, saturation_factor, hue_factor = get_color_jitter_factors(self._color_jitter_brightness, self._color_jitter_contrast, self._color_jitter_saturation, self._color_jitter_hue)

        imgs, xys, visis = [], [], []
        hms = defaultdict(list)

        for idx, img_path in enumerate(img_paths):
            img = read_image(img_path)
            
            if trans_input is None:
                trans_input  = get_transform(np.asarray(img), self._input_wh)
                out_w, out_h = self._output_wh
                for scale in self._out_scales:
                    trans_outputs[scale] = get_transform(np.asarray(img), (out_w, out_h))
                    out_w = out_w // 2
                    out_h = out_h // 2

                if not self._is_train:
                    trans_input_inv = get_transform(np.asarray(img), self._input_wh, inv=1)
                    out_w, out_h    = self._output_wh
                    for scale in self._out_scales:
                        trans_outputs_inv[scale] = get_transform(np.asarray(img), (out_w, out_h), inv=1)
                        out_w = out_w // 2
                        out_h = out_h // 2

            img = Image.fromarray( cv2.warpAffine(np.array(img), trans_input, self._input_wh, flags=cv2.INTER_LINEAR) )
            imgs.append(img)

        # hms (targets)
        for idx, (img_path, anno) in enumerate(zip(img_paths, annos)):
            binary = True
            if img_path in fp1_im_list:
                binary = False

            px, py = anno['center'].xy
            visi   = anno['center'].is_visible

            xys.append([px, py])
            visis.append(visi)
            out_w, out_h = self._output_wh
            for scale in self._out_scales:
                if visi:
                    ct     = affine_transform(np.array([px,py]), trans_outputs[scale])
                    ct_int = ct.astype(np.int32)
                    hm     = self._hm_generator((out_w,out_h), ct_int, binary=binary)
                else:
                    hm     = self._hm_generator((out_w,out_h), (-1.,-1.))

                hm = np.expand_dims(hm, axis=0)
                hms[scale].append(hm)
                out_w = out_w // 2
                out_h = out_h // 2

        imgs_t = []
        hms_t  = defaultdict(list)
        for img in imgs:
            # color gitter (data augmentation)
            if apply_color_jitter:
                img = TF.adjust_brightness(img, brightness_factor)
                img = TF.adjust_contrast(img, contrast_factor)
                img = TF.adjust_saturation(img, saturation_factor)
                img = TF.adjust_hue(img, hue_factor)
            img_t = self._transform(img)
            imgs_t.append(img_t)
        for scale in self._out_scales:
            for hm in hms[scale]:
                hm_t = torch.tensor(hm)
                hms_t[scale].append(hm_t)
        
        if self._rgb_diff:
            if len(imgs_t)!=2:
                raise ValueError('assume 2 images are input but {} given'.format(len(imgs_t)))
            imgs_t[0] = torch.abs(imgs_t[1] - imgs_t[0])

        imgs_t = torch.cat(imgs_t, dim=0)
        for scale in self._out_scales:
            hms_t[scale] = torch.cat(hms_t[scale], dim=0)
        if self._seq_transform is not None:
            imgs_t, hms_t = self._seq_transform(imgs_t, hms_t)
        xys   = torch.tensor(xys)
        visis = torch.tensor(visis)
        if self._is_train:
            return imgs_t, hms_t
        else:
            return imgs_t, hms_t, trans_outputs_inv, xys, visis, img_paths_out


