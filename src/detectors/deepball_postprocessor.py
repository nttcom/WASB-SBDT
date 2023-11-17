from collections import defaultdict
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from utils.utils import _nms, _top1
from utils.image import get_affine_transform, affine_transform

class DeepBallPostprocessor(object):
    def __init__(self, cfg):
        #print(cfg['detector']['postprocessor'])
        self._score_threshold = cfg['detector']['postprocessor']['score_threshold']
        self._scales = cfg['detector']['postprocessor']['scales']
        if len(self._scales) > 1 or self._scales[0]!=0:
            raise ValueError('only scales=[0] is supported')
        self._model_name = cfg['model']['name']
        if self._model_name!='deepball':
            assert 0, 'model: {} not supported for now (2022.8.22)'.format(self._model_name)
        self._foreground_channel = cfg['model']['foreground_channel']
        #print(self._score_threshold)

    """
    def run(self, preds):
        #print(preds.shape)
        preds = F.softmax(preds, dim=1)
        hms   = preds[:,self._foreground_channel:self._foreground_channel+1,:,:]
        hms   = hms.cpu().numpy()

        b,s,h,w = hms.shape
        #print(b,s,h,w)

        visis_all = []
        xys_all   = []
        for i in range(b):
            visis = []
            xys   = []
            for j in range(s):
                #print(i,j)
                hm   = hms[i,j]
                xy   = np.array([0., 0.])
                visi = False
                if np.max(hm) > self._score_threshold:
                    visi = True

                    y,x  = np.unravel_index(np.argmax(hm), hm.shape)
                    xy   = np.array([x,y])

                visis.append(visi)
                xys.append(xy)
            visis_all.append(visis)
            xys_all.append(xys)
        #print(visis_all)
        #print(xys_all)   
        #return xys_all, visis_all
        return np.array(xys_all), np.array(visis_all)
    """

    def _detect_peak(self, hm):
        xys    = []
        scores = []
        if np.max(hm) > self._score_threshold:
            y,x   = np.unravel_index(np.argmax(hm), hm.shape)
            xy    = np.array([x,y])
            score = hm[y,x]
            #print(np.max(hm), hm[y,x], score)
            xys.append(xy)
            scores.append(score)
        return xys, scores

    def run(self, preds, affine_mats):
        results = defaultdict(lambda: defaultdict(dict))
        for scale in self._scales:
            preds_       = preds[scale]
            affine_mats_ = affine_mats[scale].cpu().numpy()
            #hms_         = preds_.sigmoid_().cpu().numpy()
            preds_ = F.softmax(preds_, dim=1)
            hms_   = preds_[:,self._foreground_channel:self._foreground_channel+1,:,:]
            hms_   = hms_.cpu().numpy()

            b,s,h,w = hms_.shape
            for i in range(b):
                for j in range(s):
                    #print(i,j)
                    #print(hms_[i,j].shape)
                    xys_, scores_ = self._detect_peak(hms_[i,j])
                    #print(xys_, scores_)

                    """
                    if self._blob_det_method=='concomp':
                        xys_, scores_ = self._detect_blob_concomp(hms_[i,j])
                    elif self._blob_det_method=='nms':
                        xys_, scores_ = self._detect_blob_nms(hms_[i,j], self._sigmas[scale])
                    else:
                        raise ValueError('undefined xy_comp_method: {}'.format(self._xy_comp_method))
                    """
                    xys_t_ = []
                    for xy_ in xys_:
                        xys_t_.append( affine_transform(xy_, affine_mats_[i]))
                    #results[i][j][scale] = {'xy': xy_, 'visi': visi_, 'blob_size': blob_size_}
                    #results[i][j][scale] = {'xy': xy_, 'visi': visi_, 'blob_score': blob_score_}
                    results[i][j][scale] = {'xys': xys_t_, 'scores': scores_, 'hm': hms_[i,j], 'trans': affine_mats_[i]}

        #print(results)
        return results

