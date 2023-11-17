# https://github.com/jjjkkkjjj/pytorch_SSD/blob/master/ssd/train/loss.py
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/ssd/model.py

from torch import nn
import torch
import math
import torch.nn.functional as F

class SSDLoss(nn.Module):
    def __init__(self, auto_weight=False, scales=[0], neg_factor=3, hnm_batch=True, background_dim=0, alpha=1, loc_loss=None, conf_loss=None):
        super().__init__()
        self._auto_weight = auto_weight
        self._scales      = scales
        self._ws          = {}
        if self._auto_weight:
            raise ValueError('auto_weight=True is not supported')
        if len(self._scales) > 1 or self._scales[0]!=0:
            raise ValueError('only scales=[0] is supported')
        self._neg_factor     = neg_factor
        self._hnm_batch      = hnm_batch
        self._background_dim = background_dim
        #print(self._auto_weight, self._scales, self._neg_factor, self._hnm_batch, self._background_dim)
        #self.alpha = alpha
        #self.loc_loss = LocalizationLoss() if loc_loss is None else loc_loss
        #self.conf_loss = ConfidenceLoss() if conf_loss is None else conf_loss
        self.conf_loss = ConfidenceLoss(neg_factor=neg_factor, hnm_batch=hnm_batch, background_dim=background_dim)

    def forward(self, predicts, targets):
        """
        :param pos_indicator: Bool Tensor, shape = (batch, default box num). this represents whether each default box is object or background.
        :param predicts: Tensor, shape is (batch, total_dbox_nums, 4+class_labels=(cx, cy, w, h, p_class,...)
        :param targets: Tensor, shape is (batch, total_dbox_nums, 4+class_labels=(cx, cy, w, h, p_class,...)
        :return:
            loss: float
        """
        #if predicts.shape[1]!=2:
        #    raise ValueError('predicts must be (B,2,H,W)')
        #if targets.shape[1]!=1:
        #    raise ValueError('targets must be (B,1,H,W). dim 1 must be 0 (background) or 1 (foreground)')
        loss = 0
        for scale in self._scales:
            predicts_ = predicts[scale]
            targets_  = targets[scale]

            b,c,h,w   = predicts_.shape
            predicts_ = predicts_.view(b,c,-1)

            targets_ = targets_.squeeze(dim=1)
            b,h,w    = targets_.shape
            targets_ = targets_.view(b,-1).long()
            #targets = targets.squeeze(dim=1)
            #targets = targets.squeeze(dim=1).long()

            #print(predicts.shape, targets.shape)
            # Confidence loss
            loss += self.conf_loss(predicts_, targets_)
        return loss

class ConfidenceLoss(nn.Module):
    def __init__(self, neg_factor=3, hnm_batch=True, background_dim=0):
        """
        :param neg_factor: int, the ratio(1(pos): neg_factor) to learn pos and neg for hard negative mining
        :param hnm_batch: bool, whether to do hard negative mining for each batch
        """
        super().__init__()
        self._neg_factor     = neg_factor
        self._hnm_batch      = hnm_batch
        self._background_dim = background_dim
        #print(self._neg_factor, self._hnm_batch, self._background_dim)
        #self.loss = nn.CrossEntropyLoss()
        #self.loss = nn.BCEWithLogitsLoss()

    def forward(self, predicts, targets):
        if self._hnm_batch:
            #print(predicts.shape, targets.shape)
            mask    = targets > 0
            pos_num = mask.sum(dim=1)
            #print(pos_num)

            con     = F.cross_entropy(predicts, targets, reduction='none')
            con_neg = con.clone()
            #con_neg = con.detach()
            con_neg[mask] = 0
            _, con_idx  = con_neg.sort(dim=1, descending=True)
            _, con_rank = con_idx.sort(dim=1)

            neg_num  = torch.clamp(self._neg_factor * pos_num, max=mask.size(1)).unsqueeze(-1)
            neg_mask = con_rank < neg_num

            closs    = (con*((mask+neg_mask).float())).sum(dim=1)
            num_mask = (pos_num > 0).float()
            pos_num  = pos_num.float().clamp(min=1e-6)
            ret      = (closs*num_mask/pos_num).mean(dim=0)
            return ret

        else:
            assert 0, 'Not provided (2022.10.1)'
            #print(predicts.shape, targets.shape)            
            #con = F.cross_entropy(predicts, targets, reduction='mean')
            con = self.loss(predicts, targets)
            #print(con)
            return con

