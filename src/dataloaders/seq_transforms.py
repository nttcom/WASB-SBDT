import random
from collections import defaultdict
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T

class SeqTransformCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, tar):
        for t in self.transforms:
            img, tar = t(img, tar)
        return img, tar

class RandomHorizontalFlipping(object):
    """
    With a probability, flip images and heatmaps horizontally.
    Args:
        p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self, p=0.5):
        self._p    = p
        self._dims = [2] # axes to flip on

    def __call__(self, imgs, hms):
        """
        Args:
            imgs (torch tensor): images to be flipped. (C, H, W)
            hms (torch tensor): heatmaps to be flipped. (C, H, W)
        Returns:
            torch tensor: flipped images (C, H, W)
            torch tensor: flipped heatmaps (C, H, W)
        """
        #print(imgs.shape, hms.shape)
        if random.uniform(0, 1) >= self._p:
            return imgs, hms
        
        imgs_f = torch.flip(imgs, self._dims)
        hms_f  = {}
        #print(imgs.shape)
        for scale in hms.keys():
            hms_f[scale] = torch.flip(hms[scale], self._dims)
        return imgs_f, hms_f


class RandomCropping(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.
    Args:
        p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self, p=0.5, max_rescale=0.1, interpolation=T.InterpolationMode.BILINEAR):
        self._p             = p
        self._interpolation = interpolation
        self._max_rescale   = max_rescale

    def __call__(self, imgs, hms):
        """
        Args:
            imgs (pytorch tensor): images to be cropped.
            hms (pytorch tensor): heatmaps to be cropped.
        Returns:
            pytorch tensor: cropped images.
            pytorch tensor: cropped heatmaps.
        """
        if random.uniform(0, 1) >= self._p:
            return imgs, hms
        
        rescale = 1. + self._max_rescale

        # imgs
        _,img_h,img_w        = imgs.shape
        new_img_w, new_img_h = int(img_w * rescale), int(img_h * rescale)
        r_imgs               = F.resize(img=imgs, size=(new_img_h, new_img_w), interpolation=self._interpolation)
        
        r_hms = {}
        max_ratio = -1
        for scale in hms.keys():
            # hms
            _,hm_h,hm_w        = hms[scale].shape
            new_hm_w, new_hm_h = int(hm_w * rescale), int(hm_h * rescale)
            r_hms[scale]       = {'hm': F.resize(img=hms[scale], size=(new_hm_h, new_hm_w), interpolation=self._interpolation), 'h': hm_h, 'w': hm_w}

            # assume imgs is larger than hms
            if img_h % hm_h != 0:
                raise ValueError('hms size is not a product of an integer of imgs size')
            ratio = img_h / hm_h

            if max_ratio < ratio:
                max_ratio    = ratio
                min_new_hm_w = new_hm_w
                min_new_hm_h = new_hm_h
                min_hm_w     = hm_w
                min_hm_h     = hm_h

        x_maxrange_hm = min_new_hm_w - min_hm_w
        y_maxrange_hm = min_new_hm_h - min_hm_h
        x1 = int(round(random.uniform(0, x_maxrange_hm)))
        y1 = int(round(random.uniform(0, y_maxrange_hm)))
        
        rc_hms = {}
        for scale in r_hms.keys():
            hm_h, hm_w            = r_hms[scale]['h'], r_hms[scale]['w']
            _, new_hm_h, new_hm_w = r_hms[scale]['hm'].shape
            if new_hm_h % min_new_hm_h != 0:
                raise ValueError('hms size is not a product of an integer of imgs size')
            ratio = new_hm_h / min_new_hm_h
            rc_hms[scale] = F.crop(r_hms[scale]['hm'], top=int(y1*ratio), left=int(x1*ratio), width=hm_w, height=hm_h)

        if new_img_h % min_new_hm_h != 0:
            raise ValueError('hms size is not a product of an integer of imgs size')
        ratio   = new_img_h / min_new_hm_h
        rc_imgs = F.crop(r_imgs, top=int(y1*ratio), left=int(x1*ratio), width=img_w, height=img_h)

        return rc_imgs, rc_hms

