import os
import os.path as osp
import errno
import shutil
import random
import math
import numpy as np
from PIL import Image
import cv2
import torch
from torch import nn

def compute_l2_dist_mat(X, Y, axis=1):
    if X.shape[axis]!=Y.shape[axis]:
        raise RuntimeError('feat dims are different between matrices')
    X2 = np.sum(X**2, axis=1) # shape of (m)
    Y2 = np.sum(Y**2, axis=1) # shape of (n)
    XY = np.matmul(X, Y.T)
    X2 = X2.reshape(-1, 1)
    return np.sqrt(X2 - 2*XY + Y2)

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def list2txt(list_data):
    txt = ''
    for cnt, d in enumerate(list_data):
        txt += '{}'.format(d)
        if cnt<len(list_data)-1:
            txt += '-'
    # print(txt)
    return txt

def count_params(model, only_trainable=True):
    if only_trainable:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        num_params = sum(p.numel() for p in model.parameters() )
    return num_params

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def save_checkpoint(state, 
                    is_best, 
                    model_path,
                    best_model_name = 'best_model.pth.tar',
):
    mkdir_if_missing(osp.dirname(model_path))
    torch.save(state, model_path)
    if is_best:
        shutil.copy(model_path, osp.join(osp.dirname(model_path), best_model_name))

def set_seed(seed=None):
    if seed is None:
        return
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = ("%s" % seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _top1(scores):
    batch, seq, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, seq, -1), 1)
    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_ys, topk_xs

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

