from typing import Tuple
from PIL import Image
import numpy as np
import cv2

def gen_binary_map(wh: Tuple[int, int],
                   cxy: Tuple[float, float],
                   r: float,
                   data_type: np.dtype = np.float32,
):
    w, h   = wh
    cx, cy = cxy
    if cx < 0 or cy < 0:
        return np.zeros((h,w), dtype=data_type)
    x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    distmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
    bmap    = np.zeros_like(distmap)
    bmap[distmap <= r**2] = 1
    return bmap.astype(data_type)

def gen_heatmap(wh: Tuple[int, int],
                   cxy: Tuple[float, float],
                   r: float,
                   data_type: np.dtype = np.float32,
                   min_value: float = 0.7,
):
    w, h   = wh
    cx, cy = cxy
    if cx < 0 or cy < 0:
        return np.zeros((h,w), dtype=data_type)
    x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    distmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
    r2 = r**2
    heatmap = np.exp(-distmap/r2 ) / np.exp(-1.) * min_value
    heatmap[heatmap < 0.5] = 0
    heatmap[heatmap > 1] = 1.
    return heatmap.astype(data_type)

