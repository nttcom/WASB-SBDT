from typing import Tuple
import numpy as np

from utils import gen_binary_map, gen_heatmap

class BinaryFixedSizeMapGenerator:
    def __init__(self, cfg):
        self._sigma     = cfg['sigmas'][0]
        self._data_type = np.float32
        self._min_value = cfg['min_value']
        
    def __call__(self, 
                 wh: Tuple[int, int],
                 cxy: Tuple[float, float],
                 binary: bool = True,
                 ):
        if binary:
            return gen_binary_map(wh, cxy, self._sigma, self._data_type)
        else:
            return gen_heatmap(wh, cxy, self._sigma, self._data_type, min_value=self._min_value)

class PrototypeBasedBinaryMapGenerator:
    def __init__(self, cfg):
        npz_path =cfg['npz_path']
        if npz_path is None:
            raise ValueError('npz_path is mandatory')
        tmp = np.load(npz_path)
        self._centroids    = tmp['centroids']
        self._num_clusters = tmp['num_clusters']
        self._margin       = tmp['margin']
        self._num_data     = tmp['num_data']
        self._hms          = tmp['heatmaps']
        self._cxys         = tmp['cxys']
        self._sizes        = tmp['sizes']

    def __call__(self, w, h, cxy):
        pass


