import logging
import copy
import random
from collections import defaultdict
from torch.utils.data.sampler import Sampler

log = logging.getLogger(__name__)

class RandomSampler(Sampler):
    '''
    every batch is build randomly
    '''
    _ret = []
    def __init__(self, dataset, batch_size=4, shuffle_batch=True, drop_last=True):
        self._batch_size           = batch_size
        self._shuffle_batch        = shuffle_batch
        self._drop_last            = drop_last
        #log.info('launch RandomSampler. batch size: {}, shuffle_batch: {}, drop_last: {}'.format(self._batch_size, self._shuffle_batch, self._drop_last))
        self._idxs = []
        for idx, d in enumerate(dataset):
            self._idxs.append(idx)
        self._length = len(self._idxs) // self._batch_size
        if (not self._drop_last) and (len(self._idxs)%self._batch_size !=0):
            self._length += 1
        #log.info('# of batches: {}'.format(self._length) )

    def __iter__(self):
        #print(self._idxs)
        #print(self._batch_size)
        ret = []
        if self._shuffle_batch:
            random.shuffle(self._idxs)
        for i in range(len(self._idxs)//self._batch_size):
            ret.append( self._idxs[i*self._batch_size:(i+1)*self._batch_size])
        if (not self._drop_last) and (len(self._idxs)%self._batch_size !=0):
            ret.append( self._idxs[(i+1)*self._batch_size:])
        return iter(ret)

    def __len__(self):
        return self._length

class MatchSampler(Sampler):
    '''
    every batch is build using data in the same match (but may be from different clips)
    '''
    _ret = []
    def __init__(self, dataset, batch_size=4, shuffle_within_match=True, shuffle_batch=True, drop_last=True):
        self._batch_size           = batch_size
        self._shuffle_batch        = shuffle_batch
        self._shuffle_within_match = shuffle_within_match
        self._drop_last            = drop_last
        log.info('launch MatchSampler. batch size: {}, shuffle_batch: {}, shuffle_within_match: {}, drop_last: {}'.format(self._batch_size, self._shuffle_batch, self._shuffle_within_match, self._drop_last))
        self._data_dict = defaultdict(list)
        for idx, d in enumerate(dataset):
            match = d['match']
            self._data_dict[match].append(idx)
        #print(self._data_dict)
        self._length = 0
        for key, idxs in self._data_dict.items():
            self._length += len(idxs) // self._batch_size
            if not self._drop_last:
                self._length += 1
        log.info('# of batches: {}'.format(self._length) )

    def __iter__(self):
        ret = []
        for key, idxs in self._data_dict.items():
            if self._shuffle_within_match:
                random.shuffle(idxs)
            for i in range(len(idxs)//self._batch_size):
                ret.append( idxs[i*self._batch_size:(i+1)*self._batch_size])
            if not self._drop_last:
                ret.append( idxs[i*self._batch_size:])
        if self._shuffle_batch:
            random.shuffle(ret)
        return iter(ret)

    def __len__(self):
        return self._length


class ClipSampler(Sampler):
    '''
    every batch is build using data in the same clip (also in the same match)
    '''
    _ret = []
    def __init__(self, dataset, batch_size=4, shuffle_within_clip=True, shuffle_batch=True, drop_last=True):
        self._batch_size          = batch_size
        self._shuffle_batch       = shuffle_batch
        self._shuffle_within_clip = shuffle_within_clip
        self._drop_last           = drop_last
        log.info('launch ClipSampler. batch size: {}, shuffle_batch: {}, shuffle_within_clip: {}, drop_last: {}'.format(self._batch_size, self._shuffle_batch, self._shuffle_within_clip, self._drop_last))
        self._data_dict = defaultdict(list)
        for idx, d in enumerate(dataset):
            match, clip = d['match'], d['clip']
            self._data_dict[(match, clip)].append(idx)
        #print(self._data_dict)
        self._length = 0
        for key, idxs in self._data_dict.items():
            self._length += len(idxs) // self._batch_size
            if not self._drop_last:
                self._length += 1
        #print(self._length)
        log.info('# of batches: {}'.format(self._length) )

    def __iter__(self):
        ret = []
        for key, idxs in self._data_dict.items():
            if self._shuffle_within_clip:
                random.shuffle(idxs)
            for i in range(len(idxs)//self._batch_size):
                ret.append( idxs[i*self._batch_size:(i+1)*self._batch_size])
            if not self._drop_last:
                ret.append( idxs[i*self._batch_size:])
        if self._shuffle_batch:
            random.shuffle(ret)
        return iter(ret)

    def __len__(self):
        return self._length

