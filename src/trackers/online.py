import numpy as np

class Track:
    def __init__(self):
        self._xy_dict    = {}
        self._score_dict = {}
        self._visi_dict  = {}

    def add(self, fid, x, y, visi, score):
        self._xy_dict[fid]    = np.array([x,y])
        self._visi_dict[fid]  = visi
        self._score_dict[fid] = score
    
    def is_visible(self, fid):
        if not fid in self._visi_dict.keys():
            #raise KeyError('fid {} not found'.format(fid))
            return False
        return self._visi_dict[fid]
    
    @property
    def last_fid(self):
        fids = list( self._xy_dict.keys() )
        return max(fids)

    def xy(self, fid):
        if not fid in self._xy_dict.keys():
            raise KeyError('fid {} not found'.format(fid))
        return self._xy_dict[fid]

    def predict(self, last_fid):
        fid1 = last_fid
        fid2 = fid1 - 1
        fid3 = fid1 - 2
        if self.is_visible(fid1) and self.is_visible(fid2) and self.is_visible(fid3):
            xy1 = self._xy_dict[fid1]
            xy2 = self._xy_dict[fid2]
            xy3 = self._xy_dict[fid3]
            acc     = (xy1-xy2) - (xy2-xy3)
            vel     = (xy1-xy2) + acc
            xy_pred = xy1 + vel + acc / 2
        else:
            xy_pred = None
        return xy_pred

class OnlineTracker:
    def __init__(self, cfg):
        self._max_disp = cfg['tracker']['max_disp']
        self._fid      = 0
        self._track    = Track()

    def _select_best(self, frame_dets):
        best_score = - np.Inf
        visi       = False
        x, y       = - np.Inf, - np.Inf

        xy_pred = None
    
        for det in frame_dets:
            score = det['score']
            if xy_pred is not None:
                qscore  = self._compute_quality(xy_pred, det['xy'], self._track.xy(self._fid-1) )
                score  += qscore

            if score > best_score:
                best_score = score
                xy         = det['xy']
                x,y        = xy[0], xy[1]
                visi       = True
        return x,y,visi,best_score 

    def _select_not_too_far(self, frame_dets):
        if (self._fid==0) or (not self._track.is_visible(self._fid-1)):
            return frame_dets

        frame_dets_ = []
        for det in frame_dets:
            if np.linalg.norm( det['xy'] - self._track.xy(self._fid-1) ) < self._max_disp:
                frame_dets_.append(det)
        return frame_dets_

    def _compute_quality(self, xy1, xy2, xy3):
        return - np.linalg.norm( xy1-xy2 )

    def update(self, frame_dets):
        frame_dets     = self._select_not_too_far(frame_dets)
        x,y,visi,score = self._select_best(frame_dets)
        self._track.add(self._fid, x, y, visi, score)

        self._fid += 1
        return {'x': x, 'y': y, 'visi': visi, 'score': score}

    def refresh(self):
        self._fid   = 0
        self._track = Track()

