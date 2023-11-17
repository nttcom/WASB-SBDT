import numpy as np

class IntraFramePeakTracker:
    def __init__(self, cfg):
        pass
    
    def update(self, frame_dets):
        best_score = - np.Inf
        visi = False
        x, y = - np.Inf, - np.Inf
        for det in frame_dets:
            score = det['score']
            if score > best_score:
                best_score = score
                xy         = det['xy']
                x,y        = xy[0], xy[1]
                visi       = True

        return {'x': x, 'y': y, 'visi': visi, 'score': best_score}

    def refresh(self):
        pass

