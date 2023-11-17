import numpy as np
import logging

log = logging.getLogger(__name__)

class Evaluator(object):
    def __init__(self, cfg):
        self._dist_threshold = cfg['runner']['eval']['dist_threshold']
        self._tp  = 0
        self._fp1 = 0
        self._fp2 = 0
        self._tn  = 0
        self._fn  = 0
        self._ses = [] # squared error
        self._scores = []
        self._ys     = []

    def eval_single_frame(self, xy_pred, visi_pred, score_pred, xy_gt, visi_gt):
        tp, fp1, fp2, tn, fn = 0, 0, 0, 0, 0
        se = None
        if visi_gt:
            if visi_pred:
                if np.linalg.norm( np.array(xy_pred)-np.array(xy_gt) ) < self._dist_threshold:
                    tp += 1
                else:
                    fp1 += 1
                se = np.linalg.norm( np.array(xy_pred)-np.array(xy_gt) )**2
                self._ses.append(se)
            else:
                fn += 1
        else:
            if visi_pred:
                fp2 += 1
            else:
                tn += 1
        self._tp  += tp
        self._fp1 += fp1
        self._fp2 += fp2
        self._tn  += tn
        self._fn  += fn

        if tp > 0 or fp1 > 0 or fp2 > 0:
            if tp > 0:
                self._ys.append(1)
            else:
                self._ys.append(0)
            self._scores.append(score_pred)

        return {'tp': tp, 'tn': tn, 'fp1': fp1, 'fp2': fp2, 'fn': fn, 'se': se}

    @property
    def dist_threshold(self):
        return self._dist_threshold

    @property
    def tp_all(self):
        return self._tp

    @property
    def fp1_all(self):
        return self._fp1

    @property
    def fp2_all(self):
        return self._fp2

    @property
    def fp_all(self):
        return self.fp1_all + self.fp2_all

    @property
    def tn_all(self):
        return self._tn

    @property
    def fn_all(self):
        return self._fn

    @property
    def prec(self):
        prec = 0.
        if (self.tp_all + self.fp_all) > 0.:
            prec = self.tp_all / (self.tp_all + self.fp_all)
        return prec

    @property
    def recall(self):
        recall = 0.
        if (self.tp_all + self.fn_all) > 0.:
            recall = self.tp_all / (self.tp_all + self.fn_all)
        return recall

    @property
    def f1(self):
        f1 = 0.
        if self.prec+self.recall > 0.:
            f1 = 2 * self.prec * self.recall / (self.prec + self.recall)
        return f1
    
    @property
    def accuracy(self):
        accuracy = 0.
        if self.tp_all+self.tn_all+self.fp_all+self.fn_all > 0.:
            accuracy = (self.tp_all+self.tn_all) / (self.tp_all+self.tn_all+self.fp_all+self.fn_all)
        return accuracy

    @property
    def sq_errs(self):
        return self._ses

    @property
    def ap(self):
        inds = np.argsort(-1 * np.array(self._scores)).tolist()
        tp   = 0
        r2p  = {}
        for i, ind in enumerate(inds, start=1):
            tp += self._ys[ind]
            p   = tp / i
            r   = tp / (self.tp_all + self.fn_all)
            if not r in r2p.keys():
                r2p[r] = p
            else:
                if r2p[r] < p:
                    r2p[r] = p
        prev_r = 0
        ap = 0.
        for r, p in r2p.items():
            ap += (r-prev_r) * p
            prev_r = r
        return ap

    @property
    def rmse(self):
        _rmse = - np.Inf
        if len(self.sq_errs) > 0:
            _rmse = np.sqrt(np.array(self.sq_errs).mean())
        return _rmse

    def print_results(self, txt=None, elapsed_time=0., num_frames=0, with_ap=True):
        if txt is not None:
            log.info('{}'.format(txt))
        if num_frames > 0:
            log.info('Elapsed time: {}, FPS: {} ({}/{})'.format(elapsed_time, num_frames/elapsed_time, num_frames, elapsed_time))
        if with_ap:
            log.info('| TP   | TN   | FP1   | FP2   | FP   | FN   | Prec       | Recall       | F1       | Accuracy       | RMSE | AP  |')
            log.info('| ---- | ---- | ----- | ----- | ---- | ---- | ---------- | ------------ | -------- | -------------- | ---- | ----- |')
            log.info('| {tp} | {tn} | {fp1} | {fp2} | {fp} | {fn} | {prec:.4f} | {recall:.4f} | {f1:.4f} | {accuracy:.4f} | {rmse:.2f}({num_ses}) | {ap:.4f} |'.format(tp=self.tp_all, tn=self.tn_all, fp1=self.fp1_all, fp2=self.fp2_all, fp=self.fp_all, fn=self.fn_all, prec=self.prec, recall=self.recall, f1=self.f1, accuracy=self.accuracy, rmse=self.rmse, num_ses=len(self.sq_errs), ap=self.ap))
        else:
            log.info('| TP   | TN   | FP1   | FP2   | FP   | FN   | Prec       | Recall       | F1       | Accuracy       | RMSE |')
            log.info('| ---- | ---- | ----- | ----- | ---- | ---- | ---------- | ------------ | -------- | -------------- | ---- |')
            log.info('| {tp} | {tn} | {fp1} | {fp2} | {fp} | {fn} | {prec:.4f} | {recall:.4f} | {f1:.4f} | {accuracy:.4f} | {rmse:.2f}({num_ses}) |'.format(tp=self.tp_all, tn=self.tn_all, fp1=self.fp1_all, fp2=self.fp2_all, fp=self.fp_all, fn=self.fn_all, prec=self.prec, recall=self.recall, f1=self.f1, accuracy=self.accuracy, rmse=self.rmse, num_ses=len(self.sq_errs)))




