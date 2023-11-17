import logging
from collections import Counter
from bisect import bisect_right

from torch.optim.lr_scheduler import _LRScheduler

log = logging.getLogger(__name__)

# cf. https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py
class MultiStepLRWithWarmUp(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False):
        log.info('launch MultiStepLRWithWarmUp ')
        self.milestones = Counter(milestones)
        for k, v in self.milestones.items():
            if v > 1:
                raise ValueError('invalid milestones: {}'.format(milestones))
        self.gamma      = gamma
        super(MultiStepLRWithWarmUp, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        
        milestones       = list(sorted(self.milestones.elements()))
        first_milestone  = milestones[0]
        other_milestones = Counter(milestones[1:])
        if self.last_epoch+1 <= first_milestone:
            return [ base_lr * self.gamma * (self.last_epoch+1) for base_lr in self.base_lrs]
        else:
            if self.last_epoch+1 not in self.milestones:
                return [ group['lr'] for group in self.optimizer.param_groups]
            return [ group['lr'] * self.gamma ** other_milestones[self.last_epoch+1] for group in self.optimizer.param_groups]

"""
if __name__=='__main__':

    import torch
    import torch.optim as optim

    model = torch.nn.Sequential(
            torch.nn.Linear(3, 1),
            torch.nn.Flatten(0, 1))
    optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=0.)
    #scheduler = MultiStepLRWithWarmUp(optimizer, milestones=[10,20], gamma=0.1)
    scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=0.1)

    for epoch in range(30):
        print(epoch+1, scheduler.get_last_lr())
        optimizer.step()
        scheduler.step()
"""

