import logging
from typing import List
import torch
from torch import nn

from .dice_loss import DiceLoss

log = logging.getLogger(__name__)

class ComboLoss(nn.Module):
    
    '''
    - https://arxiv.org/abs/1805.02798 , section 2
    - https://arxiv.org/abs/2204.01899 , section 4.3 (incorrect)
    '''
    def __init__(self, alpha: float = 0.1, epsilon: float = 1e-4, auto_weight: bool = False, scales: List[int] = [0]):
        super().__init__()
        self._alpha = alpha
        self._loss1 = nn.BCELoss()
        self._loss2 = DiceLoss(epsilon=epsilon, for_combo_loss=True)
        
        self._auto_weight = auto_weight
        self._ws          = {}
        if len(scales) > 1:
            assert 0, 'not validated yet (2022.10.16)'
        for scale in scales:
            if self._auto_weight:
                self._ws['loss_w_s{}'.format(scale)] = nn.Parameter(0. * torch.ones(1))
            else:
                self._ws['loss_w_s{}'.format(scale)] = 1.0
        if len(scales)==1 and self._auto_weight:
            log.info('auto_weight=True even though len(scales)==1')
        #print(self._ws)
        if self._auto_weight:
            self._ws = nn.ParameterDict(self._ws)
        #print(self._ws)

    """
    def forward(self, inputs, targets):
        l1 = self._loss1(inputs, targets)
        l2 = self._loss2(inputs, targets)
        #log.info('loss (bce): {}, loss (dice): {}'.format(l1, l2))
        loss = (1-self._alpha) * l1 + self._alpha * l2
        return loss
    """

    def forward(self, inputs, targets):
        loss_acc = 0
        for scale in inputs.keys():
            l1 = self._loss1(inputs[scale], targets[scale])
            l2 = self._loss2(inputs[scale], targets[scale])
            #log.info('loss (bce): {}, loss (dice): {}'.format(l1, l2))
            loss = (1-self._alpha) * l1 + self._alpha * l2
            if self._auto_weight:
                loss_acc += loss * torch.exp(-self._ws['loss_w_s{}'.format(scale)]) + self._ws['loss_w_s{}'.format(scale)]
            else:
                loss_acc += loss * self._ws['loss_w_s{}'.format(scale)]
        return loss_acc

#
if __name__=='__main__':

    loss_criteria = MonoTrackLoss()
    inputs  = torch.rand((8,9,288,512), requires_grad=True) # [0,1]
    targets = (torch.rand((8,9,288,512)) > 0.5).float()     # 0 or 1
    #print(inputs)
    #print(targets)
    loss = loss_criteria(inputs, targets)
    print(loss)

