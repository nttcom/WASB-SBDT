import logging
import torch
from torch import nn

log = logging.getLogger(__name__)

class BinaryFocalLoss(nn.Module):
    '''
    https://arxiv.org/abs/1708.02002
    '''
    def __init__(self, gamma: float = 2, auto_weight: bool = False, scales=[0]):
        super().__init__()
        self._gamma       = gamma
        self._auto_weight = auto_weight
        self._scales      = scales
        self._ws          = {}
        log.info('(focal loss) gamma: {}, auto_weight: {}, scales: {}'.format(self._gamma, self._auto_weight, self._scales))
        for scale in self._scales:
            if self._auto_weight:
                self._ws['loss_w_s{}'.format(scale)] = nn.Parameter(0. * torch.ones(1))
            else:
                self._ws['loss_w_s{}'.format(scale)] = 1.0
        if len(self._scales)==1 and self._auto_weight:
            log.info('auto_weight=True even though len(scales)==1')
        #print(self._ws)
        if self._auto_weight:
            self._ws = nn.ParameterDict(self._ws)

    """
    def forward(self, inputs, targets):
        #loss = ((1-inputs)**2) * targets * torch.log(inputs) + (inputs**2) * (1-targets) * torch.log(1-inputs)
        loss = targets * (1-inputs)**self._gamma * torch.log(inputs) + (1-targets) * inputs**self._gamma * torch.log(1-inputs)
        loss = torch.mean(-loss)
        return loss
    """

    def forward(self, inputs, targets):
        loss_acc = 0
        for scale in inputs.keys():
            #print(inputs[scale].shape, targets[scale].shape)
            #loss = ((1-inputs[scale])**2) * targets[scale] * torch.log(inputs[scale]) + (inputs[scale]**2) * (1-targets[scale]) * torch.log(1-inputs[scale])
            loss = targets[scale] * (1-inputs[scale])**self._gamma * torch.log(inputs[scale]) + (1-targets[scale]) * inputs[scale]**self._gamma * torch.log(1-inputs[scale])
            loss = torch.mean(-loss)
            if self._auto_weight:
                loss_acc += loss * torch.exp(-self._ws['loss_w_s{}'.format(scale)]) + self._ws['loss_w_s{}'.format(scale)]
            else:
                loss_acc += loss * self._ws['loss_w_s{}'.format(scale)]
        return loss_acc

