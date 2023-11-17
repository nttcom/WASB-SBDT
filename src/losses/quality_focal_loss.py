import torch
from torch import nn

class QualityFocalLoss(nn.Module):
    '''
    https://arxiv.org/abs/2006.04388
    '''
    def __init__(self, beta: float = 2, auto_weight: bool = False, scales=[0]):
        super().__init__()
        self._beta        = beta
        self._auto_weight = auto_weight
        self._ws          = {}
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
        #loss = ((1-inputs)**2) * targets * torch.log(inputs) + (inputs**2) * (1-targets) * torch.log(1-inputs)
        loss = targets * (1-inputs)**self._gamma * torch.log(inputs) + (1-targets) * inputs**self._gamma * torch.log(1-inputs)
        loss = torch.mean(-loss)
        return loss
    """

    def forward(self, inputs, targets):
        loss_acc = 0
        for scale in inputs.keys():
            #print(targets[scale].shape)
            #print(targets[scale].sum())
            #print(inputs[scale].shape, targets[scale].shape)
            #loss = ((1-inputs[scale])**2) * targets[scale] * torch.log(inputs[scale]) + (inputs[scale]**2) * (1-targets[scale]) * torch.log(1-inputs[scale])
            #loss = torch.mean(-loss)
            loss = torch.abs( inputs[scale]-targets[scale] )**self._beta * ( (1-targets[scale])*torch.log(1-inputs[scale]) + targets[scale]*torch.log(inputs[scale]) )
            #print(loss)
            #loss = targets * (1-inputs)**self._gamma * torch.log(inputs) + (1-targets) * inputs**self._gamma * torch.log(1-inputs)
            loss = torch.mean(-loss)
            #print(loss)
            if self._auto_weight:
                loss_acc += loss * torch.exp(-self._ws['loss_w_s{}'.format(scale)]) + self._ws['loss_w_s{}'.format(scale)]
            else:
                loss_acc += loss * self._ws['loss_w_s{}'.format(scale)]
        
        #print(loss_acc)
        return loss_acc



