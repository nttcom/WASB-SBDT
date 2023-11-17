import torch
from torch import nn

class DiceLoss(nn.Module):
    '''
    https://arxiv.org/abs/1805.02798 , section 2
    '''
    def __init__(self, epsilon: float = 1e-4, for_combo_loss : bool = False):
        super().__init__()
        self._epsilon        = epsilon
        self._for_combo_loss = for_combo_loss

    def forward(self, inputs, targets):
        intersection = inputs * targets
        #loss = 1. - 2. * intersection.sum() / ( inputs.sum() + targets.sum() + self.epsilon)
        loss = - 2. * ( intersection.sum() + self._epsilon) / ( inputs.sum() + targets.sum() + self._epsilon)
        if not self._for_combo_loss:
            loss += 1.
        return loss

#
if __name__=='__main__':

    loss_criteria = DiceLoss()
    inputs  = torch.rand((8,9,288,512), requires_grad=True) # [0,1]
    targets = (torch.rand((8,9,288,512)) > 0.5).float()     # 0 or 1
    #print(inputs)
    #print(targets)
    loss = loss_criteria(inputs, targets)
    #print(loss)
