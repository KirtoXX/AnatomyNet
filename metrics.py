import torch.nn as nn

class Dice_coff(nn.Module):
    def __init__(self):
        super(Dice_coff,self).__init__()

    def forward(self,pred,true):
        b = pred.size(0)
        p = pred.view(b,-1)
        t = true.view(b,-1)
        coff = 2*(p*t).sum(1)/(p.sum(1)+t.sum(1))
        return coff