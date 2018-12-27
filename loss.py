import torch.nn as nn
import torch.nn.functional as F
import torch

class Dice_loss(nn.Module):
    def __init__(self):
        super(Dice_loss,self).__init__()

    def forward(self,pred,true):
        b = pred.size(0)
        p = pred.view(b,-1)
        t = true.view(b,-1)
        coff = 2*(p*t).sum(1)/(p.sum(1)+t.sum(1))
        loss = 1- coff.mean(0)
        return loss

class Focal_loss(nn.Module):
    def __init__(self,pow=2):
        super(Focal_loss,self).__init__()
        self.power = pow

    def forward(self,pred,true):
        b,c,w,h,d = pred.size()
        p = pred.view(b, -1)
        t = true.view(b, -1)
        loss = t*(1 - p).pow(self.power)*p.log()
        loss = loss.sum(1)/(w*h*d)
        loss = -loss.mean(0)
        return loss

class Hybrid_loss(nn.Module):
    def __int__(self,lambdaa=0.5,classes=10):
        super(Hybrid_loss,self).__init__()
        self.dice_loss = Dice_loss()
        self.focal_loss = Focal_loss()
        self.lambdaa = lambdaa
        self.classes = classes

    def forward(self,pred,true):
        loss1 = self.dice_loss(pred,true)
        loss2 = self.focal_loss(pred,true)
        total_loss = loss1 + self.lambdaa*loss2
        total_loss = total_loss*self.classes
        return total_loss






