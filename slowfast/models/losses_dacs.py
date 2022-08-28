import torch
import torch.nn as nn


class BCELossWeighted(nn.Module):
    def __init__(self, weight=None, reduction='none'):
        super(BCELossWeighted, self).__init__()
        self.BCE = nn.BCELoss(weight=weight, reduction=reduction)

    def forward(self, output, target, weight):
        loss = self.BCE(output, target)
        loss = torch.mean(loss * weight)
        return loss


class MSELossWeighted(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='none'):
        super(MSELossWeighted, self).__init__()
        self.MSE = nn.MSELoss(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, output, target, weight):
        loss = self.MSE(output, target)
        loss = torch.mean(loss * weight)
        return loss


class CrossEntropyLossWeighted(nn.Module):
    def __init__(self, weight=None, reduction='none'):
        super(CrossEntropyLossWeighted, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, output, target, weight):
        loss = self.CE(output, target)
        loss = torch.mean(loss * weight)
        return loss