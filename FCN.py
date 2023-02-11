import torch
import torch.nn as nn


class common_FCN(nn.Module):
    """
    Hydrogen IB model (FC).
    """
    def __init__(self, device):
        super(common_FCN, self).__init__()
        self.device = device
        self.batch_size = 100

        self.fcn1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True))
        self.fcn2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True))
        self.fcn3 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True))
        self.fcn4 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True))
        self.fcn5 = nn.Sequential(nn.Conv2d(32, 1, kernel_size=3, padding=1),nn.ReLU(inplace=True))


    def forward(self, x):
        x = self.fcn1(x)
        x = self.fcn2(x)
        x = self.fcn3(x)
        x = self.fcn4(x)
        y_pre = self.fcn5(x)
        return y_pre

    def batch_loss(self, y_pre, y_batch):

        loss_func = nn.CrossEntropyLoss(reduce=False)
        cross_entropy_loss = loss_func(y_pre, y_batch)
        loss = torch.mean(cross_entropy_loss, dim=0)

        return loss



