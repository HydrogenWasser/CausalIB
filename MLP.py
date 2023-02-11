import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Hydrogen IB model (FC).
    """
    def __init__(self, device):
        super(MLP, self).__init__()
        self.device = device
        self.batch_size = 100
        self.Encoder = nn.Sequential(nn.Linear(in_features=784, out_features=1024),
                                               nn.ReLU(),
                                               nn.Linear(in_features=1024, out_features=1024),
                                               nn.ReLU(),
                                               nn.Linear(in_features=1024, out_features=256),
                                               nn.ReLU(),
                                               nn.Linear(in_features=256, out_features=10)
                                     )

    def forward(self, x):
        x = x.view(-1, 28*28)
        y_pre = self.Encoder(x * 2 - 1)
        return torch.softmax(y_pre, dim=1)

    def batch_loss(self, y_pre, y_batch):

        loss_func = nn.CrossEntropyLoss(reduce=False)
        CE_loss = torch.mean(loss_func(y_pre, y_batch), dim=0)


        return CE_loss



