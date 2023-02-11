import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
samples_amount = 10

class attack_model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    def parse_params(self, eps=0.3, ord=np.inf, clip_min=0.0, clip_max=1.0,
                     y=None, rand_init=False, flag_target=False):
        self.eps = eps
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.y = y
        self.rand_init = rand_init
        self.model.to(self.device)
        self.flag_target = flag_target

    def generate(self, x, **params):
        self.parse_params(**params)
        labels = self.y
        if self.rand_init:
            x_new = x + torch.Tensor(np.random.uniform(-self.eps, self.eps, x.shape)).type_as(x).cuda()
        else:
            x_new = x

        # get the gradient of x
        x_new = Variable(x_new, requires_grad=True)
        loss_func = nn.CrossEntropyLoss()
        preds = self.model(x_new)

        loss = loss_func(preds, labels)
        self.model.zero_grad()
        loss.backward()
        grad = x_new.grad.cpu().detach().numpy()
        # get the pertubation of an iter_eps
        if self.ord == np.inf:
            grad = np.sign(grad)
        else:
            tmp = grad.reshape(grad.shape[0], -1)
            norm = 1e-12 + np.linalg.norm(tmp, ord=self.ord, axis=1, keepdims=False).reshape(-1, 1, 1, 1)
            # 选择更小的扰动
            grad = grad / norm
        pertubation = grad * self.eps

        adv_x = x.cpu().detach().numpy() + pertubation
        adv_x = np.clip(adv_x, self.clip_min, self.clip_max)
        adv_x = torch.Tensor(adv_x).type_as(x).to(self.device)

        return adv_x

