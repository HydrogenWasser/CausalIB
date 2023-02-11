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

    def generate(self, x, **params):
        self.parse_params(**params)
        labels = self.y

        adv_x = self.attack(x, labels)
        return adv_x

    def parse_params(self, eps=0.3, iter_eps=0.01, nb_iter=40, clip_min=0.0, clip_max=1.0, C=0.0,
                     y=None, ord=np.inf, rand_init=True, flag_target=False):
        self.eps = eps
        self.iter_eps = iter_eps
        self.nb_iter = nb_iter
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.y = y
        self.ord = ord
        self.rand_init = rand_init
        self.model.to(self.device)
        self.flag_target = flag_target
        self.C = C

    def sigle_step_attack(self, x, pertubation, labels):
        adv_x = x + pertubation
        adv_x = torch.clamp(adv_x, self.clip_min, self.clip_max)
        # get the gradient of x
        adv_x = Variable(adv_x)
        adv_x.requires_grad = True

        loss_func = nn.CrossEntropyLoss()

        preds = self.model.forward(adv_x)

        if self.flag_target:
            loss = -loss_func(preds, labels)
        else:
            loss = loss_func(preds, labels)
        # label_mask=torch_one_hot(labels)
        #
        # correct_logit=torch.mean(torch.sum(label_mask * preds,dim=1))
        # wrong_logit = torch.mean(torch.max((1 - label_mask) * preds, dim=1)[0])
        # loss=-F.relu(correct_logit-wrong_logit+self.C)


        self.model.zero_grad()
        loss.backward()
        grad = adv_x.grad.data
        # get the pertubation of an iter_eps
        pertubation = self.iter_eps * grad.sign()
        adv_x = adv_x.cpu().detach().numpy() + pertubation.cpu().numpy()
        x = x.cpu().detach().numpy()

        pertubation = np.clip(adv_x-x, -self.eps, self.eps)

        return pertubation

    def attack(self, x, labels):
        labels = labels.to(self.device)
        if self.rand_init:
            x_tmp = x + torch.Tensor(np.random.uniform(-self.eps, self.eps, x.shape)).type_as(x).to(self.device)
        else:
            x_tmp = x
        pertubation = torch.zeros(x.shape).type_as(x).to(self.device)
        for i in range(self.nb_iter):
            pertubation = self.sigle_step_attack(x_tmp, pertubation=pertubation, labels=labels)
            pertubation = torch.Tensor(pertubation).type_as(x).to(self.device)
        adv_x = x + pertubation
        adv_x = adv_x.cpu().detach().numpy()

        adv_x = np.clip(adv_x, self.clip_min, self.clip_max)
        adv_x = torch.Tensor(adv_x).type_as(x).to(self.device)

        return adv_x
