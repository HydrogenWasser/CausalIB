import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt

def KL_between_normals(q_distr, p_distr):
    mu_q, sigma_q = q_distr
    mu_p, sigma_p = p_distr
    k = mu_q.size(1)

    mu_diff = mu_p - mu_q
    mu_diff_sq = torch.mul(mu_diff, mu_diff)
    logdet_sigma_q = torch.sum(2 * torch.log(torch.clamp(sigma_q, min=1e-8)), dim=1)
    logdet_sigma_p = torch.sum(2 * torch.log(torch.clamp(sigma_p, min=1e-8)), dim=1)

    fs = torch.sum(torch.div(sigma_q ** 2, sigma_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, sigma_p ** 2), dim=1)
    two_kl = fs - k + logdet_sigma_p - logdet_sigma_q
    return two_kl * 0.5

class causalIB(nn.Module):
    """
    Hydrogen IB model (FC).
    """
    def __init__(self, device):
        super(causalIB, self).__init__()
        self.beta = 0.2
        self.z_dim = 256
        self.num_sample = 12
        self.device = device
        self.batch_size = 100
        self.w_ce = 1
        self.w_reg = 0.2
        self.relu = nn.ReLU(inplace=True)
        self.Encoder = nn.Sequential(nn.Linear(in_features=784, out_features=1024),
                                     nn.ReLU(),
                                     nn.Linear(in_features=1024, out_features=1024),
                                     nn.ReLU()
                                     )
        self.getMean = nn.Sequential(nn.Linear(in_features=1024, out_features=512),
                                                nn.ReLU(),
                                                nn.Linear(in_features=512, out_features=self.z_dim)
                                                )
        self.getStd = nn.Sequential(  nn.Linear(in_features=1024, out_features=512),
                                               nn.ReLU(),
                                               nn.Linear(in_features=512, out_features=self.z_dim)
                                                )

        # self.Instrument = nn.Sequential(nn.Linear(in_features=self.z_dim, out_features=self.z_dim//2))


        self.Decoder = nn.Sequential(nn.Linear(in_features=self.z_dim, out_features=10))

    def gaussian_noise(self, num_samples, K):
        return torch.normal(torch.zeros(*num_samples, K), torch.ones(*num_samples, K)).to(self.device)

    def sample_prior_Z(self, num_samples):
        return self.gaussian_noise(num_samples=num_samples, K=self.dimZ)

    def encoder_result(self, x):
        encoder_output = self.Encoder(x*2-1)
        mean = self.getMean(encoder_output)
        std = self.getStd(encoder_output)
        return mean, std

    def sample_Z(self, num_samples, x):
        mean, std = self.encoder_result(x)
        return mean, std, mean + std * self.gaussian_noise(num_samples=(num_samples, self.batch_size), K=self.z_dim)
               # mean_S, std_S, mean_S + std_S * self.gaussian_noise(num_samples=(num_samples, batch_size), K=self.z_dim)

    def get_logits(self, x):
        #encode
        mean, std, z = self.sample_Z(num_samples=self.num_sample, x=x)

        #scores
        z_scores = []
        features = []
        for i in range(self.num_sample):
            z_sample = z[i]
            z_score = (abs(z_sample-mean) < 2*std).float()
            z_score = torch.mean(z_score, dim=1)
            z_scores.append(z_score.view(-1, 1))
            features.append(z[i])
        tempt_z_scores = torch.cat(z_scores, dim=0).reshape(100, 12, 1)
        #decode
        # tempt_features = self.Instrument(z)
        y_logits = self.Decoder(z)

        # for i in range(self.num_sample):
        #     features.append(tempt_features[i])

        return mean, std, features, y_logits, z_scores, tempt_z_scores
        # mean (100, 256)
        # std (100, 256)
        # features (12, 100, 128)
        # z_scores (100, 12, 1)
        # y_logits (12, 100, 10)

    def smooth_l1_loss(self, x, y):
        diff = F.smooth_l1_loss(x, y, reduction='none')
        diff = diff.sum(1)
        diff = diff.mean(0)
        return diff

    def get_mean_wo_i(self, inputs, i):
        return (sum(inputs) - inputs[i]) / float(len(inputs) - 1)

    def myForward(self, x):
        self.batch_size = x.shape[0]
        x = x.view(x.size(0), -1)
        mean, std, features, y_logitss, z_scores, tempt_z_scores = self.get_logits(x)

        final_pred = torch.mean(y_logitss, dim=0)
        y_pre = torch.softmax(final_pred, dim=1)

        # y_logits = y_logitss.permute(1, 2, 0)
        # final_pred = torch.bmm(y_logits, tempt_z_scores).reshape(100, 10)
        # final_pred = final_pred / self.num_sample


        return y_pre, mean, std, features, y_logitss, z_scores

        # y_pre (100, 10)

    def forward(self, x):
        self.batch_size = x.shape[0]
        x = x.view(x.size(0), -1)
        mean, std, z = self.sample_Z(num_samples=self.num_sample, x=x)
        y_logits = self.Decoder(z)
        final_pred = torch.mean(y_logits, dim=0)
        y_pre = torch.softmax(final_pred, dim=1)
        return y_pre

    def IB_loss(self, mean, std, y_logits, y_batch, num_samples):
       # compute I(X,T)
        prior_Z_distr = torch.zeros(self.batch_size, self.z_dim).to(self.device), torch.ones(self.batch_size, self.z_dim).to(self.device)
        enc_dist = mean, std
        I_X_T_bound = torch.mean(KL_between_normals(enc_dist, prior_Z_distr)) / math.log(2)

        # compute I(Y,T)
        loss_func = nn.CrossEntropyLoss(reduce=False)
        y_logits = y_logits.permute(1, 2, 0)    # y_logits (100, 10, 12)
        y_label = y_batch[:, None].expand(-1, num_samples)      # y_label (100, 12)
        cross_entropy_loss = loss_func(y_logits, y_label)
        cross_entropy_loss_montecarlo = torch.mean(cross_entropy_loss, dim=-1)
        I_Y_T_bound = torch.mean(cross_entropy_loss_montecarlo, dim=0) / math.log(2)

        # compute Loss
        loss = I_Y_T_bound + self.beta*I_X_T_bound

        return loss, I_X_T_bound, math.log(10, 2) - I_Y_T_bound


    def batch_loss(self, mean, std, y_logits, features, z_scores, y_batch):


        Ibloss, I_X_T_bound, I_Y_T_bound = self.IB_loss(mean, std, y_logits, y_batch, num_samples=12)

        all_regs = []
        for i in range(len(features)):
            reg_loss = self.smooth_l1_loss(features[i] * self.get_mean_wo_i(z_scores, i), self.get_mean_wo_i(features, i) * z_scores[i])
            all_regs.append(reg_loss)

        loss = self.w_ce * Ibloss + self.w_reg * sum(all_regs) / len(all_regs)

        return loss, I_X_T_bound, I_Y_T_bound

    def give_beta(self, beta):
        self.beta = beta
    def give_w_reg(self, w_reg):
        self.w_reg = w_reg

class EMA(nn.Module):
    def __init__(self, mu):
        super(EMA, self).__init__()
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def forward(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average
