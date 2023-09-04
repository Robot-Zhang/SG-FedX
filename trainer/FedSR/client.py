import copy

from torch.utils.data import DataLoader

from trainer.FedSR.SRNet import SRNet
from utils.base_train import evaluate_model
from trainer.FedAvg.client import Client as BaseClient

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.nets import TwinBranchNets, mlp


class Client(BaseClient):
    """
    Client for FedSR

    Args:

    """

    def __init__(self, alpha_l2r=0.01, alpha_cmi=0.001, **config):
        """

        :param alpha_l2r: alpha of l2r, 0.01 as default
        :param alpha_cmi: alpha of cmi, 0.001 as default
        (This two param is set according to authors' opensource code)
        """

        super(Client, self).__init__(**config)

        assert isinstance(self.model, TwinBranchNets), \
            "FedSR need model in format of [feature_extractor, classifier]. Now, only TwinBranchNets is ok."

        self.alpha_l2r = alpha_l2r
        self.alpha_cmi = alpha_cmi

        # modify model
        self.model = SRNet(self.model, self.num_classes)
        self.z_dim = self.model.z_dim

        self.r_mu = nn.Parameter(torch.zeros(self.num_classes, self.z_dim))
        self.r_sigma = nn.Parameter(torch.ones(self.num_classes, self.z_dim))
        self.C = nn.Parameter(torch.ones([]))
        self.optimizer.add_param_group({'params': [self.r_mu, self.r_sigma, self.C], 'lr': self.lr, 'momentum': 0.9})

    def update(self, epochs=1, verbose=0):
        # step 1. model init
        self.model.to(self.device)
        self.r_mu, self.r_sigma, self.C = self.r_mu.to(self.device), self.r_sigma.to(self.device), self.C.to(self.device)
        self.model.train()
        # step 2. train loop
        loss_metric = []  # to record avg loss
        for epoch in range(epochs):
            # init loss value
            loss_value, num_samples = 0, 0
            # one epoch train
            for i, (x, y) in enumerate(self.train_loader):
                # put tensor into same device
                x, y = x.to(self.device), y.to(self.device)
                # calculate loss
                z, (z_mu, z_sigma) = self.model.featurize(x, return_dist=True)

                logits = self.model.classifier(z)
                loss = self.loss_fn(logits, y)

                obj = loss
                regL2R = torch.zeros_like(obj)
                regCMI = torch.zeros_like(obj)
                # regNegEnt = torch.zeros_like(obj)
                if self.alpha_l2r != 0.0:
                    regL2R = z.norm(dim=1).mean()
                    obj = obj + self.alpha_l2r * regL2R
                if self.alpha_cmi != 0.0:
                    r_sigma_softplus = F.softplus(self.r_sigma)
                    r_mu = self.r_mu[y]
                    r_sigma = r_sigma_softplus[y]
                    z_mu_scaled = z_mu * self.C
                    z_sigma_scaled = z_sigma * self.C
                    regCMI = torch.log(r_sigma) - torch.log(z_sigma_scaled) + \
                             (z_sigma_scaled ** 2 + (z_mu_scaled - r_mu) ** 2) / (2 * r_sigma ** 2) - 0.5
                    regCMI = regCMI.sum(1).mean()
                    obj = obj + self.alpha_cmi * regCMI
                # y_ = F.log_softmax(y_, dim=-1)

                # backward & step optim
                self.optimizer.zero_grad()
                obj.backward()
                self.optimizer.step()
                # get loss valur of current bath
                loss_value += loss.item()
                num_samples += y.size(0)

            # Use mean loss value of each epoch as metric
            # Just a reference value, not precise. If you want precise, dataloader should set `drop_last = True`.
            loss_value = loss_value / num_samples
            loss_metric.append(loss_value)
        # step 3. release gpu resource
        # self.model.to('cpu')
        # self.r_mu, self.r_sigma, self.C = \
        #     self.r_mu.to('cpu'), self.r_sigma.to('cpu'), self.C.to('cpu')
        torch.cuda.empty_cache()

        avg_loss = sum(loss_metric) / len(loss_metric)
        return avg_loss
