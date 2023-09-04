from torch import nn
from utils.nets import TwinBranchNets, mlp
import torch.distributions as distributions
import torch.nn.functional as F


class SRNet(nn.Module):

    def __init__(self, twin_net: TwinBranchNets, num_classes):
        super(SRNet, self).__init__()
        self.probabilistic = True
        self.feature_extractor = twin_net.feature_extractor

        self.z_dim = int(twin_net.z_dim / 2)
        self.classifier = mlp(in_dim=self.z_dim, out_dim=num_classes)

    def forward(self, x):
        z = self.featurize(x)
        x = self.classifier(z)
        return x

    def featurize(self, x, num_samples=1, return_dist=False):
        if not self.probabilistic:
            return self.feature_extractor(x)
        else:
            z_params = self.feature_extractor(x)
            z_mu = z_params[:, :self.z_dim]
            z_sigma = F.softplus(z_params[:, self.z_dim:])
            z_dist = distributions.Independent(distributions.normal.Normal(z_mu, z_sigma), 1)
            z = z_dist.rsample([num_samples]).view([-1, self.z_dim])

            if return_dist:
                return z, (z_mu, z_sigma)
            else:
                return z
