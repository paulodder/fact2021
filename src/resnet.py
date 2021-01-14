import torchvision.models as models
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl


class ResNetEncoder(nn.Module):
    def __init__(self, z_dim=2):
        super().__init__()
        self.z_dim = z_dim
        output_dim = z_dim * 4  # 2 means and 2 covariances for each dim
        self.net = models.resnet18(pretrained=True)
        for params in self.parameters():
            params.requires_grad = False
        fc_size = list(self.net.children())[-1].in_features
        self.net.fc = nn.Linear(fc_size, output_dim)
        self.nonlinear = nn.Sigmoid()

    def forward(self, X):
        vals = self.nonlinear(self.net(X))
        return [
            vals[:, i * self.z_dim : (i + 1) * self.z_dim] for i in range(4)
        ]
