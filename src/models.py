import torch
import torchvision
import torch.nn as nn
import itertools as it
import pytorch_lightning as pl
import torchvision.models


class MLPEncoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[64], z_dim=2):
        super().__init__()
        self.z_dim = z_dim
        output_dim = z_dim * 4  # 2 means and 2 covariances for each dim
        layers = list(
            it.chain.from_iterable(
                [
                    (nn.Linear(inp_dim, out_dim), nn.ReLU())
                    for (inp_dim, out_dim) in zip(
                        [input_dim] + hidden_dims,
                        hidden_dims + [output_dim],
                    )
                ]
            )
        )
        layers = layers[:-1]
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        vals = self.net(X)
        return [
            vals[:, i * self.z_dim : (i + 1) * self.z_dim] for i in range(4)
        ]


class MLP(nn.Module):
    def __init__(
        self,
        input_dim=2,
        hidden_dims=[64, 64],
        output_dim=1,
        batch_norm=False,
        nonlinearity=nn.Sigmoid,
    ):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        modules = []
        for i in range(len(dims) - 1):
            modules.append(
                nn.Linear(dims[i], dims[i + 1]),
            )
            if i < len(dims) - 2:
                modules.append(nn.ReLU())
                if batch_norm:
                    modules.append(nn.BatchNorm1d(dims[i + 1]))
        self.net = nn.Sequential(*modules)
        self.nonlinear = (
            nonlinearity(dim=1)
            if nonlinearity is nn.Softmax
            else nonlinearity()
        )

    def forward(self, X):
        return self.nonlinear(self.net(X))


class ResNetEncoder(nn.Module):
    def __init__(self, z_dim=2, continue_training=True):
        super().__init__()
        self.z_dim = z_dim
        output_dim = z_dim * 4  # 2 means and 2 covariances for each dim
        self.net = torchvision.models.resnet18(pretrained=True)
        for params in self.parameters():
            params.requires_grad = continue_training
        fc_size = list(self.net.children())[-1].in_features
        self.net.fc = nn.Linear(fc_size, output_dim)

    def param_mean(self):
        means = []
        for param in self.parameters():
            means.append(param.data.mean().item())
        return sum(means) / len(means)

    def forward(self, X):
        vals = self.net(X)
        return [
            vals[:, i * self.z_dim : (i + 1) * self.z_dim] for i in range(4)
        ]
