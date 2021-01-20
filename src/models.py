import torch
import torchvision
import torch.nn as nn
import itertools as it
import pytorch_lightning as pl
import torchvision.models


class LinearToLatentRepresentation(nn.Module):
    def __init__(self, input_dim, z_dim):
        super().__init__()
        self.z_dim = z_dim
        output_dim_half = self.z_dim * 2
        self.linear_target = nn.Linear(input_dim, output_dim_half)
        self.linear_sens = nn.Linear(input_dim, output_dim_half)

    def forward(self, x):
        """This linear layer outputs a tuple of four tensors:
        - target_mean
        - log_target_std
        - sensitive_mean
        - log_sensitive_std
        The sensitive mean and log-std have been detached w.r.t.
        the input x, such that their gradients will only affect
        the Parameters in the linear_sens layer.
        """
        out_target = self.linear_target(x)
        x_detached = x.detach()
        out_sens = self.linear_sens(x_detached)
        return (
            out_target[:, : self.z_dim],  # target_mean
            out_target[:, self.z_dim :],  # log_target_std
            out_sens[:, : self.z_dim],  # sens_mean
            out_sens[:, self.z_dim :],  # log_sens_std
        )


class MLPEncoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[64], z_dim=2):
        super().__init__()

        dims = [input_dim] + hidden_dims
        modules = []
        for i in range(len(dims) - 1):
            modules.append(
                nn.Linear(dims[i], dims[i + 1]),
            )
            # Always append nonlinear layer because
            # we still append a final linear layer after this
            # for-loop.
            modules.append(nn.ReLU())
        modules.append(LinearToLatentRepresentation(dims[-1], z_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, X):
        return self.net(X)


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
        self.net = torchvision.models.resnet18(pretrained=True)
        for params in self.parameters():
            params.requires_grad = continue_training
        fc_size = list(self.net.children())[-1].in_features
        self.net.fc = LinearToLatentRepresentation(fc_size, z_dim)

    def forward(self, X):
        return self.net(X)
