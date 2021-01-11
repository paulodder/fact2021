import torch
import torch.nn as nn
import itertools as it
import pytorch_lightning as pl


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
                        [input_dim] + hidden_dims, hidden_dims + [output_dim],
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


class MLPDiscriminator(pl.LightningModule):
    def __init__(self, z_dim=2, hidden_dims=[64, 64], output_dim=1):
        super().__init__()
        layers = list(
            it.chain.from_iterable(
                [
                    (nn.Linear(inp_dim, out_dim), nn.ReLU())
                    for (inp_dim, out_dim) in zip(
                        [z_dim] + hidden_dims, hidden_dims + [output_dim]
                    )
                ]
            )
        )
        layers = layers[:-1]
        self.net = nn.Sequential(*layers)
        self.nonlinear = nn.Sigmoid()

    def forward(self, X):
        return self.nonlinear(self.net(X))

    @property
    def automatic_optimization(self):
        return False

    def training_step(self, batch, batch_idx):
        X, y, s = batch
        y_pred = self.forward(X)
        loss = loss_entropy_binary(y, y_pred)
        loss.backward()

    def configure_optimizers(self):
        optim_all = torch.optim.Adam(
            self.parameters(), lr=1 * 10e-4, weight_decay=5 * 10e-4
        )
        return optim_all
