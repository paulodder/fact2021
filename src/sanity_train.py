import torch
import torch.nn as nn
import pytorch_lightning as pl
from dataloaders import load_data
from models import ResNetEncoder, MLP
from utils import (
    loss_representation,
    loss_entropy_binary,
    sample_reparameterize,
    KLD,
    reshape_tensor,
    current_device,
)


class SanityCIFAR10(pl.LightningModule):
    def __init__(self):
        super().__init__()
        z_dim = 4
        self.encoder = ResNetEncoder(z_dim=z_dim, continue_training=True)
        self.linear = nn.Linear(z_dim, 1)
        self.nonlinear = nn.Sigmoid()
        # self.discriminator_target = MLP(
        #     input_dim=z_dim,
        #     hidden_dims=[256, 128],
        #     output_dim=1,
        #     nonlinearity=nn.Sigmoid,
        # )
        # self.discriminator_target = nn.Sequential(
        #     nn.Linear(in_features=128, out_features=256, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=256, out_features=128, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=128, out_features=1, bias=True),
        #     nn.Sigmoid(),
        # )

    def configure_optimizers(self):
        # optim_encoder = torch.optim.Adam(
        #     self.encoder.parameters(), lr=10 ** -4, weight_decay=10 ** -2
        # )
        # disc_params = list(self.discriminator_target.parameters())
        # # + list(
        # # self.discriminator_sensitive.parameters()
        # # )
        # optim_disc = torch.optim.Adam(
        #     disc_params, lr=10 ** -2, weight_decay=10 ** -3
        # )

        # return optim_encoder, optim_disc
        return torch.optim.Adam(self.parameters())

    def accuracy(self, pred_y, y):
        correct = (pred_y > 0.5).squeeze().long() == y.squeeze().long()
        return correct.float().mean().item()

    automatic_optimization = False

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        optimizers = self.optimizers()
        if type(optimizers) is not list:
            optimizers = [optimizers]
        x, y, s = batch
        (
            mean_target,
            log_std_target,
            mean_sensitive,
            log_std_sensitive,
        ) = self.encoder(x)
        # std_target = log_std_target
        std_target = torch.exp(log_std_target)
        # std_sensitive = torch.exp(log_std_sensitive)
        sample = sample_reparameterize(mean_target, std_target)

        # not same outputs:
        pred_y = self.nonlinear(self.linear(sample).squeeze())
        # same outputs:
        # pred_y = self.discriminator_target(sample).squeeze()

        loss = loss_representation(pred_y, y.float()).mean()
        print()
        print(pred_y[:4])
        # print(y)
        print(self.accuracy(pred_y, y))
        loss.backward()

        for optim in optimizers:
            optim.step()


def main():
    torch.manual_seed(666)
    trainer = pl.Trainer(max_epochs=2)
    model = SanityCIFAR10()
    train_dl, val_dl = load_data("cifar10", batch_size=128)
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    main()
