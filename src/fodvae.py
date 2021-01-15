import torch
from random import choices
import torch.nn as nn
import numpy as np
import itertools as it
import pytorch_lightning as pl
from models import MLPEncoder, MLP
from resnet import ResNetEncoder
from utils import (
    loss_representation,
    loss_entropy_binary,
    sample_reparameterize,
    KLD,
    reshape_tensor,
)


class FODVAE(pl.LightningModule):
    def __init__(
        self,
        encoder,
        discriminator_target,
        discriminator_sensitive,
        z_dim,
        dataset,
        **kwargs
    ):
        super().__init__()
        self.prior_mean_target = torch.ones(z_dim)
        self.prior_mean_target[int(z_dim / 2) :] = 0
        self.prior_mean_sensitive = -(self.prior_mean_target - 1)
        self.prior_mean_target = self.prior_mean_target / sum(
            self.prior_mean_target ** 2
        )
        self.prior_mean_sensitive = self.prior_mean_sensitive / sum(
            self.prior_mean_sensitive ** 2
        )
        self.encoder = encoder
        self.discriminator_target = discriminator_target
        self.discriminator_sensitive = discriminator_sensitive
        self.dataset = dataset
        param2default = {
            "lambda_od": 0.036,
            "lambda_entropy": 0.55,
            "gamma_od": 0.8,
            "gamma_entropy": 1.33,
            "step_size": 1000,
        }
        for param, default in param2default.items():
            setattr(self, param, kwargs.get(param, default))

    def forward(self, x):
        """
        Inputs:
            x - Something that fits in our network (batch_size x dim)
        Outputs:
            mean - Tensor of shape [B,z_dim] representing the predicted mean of
              the latent distributions.
            log_std - Tensor of shape [B,z_dim] representing the predicted log
              standard deviation of the latent distributions.
        """
        # Remark: Make sure to understand why we are predicting the log_std and
        # not std
        # x = x.view(x.shape[0], self.input_dim)
        (
            mean_target,
            log_std_target,
            mean_sensitive,
            log_std_sensitive,
        ) = self.encoder(x)
        # target
        std_target = torch.exp(log_std_target)
        std_sensitive = torch.exp(log_std_sensitive)
        return (
            mean_target,
            std_target,
            mean_sensitive,
            std_sensitive,
        )

    def yield_sensitive_repr_parameters(self):
        """Return generator with parameters that should be updated according to the
        sensitive representation loss, assuming that the second half of the
        encoder output is used as sensitive latent approximated posterior
        distribution parameters
        """
        final_linear_enc = self.encoder.net[-2]
        w, b = list(final_linear_enc.parameters())
        rel_enc_w = nn.Parameter(w[int(w.shape[0] / 2) :, :])
        rel_enc_b = nn.Parameter(b[int(b.shape[0] / 2) :])
        return (
            params
            for params in (
                rel_enc_w,
                rel_enc_b,
                *self.discriminator_sensitive.parameters(),
            )
        )

    def configure_optimizers(self):
        # Optimizer for CIFAR datasets
        if self.dataset in {"cifar10", "cifar100"}:
            optim_encoder = torch.optim.Adam(
                self.encoder.parameters(), lr=1e-4, weight_decay=1e-2
            )
            disc_params = list(self.discriminator_target.parameters()) + list(
                self.discriminator_sensitive.parameters()
            )
            optim_disc = torch.optim.Adam(
                disc_params, lr=1e-2, weight_decay=5e-2
            )

            return optim_encoder, optim_disc

        # Optimizer for YaleB dataset
        elif self.dataset == "yaleb":
            optim = torch.optim.Adam(
                self.parameters(),
                lr=1e-4,  # weight_decay=5 * (10 ** -2)
            )
            return optim
        # Optimizer for Adult and German datasets
        elif self.dataset in {"adult", "german"}:
            optim = torch.optim.Adam(
                self.parameters(), lr=1e-3, weight_decay=5e-4
            )
            return optim

    def manual_backward(self, loss, retain_graph=False):
        loss.backward(retain_graph=retain_graph)

    @property
    def automatic_optimization(self):
        return False

    def set_grad_target_encoder(self, to):
        """Sets requires_grad of the parameters of the target encoder accordingly"""
        params = list(self.encoder.parameters())
        if self.dataset in {"german", "adult"}:
            for param in params[:-2]:
                param.requires_grad = to
        # if self.dataset in {"cifar10", "cifar100"}:
        #     print(params[:-2][)
        # param.requires_grad = to
        # w, b = list(self.encoder.parameters())[-2:]
        # target_w = nn.Parameter(w[: int(w.shape[0] / 2), :]).requires_grad = to
        # target_b = nn.Parameter(b[: int(b.shape[0] / 2)]).requires_grad = to

    def decay_lambdas(self):
        self.lambda_od *= self.gamma_od ** (
            self.current_epoch / self.step_size
        )
        self.lambda_entropy *= self.gamma_entropy ** (
            self.current_epoch / self.step_size
        )
        # print(self.lambda_od, self.lambda_entropy)

    def training_epoch_end(self, outputs):
        self.decay_lambdas()

    def accuracy(self, y, y_pred):
        y = reshape_tensor(y)
        y_pred = reshape_tensor(y_pred)
        acc = (y == y_pred).float().mean().item()
        return acc

    def update_total_nof_batches(self, batch_idx):
        if self.current_epoch == 0 and batch_idx == 0:
            self.total_nof_batches = 0
        else:
            self.total_nof_batches += 1

    def set_logger(self, logger):
        if logger is not None:
            self.logger = logger

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        self.update_total_nof_batches(batch_idx)
        # self.decay_lambdas()
        optim_all = self.optimizers()
        # X, y = batch
        X, y, s = batch
        (
            mean_target,
            std_target,
            mean_sensitive,
            std_sensitive,
        ) = self.forward(X)
        # print(mean_target)
        # Sample from latent distributions
        sample_target = sample_reparameterize(mean_target, std_target)
        sample_sensitive = sample_reparameterize(mean_sensitive, std_sensitive)
        # Predict using discriminators
        pred_y = self.discriminator_target(sample_target).squeeze()
        # print("sample_sensitive", sample_sensitive)
        pred_s = self.discriminator_sensitive(sample_sensitive).squeeze()
        # Compute "crossover" posterior, i.e. to what extent is the sensitive
        # discriminator able to predict the sensitive attribute based on the
        # target latent representation
        crossover_posterior = self.discriminator_sensitive(sample_target)

        ## Compute losses
        # Representation losses
        loss_repr_target = loss_representation(pred_y, y.float()).mean()
        loss_repr_sensitive = loss_representation(pred_s, s.float()).mean()
        # OD losses
        loss_od_target = KLD(
            mean_target, std_target, self.prior_mean_target
        ).mean()
        loss_od_sensitive = KLD(
            mean_sensitive, std_sensitive, self.prior_mean_sensitive
        ).mean()
        loss_od = loss_od_target + loss_od_sensitive
        # Entropy loss
        # print("crossover_posterior", crossover_posterior)
        loss_entropy = loss_entropy_binary(crossover_posterior).mean()

        optim_all = optim_all if type(optim_all) == list else [optim_all]
        [optim.zero_grad() for optim in optim_all]
        # Freeze target encoder
        self.set_grad_target_encoder(False)
        # Backprop sensitive representation loss
        loss_repr_sensitive.backward(retain_graph=True)
        # Unfreeze target encoder
        self.set_grad_target_encoder(True)
        # Backprop remaining loss
        remaining_loss = (
            loss_repr_target
            + self.lambda_od * loss_od
            + self.lambda_entropy * loss_entropy
        )
        remaining_loss.backward()

        loss_total = loss_repr_sensitive + remaining_loss

        train_target_acc = self.accuracy(y, pred_y)
        train_sens_acc = self.accuracy(s, pred_s)

        use_logger = hasattr(self, "logger") and self.logger is not None
        if use_logger:
            self.logger.log_metrics(
                {
                    "train_loss_total": loss_total.item(),
                    "train_loss_od": loss_od.item(),
                    "train_loss_entropy": loss_entropy.item(),
                    "train_loss_repr_sensitive": loss_repr_sensitive.item(),
                    "train_loss_repr_target": loss_repr_target.item(),
                    "train_target_acc": train_target_acc,
                    "train_sens_acc": train_sens_acc,
                }
            )

        # Step for all optimizers
        for optim in optim_all:
            optim.step()

        if batch_idx == 0 and not use_logger:
            PRECISION = 3
            print(
                "\n{:<20}{:>5}".format(
                    "loss_repr_sensitive",
                    round(loss_repr_sensitive.item(), PRECISION),
                )
            )
            print(
                "{:<20}{:>5}".format(
                    "loss_repr_target",
                    round(loss_repr_target.item(), PRECISION),
                )
            )
            print(
                "{:<20}{:>5}".format(
                    "loss_od",
                    round(self.lambda_od * loss_od.item(), PRECISION),
                )
            )
            print(
                "{:<20}{:>5}".format(
                    "loss_entropy",
                    round(
                        self.lambda_entropy * loss_entropy.item(), PRECISION
                    ),
                )
            )
            loss = loss_repr_sensitive.item() + remaining_loss.item()
            print("{:<20}{:>5}".format("loss", round(loss, PRECISION)))
