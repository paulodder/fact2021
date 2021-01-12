import torch
from random import choices
import torch.nn as nn
import numpy as np
import itertools as it
import pytorch_lightning as pl
from models import MLPEncoder, MLP
from utils import (
    loss_representation,
    loss_entropy_binary,
    sample_reparameterize,
    KLD,
)


class FODVAE(pl.LightningModule):
    def __init__(
        self,
        encoder,
        discriminator_target,
        discriminator_sensitive,
        z_dim,
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

        # return (
        #     mean_target,
        #     std_target,
        #     pred_target,
        #     mean_sensitive,
        #     std_sensitive,
        #     pred_sensitive,
        #     crossover_posterior,
        # )

        # # # print(torch.log(self.std(x)))
        # # log_std = self.std(x)
        # return mean, log_std

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
        optim_all = torch.optim.Adam(
            self.parameters(), lr=1 * 10e-4, weight_decay=5 * 10e-2
        )
        # optim_all = torch.optim.Adam(
        #     self.parameters(), lr=1 * 10e-3, weight_decay=5 * 10e-4
        # )
        # only concerns itself with parameters
        return optim_all
        # optim_sensitive = torch.optim.Adam(
        #     self.yield_sensitive_repr_parameters(),
        #     lr=1 * 10e-3,
        #     weight_decay=5 * 10e-4,
        # )
        # return [optim_all, optim_sensitive]
        #

    def manual_backward(self, loss, retain_graph=False):
        loss.backward(retain_graph=retain_graph)

    @property
    def automatic_optimization(self):
        return False

    def set_grad_target_encoder(self, to):
        """Sets requires_grad of the parameters of the target encoder accordingly"""
        params = list(self.encoder.parameters())
        for param in params[:-2]:
            param.requires_grad = to
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
        print(self.lambda_od, self.lambda_entropy)

    def training_epoch_end(self, outputs):
        self.decay_lambdas()

    def update_total_nof_batches(self, batch_idx):
        if self.current_epoch == 0 and batch_idx == 0:
            self.total_nof_batches = 0
        else:
            self.total_nof_batches += 1

    def training_step(self, batch, batch_idx):
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
        loss_od = self.lambda_od * (loss_od_target + loss_od_sensitive)
        # Entropy loss
        # print("crossover_posterior", crossover_posterior)
        loss_entropy = self.lambda_entropy * (
            loss_entropy_binary(crossover_posterior).mean()
        )
        optim_all.zero_grad()
        # Freeze target encoder
        self.set_grad_target_encoder(False)
        # Backprop sensitive representation loss
        loss_repr_sensitive.backward(retain_graph=True)
        # Unfreeze target encoder
        self.set_grad_target_encoder(True)
        # Backprop remaining loss
        remaining_loss = loss_repr_target + loss_od + loss_entropy
        remaining_loss.backward()
        optim_all.step()

        if batch_idx % 100 == 0 or batch_idx == 1:
            print("loss_repr_sensitive\t", loss_repr_sensitive.item())
            print("loss_repr_target\t", loss_repr_target.item())
            print("loss_od\t", loss_od.item())
            print("loss_entropy\t", loss_entropy.item())
            loss = loss_repr_sensitive.item() + remaining_loss.item()
            print("loss", loss)


def get_sensitive_discriminator(args):
    if args.dataset in {"adult", "german"}:
        model = MLP(
            input_dim=args.z_dim,
            hidden_dims=[64, 64],
            output_dim=1,
            nonlinearity=nn.Sigmoid,
        )
    elif args.dataset == "yaleb":
        model = MLP(
            input_dim=args.z_dim,
            hidden_dims=[100, 100],
            output_dim=65,
            nonlinearity=nn.Softmax,
        )
    return model


def get_fodvae(args):
    "gets FODVAE according to args"
    if args.dataset == "adult":
        input_dim = 108
        encoder = MLPEncoder(input_dim=input_dim, z_dim=args.z_dim)
        disc_target = MLP(
            input_dim=args.z_dim,
            hidden_dims=[64, 64],
            output_dim=1,
            nonlinearity=nn.Sigmoid,
        )
        disc_sensitive = get_sensitive_discriminator(args)
        fvae = FODVAE(
            encoder,
            disc_target,
            disc_sensitive,
            lambda_od=0.036,
            lambda_entropy=0.55,
            gamma_od=0.8,
            gamma_entropy=1.33,
            step_size=1000,
            z_dim=args.z_dim,
        )
        return fvae
    elif args.dataset == "german":
        input_dim = 61
        encoder = MLPEncoder(input_dim=input_dim, z_dim=args.z_dim)
        disc_target = MLP(
            input_dim=args.z_dim,
            hidden_dims=[64, 64],
            output_dim=1,
            nonlinearity=nn.Sigmoid,
        )
        disc_sensitive = get_sensitive_discriminator(args)
        fvae = FODVAE(
            encoder,
            disc_target,
            disc_sensitive,
            lambda_od=0.036,
            lambda_entropy=0.55,
            gamma_od=0.8,
            gamma_entropy=1.33,
            step_size=1000,
            z_dim=args.z_dim,
        )
        return fvae
    elif args.dataset == "yaleb":
        input_dim = 32256
        encoder = MLPEncoder(
            input_dim=input_dim, hidden_dims=[], z_dim=args.z_dim
        )
        disc_target = MLP(
            input_dim=args.z_dim,
            hidden_dims=[100, 100],
            output_dim=38,
            nonlinearity=nn.Softmax,
        )
        disc_sensitive = get_sensitive_discriminator(args)
        # fvae = FODVAE(
        #     encoder,
        #     disc_target,
        #     disc_sensitive,
        #     lambda_od=0.036,
        #     lambda_entropy=0.5,
        #     gamma_od=0.8,
        #     gamma_entropy=1.33,
        #     step_size=1000,
        #     z_dim=args.z_dim,
        # )
        fvae = FODVAE(
            encoder,
            disc_target,
            disc_sensitive,
            lambda_od=0.1,
            lambda_entropy=0.1,
            gamma_od=0.8,
            gamma_entropy=1.33,
            step_size=1000,
            z_dim=args.z_dim,
        )

        return fvae
