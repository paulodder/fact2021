import torch
import torch.nn as nn
import numpy as np
import itertools as it
import pytorch_lightning as pl
from utils import (
    # loss_od,
    loss_representation,
    loss_entropy_binary,
    sample_reparameterize,
    KLD,
)


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


class MLPDiscriminator(nn.Module):
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


class FODVAE(pl.LightningModule):
    def __init__(self, encoder, discriminator_target, discriminator_sensitive):
        super().__init__()
        self.prior_mean_target = torch.Tensor([0, 1])
        self.prior_mean_sensitive = torch.Tensor([1, 0])
        self.encoder = encoder
        self.discriminator_target = discriminator_target
        self.discriminator_sensitive = discriminator_sensitive
        self.lambda_od = 0.036
        self.lambda_entropy = 0.55
        self.gamma_od = 0.8
        self.gamma_entropy = 1.33
        self.step_size = 1000

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
            self.parameters(), lr=1 * 10e-4, weight_decay=5 * 10e-4
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
        # print(loss)
        loss.backward(retain_graph=retain_graph)

    @property
    def automatic_optimization(self):
        return False

    # def yield_target_encoder_params(self):
    #     "assumes final two parameter sets are last weights and last biases"

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
        X, y = batch["feat"], batch["label"]
        # tmp solution to get a sensitive value
        s = (X[:, 0] > 0).float()
        X = X[:, 1:]
        (
            mean_target,
            std_target,
            mean_sensitive,
            std_sensitive,
        ) = self.forward(X)
        # Sample from latent distributions
        sample_target = sample_reparameterize(mean_target, std_target)
        sample_sensitive = sample_reparameterize(mean_sensitive, std_sensitive)
        # Predict using discriminators
        pred_y = self.discriminator_target(sample_target).squeeze()
        pred_s = self.discriminator_sensitive(sample_sensitive).squeeze()
        # Compute "crossover" posterior, i.e. to what extent is the sensitive
        # discriminator able to predict the sensitive attribute based on the
        # target latent representation
        crossover_posterior = self.discriminator_sensitive(sample_target)

        ## Compute losses
        # Representation losses
        loss_repr_target = loss_representation(pred_y, y).mean()
        loss_repr_sensitive = loss_representation(pred_s, s).mean()
        # OD losses
        # print("mean_target", mean_target)
        # print("std_target", std_target)
        loss_od_target = KLD(
            mean_target, std_target, self.prior_mean_target
        ).mean()
        # print(
        #     "mean_target, std_target, self.prior_mean_target",
        #     mean_target,
        #     std_target,
        #     self.prior_mean_target,
        # )
        # print(
        #     "mean_sensitive, std_sensitive, self.prior_mean_sensitive",
        #     mean_sensitive,
        #     std_sensitive,
        #     self.prior_mean_sensitive,
        # )
        loss_od_sensitive = KLD(
            mean_sensitive, std_sensitive, self.prior_mean_sensitive
        ).mean()
        loss_od = self.lambda_od * (loss_od_target + loss_od_sensitive)
        # Entropy loss
        loss_entropy = self.lambda_entropy * (
            loss_entropy_binary(crossover_posterior).squeeze().mean()
        )
        optim_all.zero_grad()
        # Freeze target encoder
        # self.set_grad_target_encoder(False)
        # Backprop sensitive representation loss
        loss_repr_sensitive.backward(retain_graph=True)
        # Unfreeze target encoder
        # self.set_grad_target_encoder(True)
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

        # print((y == (pred_y > 0.5).int()).float().mean())
        # print((y == pred_y).sum() / len(pred_y))
        # representation losses
        # print("pred_y", pred_y.shape)
        # print("pred_s", pred_s.shape)

        # print("loss_repr_target", loss_repr_target.shape)
        # print("loss_repr_target", loss_repr_target)
        # loss_repr_sensitive = loss_representation(pred_s, s)
        # # print("loss_repr_sensitive", loss_repr_sensitive.shape)
        # # print("loss_repr_sensitive", loss_repr_sensitive)
        # # od losses
        # loss_od_target = KLD(mean_target, std_target, self.prior_mean_target)
        # # print("loss_od_target", loss_od_target.shape)
        # loss_od_sensitive = KLD(
        #     mean_sensitive, std_sensitive, self.prior_mean_sensitive
        # )
        # # print("loss_od_sensitive", loss_od_sensitive.shape)
        # loss_entropy = loss_entropy_binary(crossover_posterior).squeeze()
        # # print("loss_entropy", loss_entropy.shape)

        # remaining_loss = (
        #     loss_repr_sensitive.sum()
        #     + loss_repr_target.sum()
        #     # + loss_od_target.sum()
        #     # + loss_od_sensitive.sum()
        #     + loss_entropy.sum()
        # )
        # print(
        #     (
        #         loss_repr_target
        #         + loss_od_target
        #         + loss_od_sensitive
        #         + loss_entropy
        #     )
        # )
        # print("loss", remaining_loss.sum())

        # self.manual_backward(loss_repr_sensitive.sum(), retain_graph=True)
        # self.set_requires_grad_theta_t(False)
        # optim_sensitive.step()
        # self.manual_backward(remaining_loss.mean())
        # self.set_requires_grad_theta_t(True)
        # optim_sensitive.zero_grad()
        # optim_sensitive.zero_grad()

    def set_requires_grad_theta_t(self, to=False):
        for p in list(self.encoder.parameters()):
            p.requires_grad = to
        for p in self.yield_sensitive_repr_parameters():
            p.requires_grad = True

        #     "train_reconstruction_loss", L_rec, on_step=False, on_epoch=True
        # )
        # self.log(
        #     "train_regularization_loss", L_reg, on_step=False, on_epoch=True
        # )
        # return {
        #     "remaining_loss": loss,
        #     "loss_repr_sensitive": loss_repr_sensitive,
        # }
