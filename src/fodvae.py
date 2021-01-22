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
    accuracy,
    current_device,
)


class FODVAE(pl.LightningModule):
    def __init__(
        self,
        encoder,
        discriminator_target,
        discriminator_sensitive,
        z_dim,
        dataset,
        loss_components=["entropy", "kl", "orth"],
        **kwargs,
    ):
        print(loss_components)
        super().__init__()
        self.z_dim = z_dim
        self.encoder = encoder
        self.discriminator_target = discriminator_target
        self.discriminator_sensitive = discriminator_sensitive
        self.dataset = dataset
        # Configure hyperparameters that have defaults
        hparams = {
            "gamma_entropy",
            "gamma_od",
            "lambda_entropy",
            "lambda_od",
            "step_size",
            "encoder_lr",
            "encoder_weight_decay",
            "discs_lr",
            "discs_weight_decay",
        }
        for param in hparams:
            param_val = kwargs.get(param)
            if param_val is None:
                raise Exception(f"FODVAE missing required value for {param}")
            setattr(self, param, param_val)
        self.loss_components = loss_components
        self.lambda_od_initial = 1 * self.lambda_od
        self.lambda_entropy_initial = 1 * self.lambda_entropy
        print(f"Using {loss_components}")
        self._init_prior_means()

    def _init_prior_means(self):
        if "orth" in self.loss_components:
            print("yes orth")
            self.prior_mean_target = torch.ones(self.z_dim).to(
                current_device()
            )
            self.prior_mean_target[int(self.z_dim / 2) :] = 0
            self.prior_mean_sensitive = -(self.prior_mean_target - 1)
            assert self.prior_mean_sensitive.dot(self.prior_mean_target) == 0
        else:
            print("no orth")
            # come back later
            self.prior_mean_target = torch.ones(self.z_dim).to(
                current_device()
            )
            self.prior_mean_target[int(self.z_dim / 2) :] = -1
            self.prior_mean_sensitive = self.prior_mean_target[
                :
            ]  # -(self.prior_mean_target - 1)
            assert self.prior_mean_sensitive.dot(self.prior_mean_target) != 0
            assert (self.prior_mean_sensitive == self.prior_mean_target).all()
        self.prior_mean_target = self.prior_mean_target / sum(
            self.prior_mean_target ** 2
        )
        self.prior_mean_sensitive = self.prior_mean_sensitive / sum(
            self.prior_mean_sensitive ** 2
        )

    def encode(self, x):
        """
        Encode input as distribution in latent representation space.

        Inputs: x - Something that fits in our network (batch_size x dim)

        Outputs:
        target_mean - Tensor of shape [B,z_dim] representing the predicted mean
        of the target partition of the latent representation.
        std_target - Tensor of shape [B,z_dim]
        representing the predicted standard deviation of the target partition
        of the latent representation.
        sensitive_mean - Tensor of shape [B,z_dim] representing the predicted
        mean of the sensitive partition of the latent representation.
        std_sensitive - Tensor of shape [B,z_dim]
        representing the predicted standard deviation of the sensitive partition
        of the latent representation.
        """
        (
            mean_target,
            log_std_target,
            mean_sensitive,
            log_std_sensitive,
        ) = self.encoder(x)

        std_target = torch.exp(log_std_target)
        std_sensitive = torch.exp(log_std_sensitive)
        return (
            mean_target,
            std_target,
            mean_sensitive,
            std_sensitive,
        )

    # def yield_sensitive_repr_parameters(self):
    #     """Return generator with parameters that should be updated according to the
    #     sensitive representation loss, assuming that the second half of the
    #     encoder output is used as sensitive latent approximated posterior
    #     distribution parameters
    #     """
    #     final_linear_enc = self.encoder.net[-2]
    #     w, b = list(final_linear_enc.parameters())
    #     rel_enc_w = nn.Parameter(w[int(w.shape[0] / 2) :, :])
    #     rel_enc_b = nn.Parameter(b[int(b.shape[0] / 2) :])
    #     return (
    #         params
    #         for params in (
    #             rel_enc_w,
    #             rel_enc_b,
    #             *self.discriminator_sensitive.parameters(),
    #         )
    #     )

    def configure_optimizers(self):
        # Optimizer for CIFAR datasets
        optim_encoder = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.encoder_lr,
            weight_decay=self.encoder_weight_decay,
        )
        disc_params = list(self.discriminator_target.parameters()) + list(
            self.discriminator_sensitive.parameters()
        )
        optim_discs = torch.optim.Adam(
            disc_params, lr=self.discs_lr, weight_decay=self.discs_weight_decay
        )
        # return [
        #     torch.optim.Adam(self.parameters()),
        # ]
        return optim_encoder, optim_discs

        # # Optimizer for YaleB dataset
        # elif self.dataset == "yaleb":
        #     optim = torch.optim.Adam(
        #         self.parameters(), lr=10 ** -4, weight_decay=5 * (10 ** -2)
        #     )
        #     return optim

        # # Optimizer for Adult and German datasets
        # elif self.dataset in {"adult", "german"}:
        #     optim = torch.optim.Adam(
        #         self.parameters(), lr=1e-3, weight_decay=5e-4
        #     )
        #     return optim

    automatic_optimization = False

    def decay_lambdas(self):
        # Every step_size epochs, the lambda should be increased by
        # a factor denoted by corresponding gamma.
        progress = self.current_epoch / self.step_size
        self.lambda_od = self.lambda_od_initial * self.gamma_od ** progress
        self.lambda_entropy = (
            self.lambda_entropy_initial * self.gamma_entropy ** progress
        )

    def training_epoch_end(self, outputs):
        self.decay_lambdas()

    def accuracy(self, y, y_pred):
        return accuracy(y, y_pred)

    def update_total_nof_batches(self, batch_idx):
        if self.current_epoch == 0 and batch_idx == 0:
            self.total_nof_batches = 0
        else:
            self.total_nof_batches += 1

    def set_logger(self, logger):
        if logger is not None:
            self.logger = logger

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        # Keep track of how many batches we've trained with
        self.update_total_nof_batches(batch_idx)

        optimizers = self.optimizers()
        optimizers = optimizers if type(optimizers) == list else [optimizers]
        X, y, s = batch

        ##############################
        ## Forward through models
        ##############################

        # Encode X to latent representation
        (
            mean_target,
            std_target,
            mean_sensitive,
            std_sensitive,
        ) = self.encode(X)

        # Sample from latent distributions
        sample_target = sample_reparameterize(mean_target, std_target)
        sample_sensitive = sample_reparameterize(mean_sensitive, std_sensitive)

        # Predict using discriminators
        pred_y = self.discriminator_target(sample_target).squeeze()
        # print("\n")
        # print(pred_y)
        # print("sample_sensitive", sample_sensitive)
        pred_s = self.discriminator_sensitive(sample_sensitive).squeeze()

        # Compute "crossover" posterior, i.e. to what extent is the sensitive
        # discriminator able to predict the sensitive attribute based on the
        # target latent representation
        crossover_posterior = self.discriminator_sensitive(sample_target)

        ##############################
        ## Compute losses
        ##############################

        # Representation losses
        loss_repr_target = loss_representation(
            pred_y, y.float(), reduction="mean"
        )
        loss_repr_sensitive = loss_representation(
            pred_s, s.float(), reduction="mean"
        )

        # OD losses
        if "kl" in self.loss_components:
            loss_od_target = KLD(
                mean_target, std_target, self.prior_mean_target
            ).mean()
            loss_od_sensitive = KLD(
                mean_sensitive, std_sensitive, self.prior_mean_sensitive
            ).mean()
            loss_od = loss_od_target + loss_od_sensitive
            if torch.isnan(loss_od):
                breakpoint()
        else:
            loss_od = torch.zeros(1).to(current_device())

        # Entropy loss
        if "entropy" in self.loss_components:
            # print("crossover_posterior", crossover_posterior)
            loss_entropy = loss_entropy_binary(crossover_posterior).mean()
        else:
            loss_entropy = torch.zeros(1).to(current_device())

        ##############################
        ## Optimization
        ##############################

        # Reset gradients
        for optimizer in optimizers:
            optimizer.zero_grad()

        # Backprop sensitive representation loss

        # Backprop remaining loss
        remaining_loss = (
            loss_repr_target
            + self.lambda_od * loss_od
            + self.lambda_entropy * loss_entropy
        )
        remaining_loss.backward()

        # Step for all optimizers
        for optimizer in optimizers:
            optimizer.step()

        ##############################
        ## Logging
        ##############################

        loss_total = loss_repr_sensitive.item() + remaining_loss.item()

        train_target_acc = self.accuracy(y, pred_y)
        train_sens_acc = self.accuracy(s, pred_s)
        train_sens_crossover_acc = self.accuracy(s, crossover_posterior)

        use_logger = hasattr(self, "logger") and self.logger is not None
        if use_logger:
            # Log to WandbLogger
            self.logger.log_metrics(
                {
                    "train_loss_total": loss_total,
                    "train_loss_od": loss_od.item(),
                    "train_loss_entropy": loss_entropy.item(),
                    "train_loss_repr_sensitive": loss_repr_sensitive.item(),
                    "train_loss_repr_target": loss_repr_target.item(),
                    "train_target_acc": train_target_acc,
                    "train_sens_acc": train_sens_acc,
                    "train_sens_crossover_acc": train_sens_crossover_acc,
                    "lambda_od": self.lambda_od,
                    "lambda_entropy": self.lambda_entropy,
                }
            )

        if batch_idx % 10 == 0 and not use_logger:
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
            print(
                "{:<20}{:>5}".format(
                    "train_target_acc",
                    round(train_target_acc, PRECISION),
                )
            )
            print(
                "{:<20}{:>5}".format(
                    "train_sens_acc",
                    round(train_sens_acc, PRECISION),
                )
            )
            print(
                "{:<20}{:>5}".format(
                    "train_sens_crossover_acc",
                    round(train_sens_crossover_acc, PRECISION),
                )
            )
            print("{:<20}{:>5}".format("loss", round(loss_total, PRECISION)))
