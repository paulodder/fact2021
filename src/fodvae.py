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
import dataloaders
import copy


class BestModelTracker:
    def __init__(self, model, dataset):
        self.model = model
        self.best_model = None
        self.best_performance = 0
        self.best_epoch = None
        self.__current_epoch = 0

        _, test_dl = dataloaders.load_data(dataset, "all")
        x, y, s = next(iter(test_dl))
        self.x = x
        self.x.requires_grad = False
        self.y = y
        self.s = s

    def end_of_epoch(self):
        self.model.eval()
        y_pred, s_pred, s_pred_crossover = self.model.forward(self.x)
        model_performance = (
            1.5 * accuracy(self.y, y_pred)
            + accuracy(self.s, s_pred)
            - accuracy(self.s, s_pred_crossover)
        )

        if model_performance > self.best_performance:
            print(
                f"\n[bmt] new best model @ epoch {self.__current_epoch} with sum of accs {round(model_performance, 4)}"
            )
            self.best_model = copy.deepcopy(self.model)
            self.best_performance = model_performance
            self.best_epoch = self.__current_epoch
        self.__current_epoch += 1

        self.model.train()


def get_encodings(model, x, get_target):
    mt, st, ms, ss = model.encode(x.view(1, -1))
    target = sample_reparameterize(mt, st)
    if not get_target:
        target = sample_reparameterize(ms, ss)
    return target


ds_train = dataloaders.dataset_registrar["german"](True)
ds_test = dataloaders.dataset_registrar["german"](False)

import matplotlib.pyplot as plt


def plot_embs(ax, model, train=True, target=True):
    ds = ds_train if train else ds_test
    embs = torch.stack(
        [get_encodings(model, ds[i][0], target) for i in range(len(ds))], dim=0
    ).squeeze()
    index = 1 if target else 2
    labels = np.array([ds[i][index].item() for i in range(len(ds))])

    al, bl = set(labels)
    emb_mat = embs.T.detach()
    ax.scatter(*emb_mat[:, labels == al], label=al, color="red")
    ax.scatter(*emb_mat[:, labels == bl], label=bl, color="blue")
    ax.set_title(f"train={train}, target={target}")
    plt.legend()


def plot_all_embs(model):
    model.eval()
    plt.figure()
    fig, axs = plt.subplots(2, 2)
    axs = axs.ravel()
    plot_embs(axs[0], model, train=True, target=True)
    plot_embs(axs[1], model, train=True, target=False)
    plot_embs(axs[2], model, train=False, target=True)
    plot_embs(axs[3], model, train=False, target=False)
    model.train()


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
        self.best_model_tracker = BestModelTracker(self, dataset)
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
            self.prior_mean_target = torch.ones(self.z_dim).to(
                current_device()
            )
            self.prior_mean_target[int(self.z_dim / 2) :] = 0
            self.prior_mean_sensitive = 1 - self.prior_mean_target
            # breakpoint()
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

    def get_best_version(self):
        print()
        print(
            f"[fodvae] best version @ epoch {self.best_model_tracker.best_epoch}"
        )
        print()
        return self.best_model_tracker.best_model

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
        progress = (self.current_epoch + 1) / self.step_size
        self.lambda_od = self.lambda_od_initial * self.gamma_od ** progress
        self.lambda_entropy = (
            self.lambda_entropy_initial * self.gamma_entropy ** progress
        )

    def training_epoch_end(self, outputs):
        self.best_model_tracker.end_of_epoch()
        # plot_all_embs(self)
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

    def forward(self, x):
        # Encode X to latent representation
        (
            mean_target,
            std_target,
            mean_sensitive,
            std_sensitive,
        ) = self.encode(x)

        # Sample from latent distributions
        sample_target = sample_reparameterize(mean_target, std_target)
        sample_sensitive = sample_reparameterize(mean_sensitive, std_sensitive)

        # Predict using discriminators
        pred_y = self.discriminator_target(sample_target).squeeze()
        pred_s = self.discriminator_sensitive(sample_sensitive).squeeze()
        crossover_posterior = self.discriminator_sensitive(sample_target)

        return pred_y, pred_s, crossover_posterior

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
        # loss_repr_sensitive.backward(retain_graph=True)
        # Backprop remaining loss
        total_loss = (
            loss_repr_sensitive
            + loss_repr_target
            + self.lambda_od * loss_od
            + self.lambda_entropy * loss_entropy
        )
        total_loss.backward()

        # Step for all optimizers
        for optimizer in optimizers:
            optimizer.step()

        loss_total = total_loss.item()
        train_target_acc = self.accuracy(y, pred_y)
        train_sens_acc = self.accuracy(s, pred_s)
        train_sens_crossover_acc = self.accuracy(s, crossover_posterior)

        ##############################
        ## Logging
        ##############################
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
