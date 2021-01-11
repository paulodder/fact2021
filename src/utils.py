import torch
import torch.nn as nn


def sample_reparameterize(mean, std):
    """
    Perform the reparameterization trick to sample from a distribution with the given mean and std
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
        std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
              of the distribution
    Outputs:
        z - A sample of the distributions, with gradient support for both mean and std.
            The tensor should have the same shape as the mean and std input tensors.
    """
    eps = torch.normal(0, 1, size=mean.shape)
    z = mean + (std * eps)
    return z


def bce_loss(x, y):
    return nn.functional.binary_cross_entropy(
        x.squeeze().float(), y.squeeze().float()
    )


def loss_representation(y_pred, y_true):
    # print("HERE", y_pred, y_true)
    out = nn.functional.binary_cross_entropy(y_pred, y_true, reduction="none")
    # print("HERE OUT", out)
    return out


# def loss_od(mean, std):
#     return nn.functional.binary_cross_entropy(y_pred, y_true)


def KLD(mean, std, mean_prior):
    "assumes prior unit variance"
    val = (
        torch.log(torch.ones_like(std) / (std + 1e-8))
        + (((std ** 2) + ((mean - mean_prior) ** 2)) / 2)
    ).sum(1) - 0.5
    return val


def loss_entropy_binary(crossover_posterior):
    """Given the sensitive approximated posterior conditioned on the target latent
    representation (i.e. p(s|z_{t})), returns the entropy"""
    other_prob = 1 - crossover_posterior
    return (
        other_prob * (other_prob + 1e-8).log()
        + (crossover_posterior + 1e-8).log() * crossover_posterior
    )
    # return (probs * probs.log()).sum(1)
