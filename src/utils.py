import math
import torch
import torch.nn as nn
from dotenv import dotenv_values, find_dotenv
from pathlib import Path
import re
import argparse

DOTENV = dotenv_values(find_dotenv())
DATA_DIR = Path(DOTENV["DATA_DIR"])


def current_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu:0"


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
    eps = torch.normal(0, 1, size=mean.shape).to(current_device())
    z = mean + (std * eps)
    return z


def bce_loss(x, y):
    return nn.functional.binary_cross_entropy(x.float(), y.float())


def loss_representation(y_pred, y_true):
    out = nn.functional.binary_cross_entropy(y_pred, y_true, reduction="none")
    return out


def KLD(mean, std, mean_prior, eps=1e-8):
    "assumes prior unit variance"
    unit_std = torch.ones(size=std.size()).float().to(current_device())
    val = (
        torch.log(unit_std / (std + eps))
        + ((std ** 2 + (mean - mean_prior) ** 2 - 1) / 2)
    ).sum(1)
    return val


def loss_entropy_binary(crossover_posterior):
    """Given the sensitive approximated posterior conditioned on the target latent
    representation (i.e. p(s|z_{t})), returns the entropy"""
    normalized = crossover_posterior / crossover_posterior.sum(1).view(-1, 1)
    if len(normalized.shape) == 1:
        other_prob = 1 - normalized
        return (
            other_prob * (other_prob + 1e-8).log()
            + (normalized + 1e-8).log() * normalized
        )
    else:
        return (normalized * (normalized + 1e-8).log()).sum(1)


def get_yaleb_poses():
    root = DATA_DIR / "yaleb" / "CroppedYale"
    filepaths = root.glob("**/*")
    reg = "A(.\d+)E(.\d+)"
    poses = []
    for filepath in filepaths:
        s = re.search(reg, str(filepath))
        if s:
            x, y = s.groups()
            poses.append((float(x), float(y), s.groups()))
    return set(poses)


def cluster_yaleb_poses():
    poses = get_yaleb_poses()
    cluster_names = [
        "front",
        "upper_left",
        "upper_right",
        "lower_left",
        "lower_right",
    ]
    clusters = {name: [] for name in cluster_names}
    for pose in poses:
        azimuth, elevation, orig_pose = pose
        cluster = "front"
        # horizontally centered poses and poses within a certain radius
        # of the origin should be kept in the front cluster
        if (
            abs(azimuth) > 0.001
            and math.sqrt(elevation ** 2 + azimuth ** 2) > 25
        ):
            vertical_side = "upper" if elevation > 0 else "lower"
            horizontal_side = "left" if azimuth < 0 else "right"
            cluster = f"{vertical_side}_{horizontal_side}"
        clusters[cluster].append((azimuth, elevation, orig_pose))
    return clusters


def reshape_tensor(t):
    s = t.shape
    if len(s) == 1:
        if (t.max() <= 1) and (0 >= t.min()):
            return t > 0.5
        else:
            return t
    elif len(s) == 2:
        b, d = s
        return t.argmax(1)
    raise ValueError(f"cannot reshape tensor in a smart way")


class NamespaceWithGet:
    def __init__(self, namespace):
        self.namespace = namespace

    def __getattr__(self, name):
        return getattr(self.namespace, name)

    def get(self, key, default):
        if key not in self.namespace:
            return default
        attr = getattr(self.namespace, key)
        if attr is None:
            return default
        return attr


class ArgumentParser(argparse.ArgumentParser):
    def parse_args(self):
        namespace = super().parse_args()
        return NamespaceWithGet(namespace)
