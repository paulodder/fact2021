import math
import pandas as pd
import json
import torch
import torch.nn as nn
from dotenv import dotenv_values, find_dotenv
from pathlib import Path
import re
import argparse

DOTENV = dotenv_values(find_dotenv())
DATA_DIR = Path(DOTENV["DATA_DIR"])
RESULTS_DIR = Path(DOTENV["RESULTS_DIR"])
# constants
# ENTROPY_wo_KL = "entropy_w/o_kl"
# KL_ORTH_wo_ENTROPY = "kl_orth_w/o_entropy"
# wo_ENTROPY_KL = "w/o_entropy_kl"
# ENTROPY_KL_wo_ORTH = "entropy_kl_w/o_orth"
# ENTROPY_KL_ORTH = "entropy_kl_orth"


def current_device():
    return "cuda:0" if False else "cpu:0"  # torch.cuda.is_available()


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
    return nn.functional.binary_cross_entropy(
        x.squeeze().float(), y.squeeze().float()
    )


def fillnan(tensor, value):
    tensor[tensor != tensor] = value
    return tensor


def loss_representation(y_pred, y_true):
    y_pred = fillnan(torch.clamp(y_pred, 0, 1), 0.0)
    out = nn.functional.binary_cross_entropy(y_pred, y_true, reduction="none")
    return out


def KLD(mean, std, mean_prior, eps=1e-8):
    "assumes prior unit variance"
    val = (
        torch.log(1.0 / (std + eps))
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


def get_result_fname(args):
    """Given args object, return fname formatted accordingly"""
    if args.experiment == "ablative":
        return f"{args.experiment}.{args.dataset}.{args.loss_components}.{args.seed}.json"


def parse_results_fname(fname):
    """given pathlib.Path instance, returns experiment name, dataset name, and the
    loss components as a comma-separated list (str) of component names, and the
    random seed

    """
    return fname.name.split(".")[:-1]


def get_settings2results(experiment_name, dataset):
    "takes pathlib.Path instance with ablative study results"
    rel_files = [
        f
        for f in (RESULTS_DIR.glob("*json"))
        if f.name.startswith(f"{experiment_name}.{dataset}")
    ]

    def load_results(fpath):
        with open(fpath, "r") as f:
            out = json.loads(f.read())
        return out

    settings2results = pd.Series(
        index=pd.MultiIndex.from_tuples(
            [parse_results_fname(f)[2:] for f in rel_files],
            names=["loss_components", "seed"],
        ),
        data=[load_results(f) for f in rel_files],
    )
    return settings2results
