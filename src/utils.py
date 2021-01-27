import copy
import math
import pandas as pd
from defaults import DATASET2DEFAULTS
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
    # breakpoint()
    z = mean + (std * eps)
    return z


def bce_loss(x, y):
    return nn.functional.binary_cross_entropy(
        x.squeeze().float(), y.squeeze().float()
    )


def fillnan(tensor, value):
    tensor[tensor != tensor] = value
    return tensor


def loss_representation(y_pred, y_true, reduction="none"):
    # y_pred = fillnan(torch.clamp(y_pred, 0, 1), 0.0)
    out = nn.functional.binary_cross_entropy(
        y_pred, y_true, reduction=reduction
    )
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
    if len(crossover_posterior.squeeze().shape) == 1:
        other_prob = 1 - crossover_posterior
        return (
            other_prob * (other_prob + 1e-8).log()
            + (crossover_posterior + 1e-8).log() * crossover_posterior
        )
    else:
        normalized = crossover_posterior / crossover_posterior.sum(1).view(
            -1, 1
        )
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


def prepare_tensor_for_evaluation(t):
    t = t.squeeze()
    s = t.shape

    if len(s) == 1:
        if (t.max() <= 1) and (t.min() >= 0):
            return t > 0.5
        else:
            return t
    elif len(s) == 2:
        b, d = s
        return t.argmax(1)
    raise ValueError(f"cannot reshape tensor in a smart way")


def accuracy(y, y_pred):
    # Prepare those tensors
    y = prepare_tensor_for_evaluation(y)
    y_pred = prepare_tensor_for_evaluation(y_pred)
    # With prepared tensors, we can just compare them directly.
    matches = y == y_pred

    acc = matches.float().mean().item()
    return acc


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


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Config:
    def __init__(self, args):
        """Given a parsed args, initializes parameters according to defaults unless
        they are explicitly specified in the given args object"""
        if hasattr(args, "__dict__"):
            args_key2val = vars(args)
            self.key2val = DATASET2DEFAULTS[args.dataset]
        else:
            args_key2val = args
            self.key2val = DATASET2DEFAULTS[args["dataset"]]
        for key, val in args_key2val.items():
            if val is not None:
                self.key2val[key] = val

    def __getattr__(self, key):
        return self.key2val.get(key, None)

    def __str__(self):
        keys, vals = zip(*sorted(self.key2val.items()))
        max_len = max(map(len, keys))
        out = ""
        for k, v in zip(keys, vals):
            # print(k, v)
            out += "\n{:<30}{:<5}".format(str(k), str(v))
        return out


# class ArgumentParser(argparse.ArgumentParser):
#     def parse_args(self):
#         namespace = super().parse_args()
#         return NamespaceWithGet(namespace)


def get_result_fname(config):
    """Given args object, return fname formatted accordingly"""
    if config.experiment == "ablative":
        return f"{config.experiment}.{config.dataset}.{config.loss_components}.{config.seed}.json"


def parse_results_fname(fname):
    """given pathlib.Path instance, returns experiment name, dataset name, and the
    loss components as a comma-separated list (str) of component names, and the
    random seed

    """
    return fname.name.split(".")[:-1]


def get_settings2results(experiment_name, dataset):
    """Takes experiment name and dataset name and returns pd.Series that maps
    loss_components-seed combinations"""
    rel_files = [
        f
        for f in (RESULTS_DIR.glob("*json"))
        if f.name.startswith(f"{experiment_name}.{dataset}.")
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


def get_n_gpus():
    n = torch.cuda.device_count()
    print(f"n. gpus available: {n}")
    if n > 1:
        n = 1
    print(f"n. gpus used: {n}")
    return n
