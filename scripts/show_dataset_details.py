from dotenv import dotenv_values
import numpy as np
from pathlib import Path
import argparse
import os
import seaborn as sns
import pandas as pd
import torch
import sys
import matplotlib.pyplot as plt
from functools import reduce
from collections import Counter
from math import gcd

PROJECT_DIR = Path(dotenv_values()["PROJECT_DIR"])
RESULTS_DIR = Path(dotenv_values()["RESULTS_DIR"])
FIGURES_DIR = Path(dotenv_values()["FIGURES_DIR"])

import train
import utils
from eval_raw import evaluate_raw
from eval_vae_embd import evaluate_embeddings
import train_vae


sys.path.insert(0, str(PROJECT_DIR / "src/vae"))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from vae import VAE, Encoder, Decoder
from initializers import (
    get_fodvae,
    get_evaluation_managers,
)
from dataloaders import (
    load_data,
    dataset_registrar,
)
from defaults import DATASET2DEFAULTS
from train import parse_args


def simplify_ratio(numbers):
    denominator = reduce(gcd, numbers)
    return [i / denominator for i in numbers]


def get_majority_accs(ds):
    """When given a dataset ds, return the majority vote accuracy for
    target and sensitive labels"""
    get_label = lambda label: (
        label.argmax(0) if len(label.size()) > 0 else label
    ).item()
    Y = [get_label(y) for x, y, s in ds]
    S = [get_label(s) for x, y, s in ds]
    get_majority = lambda labels: Counter(labels).most_common(1)[0][1] / len(
        labels
    )
    return get_majority(Y), get_majority(S)


def get_input_size(tensor):
    size = tensor.size()
    if len(size) == 1:
        return str(size[0])
    else:
        return " x ".join([str(s) for s in size])


if __name__ == "__main__":
    for dataset in ["adult", "german", "yaleb", "cifar10", "cifar100"]:
        # sample amount, train/test split, input size, mv target, mv sensitive
        train_ds = dataset_registrar[dataset](True)
        test_ds = dataset_registrar[dataset](False)

        total_sample_amount = len(train_ds) + len(test_ds)
        train_test_split = [
            str(round(n))
            for n in simplify_ratio([len(train_ds), len(test_ds)])
        ]
        train_test_ratio = int(train_test_split[0]) / int(train_test_split[1])
        x, _, _ = train_ds[0]

        print()
        print("====", dataset, "dataset information ======================")
        print("total sample amount:", total_sample_amount)
        print(
            "train/test split:",
            ":".join(train_test_split),
            f"(approx. {round(train_test_ratio, 2)}:1)",
        )
        print("input size:", get_input_size(x))

        for split in ("train", "test"):
            ds = train_ds if split == "train" else test_ds
            mv_target, mv_sensitive = get_majority_accs(ds)
            print(f"{split} majority vote target acc:", round(mv_target, 3))
            print(
                f"{split} majority vote sensitive acc:", round(mv_sensitive, 3)
            )
