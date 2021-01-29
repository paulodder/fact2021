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
from collections import Counter


def get_majority_accs(dataset):
    test_ds = dataset_registrar[dataset](True)
    get_label = lambda label: (
        label.argmax(0) if len(label.size()) > 0 else label
    ).item()
    Y = [get_label(y) for x, y, s in test_ds]
    S = [get_label(s) for x, y, s in test_ds]
    get_majority = lambda labels: Counter(labels).most_common(1)[0][1] / len(
        labels
    )
    return get_majority(Y), get_majority(S)


if __name__ == "__main__":
    for dataset in ["adult", "german", "yaleb", "cifar10", "cifar100"]:
        test_ds = dataset_registrar[dataset](True)
