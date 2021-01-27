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


dataset2experiment2metrics = {
    "yaleb": {
        "Sarhan et al.": (0.92, 0.52),
        "majority": (1 / 38, 0.355lo),
    },
    "adult": {
        "majority": (0.75, 0.665),
        "Sarhan et al.": (0.86, 0.6826),
    },
    "german": {
        "majority": (0.71, 0.69),
        "Sarhan et al.": (0.77, 0.71),
    },
}


def make_figure(df, dataset):
    experiment2metrics = dataset2experiment2metrics[dataset]

    for experiment, metrics in experiment2metrics.items():
        if experiment == "majority":
            continue
        if not experiment in df.index:
            df.loc[experiment] = {
                "target_acc": metrics[0],
                "sens_acc": metrics[1],
            }

    df = df.reindex(index=df.index[::-1])

    sns.set_style("darkgrid")

    plt.figure()
    df["sens_acc"].plot.bar(color="gray")
    plt.axhline(
        y=experiment2metrics["majority"][1],
        color="black",
        linestyle="--",
    )
    if dataset == "yaleb":
        plt.ylim(0, 1)
    else:
        plt.ylim(0.6, 0.9)
    plt.xlabel("Method")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.title(f"Sensitive accuracy {dataset} ")
    plt.savefig(FIGURES_DIR / f"{dataset}_sens.png", bbox_inches="tight")
    plt.clf()
    if dataset == "yaleb":
        plt.ylim(0, 1)
    else:
        plt.ylim(0.6, 0.9)

    df["target_acc"].plot.bar(color="gray")
    plt.axhline(
        y=experiment2metrics["majority"][0],
        color="black",
        linestyle="--",
    )
    plt.title(f"Target accuracy {dataset}")
    plt.xlabel("Method")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{dataset}_target.png", bbox_inches="tight")


def load_and_plot(dataset):
    df = pd.read_pickle(RESULTS_DIR / f"{dataset}_results")
    make_figure(df, dataset)


if __name__ == "__main__":
    seeds = [1, 2, 3]
    args = parse_args()
    config = utils.Config(args)
    df = pd.DataFrame(columns=["target_acc", "sens_acc"])
    print("Training normal FODVAE")
    tacc, sacc = [], []
    for seed in seeds:
        config.seed = seed
        res = train.main(config, return_results=True)
        tacc.append(
            res["target"]["accuracy"],
        )
        sacc.append(
            res["sensitive"]["accuracy"],
        )
    df.loc["Ours"] = {"target_acc": np.mean(tacc), "sens_acc": np.mean(sacc)}
    torch.cuda.empty_cache()
    if args.dataset != "yaleb":
        if f"{args.dataset}_vae" not in os.listdir(PROJECT_DIR / "models"):
            train_vae.train(args)
        print("Training using VAE embeddings")
        tacc, sacc = [], []
        for seed in seeds:
            config.seed = seed
            torch.manual_seed(seed)
            tacc_vae, sacc_vae = evaluate_embeddings(config)
            tacc.append(tacc_vae)
            sacc.append(sacc_vae)
        df.loc["VAE"] = {
            "target_acc": np.mean(tacc),
            "sens_acc": np.mean(sacc),
        }
    torch.cuda.empty_cache()
    print("Training raw")
    tacc, sacc = [], []
    for seed in seeds:
        config.seed = seed
        tacc_raw, sacc_raw = evaluate_raw(config)
        torch.manual_seed(seed)
        tacc.append(tacc_raw)
        sacc.append(sacc_raw)
    if args.dataset != "yaleb":
        df.loc["X"] = {"target_acc": np.mean(tacc), "sens_acc": np.mean(sacc)}
    else:
        df.loc["LR"] = {"target_acc": np.mean(tacc), "sens_acc": np.mean(sacc)}
    print(df)

    make_figure(df, args.dataset)

    df.to_pickle(RESULTS_DIR / f"{args.dataset}_results")
