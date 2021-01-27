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


dataset2orig_experiment2metrics = {
    "yaleb": {
        "majority": (1 / 38 + 0.03, 0.51),
        "LR": (0.79, 0.96),
        "Proposed": (0.92, 0.52),
    },
    "adult": {
        "majority": (0.75, 0.675),
        "X": (0.85, 0.84),
        "VAE": (0.82, 0.66),
        "Proposed": (0.86, 0.6826),
    },
    "german": {
        "majority": (0.715, 0.69),
        "X": (0.87, 0.80),
        "VAE": (0.725, 0.795),
        "Proposed": (0.77, 0.71),
    },
}


def make_figure(df, dataset, is_target):
    # "Ours" should be renamed "Proposed"
    df = df.rename(index={"Ours": "Proposed"})

    # We use df bar plot functionality, so prepare correct format
    df_for_plot = pd.DataFrame(index=df.index)

    metric_name = "target" if is_target else "sens"
    metric_index = 0 if is_target else 1

    # Set original results
    orig_experiment2metrics = dataset2orig_experiment2metrics[dataset]
    df_for_plot["Sarhan et al."] = pd.Series(
        {
            experiment: metrics[metric_index]
            for experiment, metrics in orig_experiment2metrics.items()
            if not experiment == "majority" and experiment in df_for_plot.index
        }
    )

    # Set our results
    df_for_plot["Ours"] = df[f"{metric_name}_acc"]

    # Sort experiments
    df_for_plot = df_for_plot.reindex(index=df_for_plot.index[::-1])

    # Get majorities
    ours_majority = get_majority_accs(dataset)
    orig_majority = orig_experiment2metrics["majority"]

    # Plotting code
    sns.set_style("darkgrid")
    orig_color = "#7678ED"
    ours_color = "#F18701"
    orig_color_dark = "#4A4DE8"
    ours_color_dark = "#CB7301"

    plt.figure()
    df_for_plot.plot.bar(
        fontsize=12,
        color={"Ours": ours_color, "Sarhan et al.": orig_color},
        rot=0,
    )
    plt.axhline(
        y=ours_majority[metric_index],
        color=ours_color_dark,
        linestyle="--",
    )
    plt.axhline(
        y=orig_majority[metric_index],
        color=orig_color_dark,
        linestyle="--",
    )

    if dataset == "yaleb":
        plt.ylim(0, 1)
    else:
        plt.ylim(0.6, 0.9)

    plt.ylabel("Accuracy")
    plt.tight_layout()

    # Save the figure
    plt.savefig(
        FIGURES_DIR / f"{dataset}_{metric_name}.png", bbox_inches="tight"
    )


def make_figures(df, dataset):
    make_figure(df, dataset, is_target=True)
    make_figure(df, dataset, is_target=False)


def load_and_plot(dataset):
    df = pd.read_pickle(RESULTS_DIR / f"{dataset}_results")
    make_figures(df, dataset)


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

    make_figures(df, args.dataset)

    df.to_pickle(RESULTS_DIR / f"{args.dataset}_results")
