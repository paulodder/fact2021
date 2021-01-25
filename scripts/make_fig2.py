from dotenv import dotenv_values
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

dataset2majority_classifier = {
    "yaleb": (1 / 38, 0.5),
    "adult": (0.75, 0.84),
    "german": (0.71, 0.69),
}


def make_figure(df, args):
    sns.set_style("darkgrid")
    df["sens_acc"].plot.bar(color="gray")
    plt.axhline(
        y=dataset2majority_classifier[args.dataset][1],
        color="black",
        linestyle="--",
    )
    plt.title(f"Sensitive accuracy {args.dataset}")
    plt.savefig(FIGURES_DIR / f"{args.dataset}_sens.png")
    plt.clf()
    df["target_acc"].plot.bar(color="gray")
    plt.axhline(
        y=dataset2majority_classifier[args.dataset][0],
        color="black",
        linestyle="--",
    )
    plt.title(f"Target accuracy {args.dataset}")
    plt.savefig(FIGURES_DIR / f"{args.dataset}_target.png")


if __name__ == "__main__":
    args = parse_args()
    config = utils.Config(args)
    df = pd.DataFrame(columns=["target_acc", "sens_acc"])
    print("Training normal FODVAE")
    res = train.main(config, return_results=True)

    # df.loc["ours"] = {
    #     "target_acc": res["target"]["accuracy"],
    #     "sens_acc": res["sensitive"]["accuracy"],
    # }
    print("Training using VAE embeddings")
    if args.dataset != "yaleb":
        if f"{args.dataset}_vae" not in os.listdir(PROJECT_DIR / "models"):
            train_vae.train(args.dataset)
        tacc_vae, sacc_vae = evaluate_embeddings(args.dataset)
        df.loc["VAE"] = {"target_acc": tacc_vae, "sens_acc": sacc_vae}
    tacc_raw, sacc_raw = evaluate_raw(args.dataset)
    print("Training raw")
    if args.dataset != "yaleb":
        df.loc["X"] = {"target_acc": tacc_raw, "sens_acc": sacc_raw}
    else:
        df.loc["LR"] = {"target_acc": tacc_raw, "sens_acc": sacc_raw}
    print(df)
    make_figure(df, args)
