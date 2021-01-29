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

sys.path.insert(0, str(PROJECT_DIR / "src"))

import train
import utils


def run(config):
    tacc, sacc = [], []
    for seed in seeds:
        config.seed = seed

        torch.cuda.empty_cache()
        res = train.main(config, return_results=True)

        tacc.append(
            res["target"]["accuracy"],
        )
        sacc.append(
            res["sensitive"]["accuracy"],
        )
    return {"target_acc": tacc, "sens_acc": sacc}


if __name__ == "__main__":
    seeds = [1, 2, 3]
    df = pd.DataFrame(columns=["target_acc", "sens_acc"])

    args = train.parse_args()
    if args.dataset not in ["cifar10", "cifar100"]:
        raise ValueError(
            "Only cifar10 and cifar100 can be ran with this script"
        )

    config = utils.Config(args)

    config.dataset = "cifar10"
    df.loc["cifar10"] = run(config)

    df.to_pickle(RESULTS_DIR / f"cifar_results")
    print(df)

    config.dataset = "cifar100"
    df.loc["cifar100"] = run(config)

    df.to_pickle(RESULTS_DIR / f"cifar_results")
    print(df)
