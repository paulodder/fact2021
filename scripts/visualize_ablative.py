import argparse
import seaborn as sns
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import Text
from pathlib import Path
from dotenv import dotenv_values

PROJECT_DIR = Path(dotenv_values()["PROJECT_DIR"])
RESULTS_DIR = Path(dotenv_values()["RESULTS_DIR"])
FIGURES_DIR = Path(dotenv_values()["FIGURES_DIR"])
import sys

sys.path.insert(0, str(PROJECT_DIR / "src"))
import utils
from dataloaders import dataset_registrar

sns.set_style("darkgrid")


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        choices=list(dataset_registrar.keys()),
        help="Dataset to visualize for",
        required=True,
    )
    return parser


if __name__ == "__main__":
    argparser = get_argparser()
    args = argparser.parse_args()
    settings2results = utils.get_settings2results(
        experiment_name="ablative", dataset=args.dataset
    )
    settings2acc_t = settings2results.apply(lambda j: j["target"]["accuracy"])
    settings2acc_s = settings2results.apply(
        lambda j: j["sensitive"]["accuracy"]
    )

    # compute mean and std for each loss combination
    loss_comp2acc_t_mean = settings2acc_t.groupby(level=0).mean().fillna(0)
    loss_comp2acc_t_std = settings2acc_t.groupby(level=0).std().fillna(0)

    # compute mean and std for each loss combination
    loss_comp2acc_s_mean = settings2acc_s.groupby(level=0).mean().fillna(0)
    loss_comp2acc_s_std = settings2acc_s.groupby(level=0).std().fillna(0)

    loss_comp2nice_name = pd.Series(
        {
            "entropy,kl": "Entropy+KL\nw/o Orth.",
            "entropy,kl,orth": "Entropy+KL\nOrth.",
            "kl,orth": "KL Orth.\nw/o Entropy",
            "entropy": "Entropy\nw/o KL",
            "none": "w/o Entropy\nw/o KL",
        }
    )
    sorted_loss_components = [
        "entropy",
        "kl,orth",
        "none",
        "entropy,kl",
        "entropy,kl,orth",
    ]
    means_s = loss_comp2acc_s_mean.loc[sorted_loss_components].values
    stds_s = loss_comp2acc_s_std.loc[sorted_loss_components].values
    means_t = loss_comp2acc_t_mean.loc[sorted_loss_components].values
    stds_t = loss_comp2acc_t_std.loc[sorted_loss_components].values
    # get labels
    labels = loss_comp2nice_name.loc[sorted_loss_components].values
    # print(labels)

    # start plotting
    plt.clf()

    TARGET_COLOR = "#F18701"
    SENSITIVE_COLOR = "#7678ED"
    TARGET_COLOR_DARK = "#4A4DE8"
    SENSITIVE_COLOR_DARK = "#CB7301"

    fig, ax = plt.subplots()
    bwidth = 0.25
    bar_idxs_t = np.arange(len(labels))
    bar_idxs_s = bar_idxs_t + bwidth

    # plot
    ax.bar(
        bar_idxs_t,
        means_t,
        width=bwidth,
        label="Target",
        color=TARGET_COLOR,
        yerr=stds_t,
    )
    ax.bar(
        bar_idxs_s,
        means_s,
        width=bwidth,
        label="Sensitive",
        color=SENSITIVE_COLOR,
        yerr=stds_s,
    )
    # plt.errorbar(bar_idxs_t, means_t, stds_t)

    if args.dataset in {"cifar100", "yaleb"}:
        legend_loc = "upper right"
    else:
        legend_loc = "lower right"
    plt.legend(loc=legend_loc)

    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    # y ticks
    minor_ticks = np.arange(0, 1, 0.05)
    ax.set_yticks(np.arange(0, 1, 0.05), minor=True)
    # ytick_labels = np.arange(0, 1, 0.05)
    # # ytick_labels[~((ytick_labels % 0.2) > 0)]
    # ax.set_yticklabels(ytick_labels)
    # add x tick labels
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(list(labels), fontsize=9)

    # add horizontal line
    dotted_line_x_coords = [-0.5 * bwidth, len(labels)]
    ax.plot(
        [-0.5 * bwidth, len(labels) - 0.5 * (1 + bwidth)],
        [
            loss_comp2acc_t_mean["entropy,kl,orth"],
            loss_comp2acc_t_mean["entropy,kl,orth"],
        ],
        "k--",
        color=TARGET_COLOR_DARK,
    )
    ax.plot(
        [-0.5 * bwidth, len(labels) - 0.5 * (1 + bwidth)],
        [
            loss_comp2acc_s_mean["entropy,kl,orth"],
            loss_comp2acc_s_mean["entropy,kl,orth"],
        ],
        "k--",
        color=SENSITIVE_COLOR_DARK,
    )
    # add bars
    out_fpath = FIGURES_DIR / f"ablative.{args.dataset}.png"
    fig.savefig(out_fpath)
    print(f"Written results to {out_fpath.relative_to(PROJECT_DIR)}")
