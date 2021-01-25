from dotenv import dotenv_values
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys
import argparse
import pandas as pd
import re


DOTENV = dotenv_values()
PROJECT_DIR = Path(DOTENV["PROJECT_DIR"])
RESULTS_DIR = Path(DOTENV["RESULTS_DIR"])
FIGURES_DIR = Path(DOTENV["FIGURES_DIR"])
sys.path.insert(0, str(PROJECT_DIR / "src"))
from dataloaders import dataset_registrar
from utils import Config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        choices=list(dataset_registrar.keys()),
        help="Dataset to visualize for",
        required=True,
    )
    return parser.parse_args()


def get_varying_param2vals2seed2results(args):
    rel_files = [
        f
        for f in (RESULTS_DIR.glob("*json"))
        if f.name.startswith(f"sensitivity&{args.dataset}")
    ]

    def get_parameter_spec(fname):
        return fname.name.split("&")[2]

    def get_varying_params(fname):
        return tuple(re.findall("[a-z_]+", get_parameter_spec(fname)))

    def get_varying_param_vals(fname):
        return tuple(re.findall("[0-9.]+", get_parameter_spec(fname)))

    def get_seed(fname):
        return fname.stem.split("&")[-1]

    def load_results(fname):
        with open(fname) as f:
            out = json.loads(f.read())
        return out

    varying_params = [get_varying_params(f) for f in rel_files]
    varying_param_vals = [get_varying_param_vals(f) for f in rel_files]
    seeds = [get_seed(f) for f in rel_files]
    results = [load_results(f) for f in rel_files]
    varying_param2vals2seed2results = pd.Series(
        index=pd.MultiIndex.from_tuples(
            zip(varying_params, varying_param_vals, seeds),
            names=["param_names", "param_vals", "seed"],
        ),
        data=results,
    )
    return varying_param2vals2seed2results


def get_mean_and_std(vals2seed2results, get_val_func):
    vals2seed2val = vals2seed2results.map(get_val_func)
    return (
        vals2seed2val.groupby(level=0).mean(),
        vals2seed2val.groupby(level=0).std(),
    )


def plot_heat_matrix(coord2val, ax):
    coord2ind0 = pd.Series(
        *zip(
            *enumerate(
                sorted(
                    set([xy[0] for xy in coord2val.index]),
                    key=float,
                    reverse=True,
                )
            )
        )
    )
    coord2ind1 = pd.Series(
        *zip(
            *enumerate(
                sorted(set([xy[1] for xy in coord2val.index]), key=float)
            )
        )
    )
    hmap = np.zeros((coord2ind0.size, coord2ind1.size))
    for coord0, ind0 in coord2ind0.iteritems():
        for coord1, ind1 in coord2ind1.iteritems():
            # breakpoint()
            hmap[ind0, ind1] = coord2val[(coord0, coord1)]
            # print(coord2val[coord2val.index == (coord0, coord1)].get(0))
            # hmap[ind0, ind1] = coord2val[
            #     coord2val.index == (coord0, coord1)
            # ].get(0)

    def get_labels(vals):
        return [
            v
            if (
                (i == 0)
                or (i == len(vals) - 1)
                or ((i % int((len(vals) / 2))) == 0)
            )
            else ""
            for i, v in enumerate(vals)
        ]

    # breakpoint()
    mappable = ax.imshow(
        hmap,
        # ax=ax,
        cmap="jet",
        extent=(
            coord2ind0.max(),
            coord2ind0.min(),
            coord2ind1.min(),
            coord2ind1.max(),
        ),
        # xticklabels=get_labels(coord2ind1.index),
        # yticklabels=get_labels(coord2ind0.index),
        # square=True,
        # interpolation="bilinear",
    )
    fig.colorbar(mappable, ax=ax, fraction=0.046)
    # ax.legend()
    # ax.set_ticks(range(len(coord2ind1.index)), get_labels(coord2ind1.index))
    ax.set_xticks(
        list(range(len(coord2ind0.index))),
    )
    ax.set_xticklabels(coord2ind0.index)
    ax.set_yticks(
        list(range(len(coord2ind1.index))),
    )
    ax.set_yticklabels(coord2ind1.index)
    # print(coord2ind0)
    # ax.set_xticks(list(range(len(coord2ind1.index.astype(float)))), minor=True)
    # ax.set_xticklabels(list(coord2ind1.index.astype(str)))
    # ax.set_yticks(coord2ind0.index.astype(float), minor=True)
    # ax.set_yticklabels(
    #     list(coord2ind0.index.astype(str)),
    # )
    return hmap
    # coord2ind1 = pd.Series(
    #     dict(
    #         (coord, ind)
    #         for ind, coord in enumerate(
    #             sorted(set([xy[1] for xy in coords2val.index]), key=float)
    #         )
    #     )
    # )

    # for (x, y), val in coords2val.iteritems():
    #     hmap[coord2ind0[x], coord2ind1[y]] = val

    # sns.heatmap(hmap, ax=ax)
    # return hmap


PARAM_NAME2PRETTY_NAME = {
    "lambda_entropy": "$\lambda_{E}$",
    "lambda_od": "$\lambda_{OD}$",
    "gamma_entropy": "$\gamma_{E}$",
    "gamma_od": "$\gamma_{OD}$",
}


dummy_coord2val = pd.Series(
    {
        (0, 0.0): 0,
        (0, 0.1): 1,
        (0, 0.2): 2,
        (1, 0.0): 3,
        (1, 0.1): 4,
        (1, 0.2): 5,
        (2, 0.0): 6,
        (2, 0.1): 7,
        (2, 0.2): 8,
    }
)
fig, ax = plt.subplots()
plot_heat_matrix(dummy_coord2val, ax)
fig.savefig("/tmp/x.png")
if __name__ == "__main__":
    args = parse_args()
    varying_param2vals2seed2results = get_varying_param2vals2seed2results(
        Config({"dataset": args.dataset})
    )
    plt.clf()
    fig, ax = plt.subplots(1, 4, figsize=(16, 4), tight_layout=True)
    varying_params_list = sorted(
        varying_param2vals2seed2results.index.levels[0], reverse=True
    )
    for i, varying_params in enumerate(varying_params_list):
        vals2seed2results = varying_param2vals2seed2results.loc[varying_params]
        vals2acc_mean_t, _ = get_mean_and_std(
            vals2seed2results, lambda d: d["target"]["accuracy"]
        )
        vals2acc_mean_s, vals2std_s = get_mean_and_std(
            vals2seed2results, lambda d: d["sensitive"]["accuracy"]
        )
        ax_t, ax_s = ax[(i * 2)], ax[(i * 2) + 1]
        plot_heat_matrix(vals2acc_mean_t, ax_t)
        # breakpoint()
        ax_t.set_title("Target accuracy")
        plot_heat_matrix(vals2acc_mean_s, ax_s)
        ax_s.set_title("Sensitive accuracy")
        for this_ax in [ax_s, ax_t]:
            this_ax.set_xlabel(PARAM_NAME2PRETTY_NAME[varying_params[0]])
            this_ax.set_ylabel(PARAM_NAME2PRETTY_NAME[varying_params[1]])
    out_fpath = FIGURES_DIR / f"sensitivity.{args.dataset}.png"
    fig.savefig(out_fpath)
    print(f"Saved fig to {out_fpath.relative_to(PROJECT_DIR)}")
