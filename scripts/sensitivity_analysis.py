from dotenv import dotenv_values
import json
from pathlib import Path
import sys
import argparse

DOTENV = dotenv_values()
DATA_DIR = Path(DOTENV["DATA_DIR"])
PROJECT_DIR = Path(DOTENV["PROJECT_DIR"])
RESULTS_DIR = Path(DOTENV["RESULTS_DIR"])
sys.path.insert(0, str(PROJECT_DIR / "src"))

import utils
from defaults import DATASET2DEFAULTS
from train import main
import itertools as it
from dataloaders import dataset_registrar
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        choices=list(dataset_registrar.keys()),
        help="Dataset to ues",
        required=True,
    )
    return parser.parse_args()


def get_sensitivity_results_fname(config, vp2val):
    "vp2val is a dict that maps the varying parameter to its value"
    parameter_spec = "$".join(
        f"{k}={round(v, 2)}" for k, v in sorted(vp2val.items())
    )
    return f"sensitivity&{config.dataset}&{parameter_spec}&{config.seed}.json"


SEEDS = [42, 420]
NOF_STEPS = 4
DATASET_VARYING_PARAMS2VALS = pd.Series(
    {
        (("adult", ("lambda_entropy", "lambda_od"))): (
            np.linspace(0.1, 1, NOF_STEPS),
            np.linspace(0.01, 0.09, NOF_STEPS),
        ),
        (("adult", ("gamma_entropy", "gamma_od"))): (
            np.linspace(1, 2, NOF_STEPS),
            np.linspace(0.8, 1.7, NOF_STEPS),
        ),
        (("german", ("lambda_entropy", "lambda_od"))): (
            np.linspace(0.1, 1, NOF_STEPS),
            np.linspace(0.01, 0.09, NOF_STEPS),
        ),
        (("german", ("gamma_entropy", "gamma_od"))): (
            np.linspace(1, 2, NOF_STEPS),
            np.linspace(0.8, 1.7, NOF_STEPS),
        ),
    }
)


if __name__ == "__main__":
    args = parse_args()
    assert (
        args.dataset in DATASET_VARYING_PARAMS2VALS.index.levels[0]
    ), f"No sensitivity sweep parameters defined for {arg.dataset}"
    constant_params = {
        "dataset": args.dataset,
        "experiment": "sensitivity",
        "return_results": True,
    }
    # vp stands for varying parameter

    # tmp
    dataset = args.dataset
    dataset = "adult"
    ##
    for seed in SEEDS:
        for (vp0, vp1), (vpr0, vpr1) in DATASET_VARYING_PARAMS2VALS.loc[
            dataset
        ].iteritems():
            for vp0_val, vp1_val in it.product(vpr0, vpr1):
                vp2val = {vp0: vp0_val, vp1: vp1_val}
                config = utils.Config(
                    {**constant_params, **vp2val, "seed": seed}
                )
                results_fname = get_sensitivity_results_fname(config, vp2val)
                print(config)
                print(results_fname)
                results = main(config, return_results=True)
                with open(RESULTS_DIR / results_fname, "w") as f:
                    f.write(json.dumps(results))
