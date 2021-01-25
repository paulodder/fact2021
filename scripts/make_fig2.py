from dotenv import dotenv_values
from pathlib import Path
import argparse


PROJECT_DIR = Path(dotenv_values()["PROJECT_DIR"])
RESULTS_DIR = Path(dotenv_values()["RESULTS_DIR"])

import torch
import train
from eval_raw import evaluate_raw
from eval_vae_embd import evaluate_embeddings
import utils
import sys

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
import utils


if __name__ == "__main__":
    args = parse_args()
    config = utils.Config(args)
    # res_normal = train.main(config, return_results=True)
    res_vae = (
        None if args.dataset == "yaleb" else evaluate_embeddings(args.dataset)
    )
    res_raw = evaluate_raw(args.dataset)
    print(res_normal, res_vae, res_raw)
