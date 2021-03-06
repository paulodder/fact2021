import argparse
from dotenv import dotenv_values
from pathlib import Path
import warnings
import sys
import numpy as np
import wandb
from pytorch_lightning.loggers import WandbLogger


warnings.filterwarnings("ignore")
PROJECT_DIR = Path(dotenv_values()["PROJECT_DIR"])
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

import train
from dataloaders import dataset_registrar
import utils


def parse_args():
    parser = train.get_argparser()

    parser.add_argument(
        "--sweep_id",
        "-s",
        type=str,
        default=None,
        help="sweep id (optional)",
    )

    args = parser.parse_args()
    return args


def get_config(dataset):
    uniform_dist = lambda mini, maxi, steps=15: {
        "values": [float(n) for n in np.linspace(mini, maxi, steps)]
    }

    if dataset == "yaleb":
        params = {
            "lambda_od": uniform_dist(0.0, 1.5, 5),
            "lambda_entropy": uniform_dist(0.0, 1.5, 5),
            "gamma_od": uniform_dist(0.0, 1.5, 5),
            "gamma_entropy": uniform_dist(0.0, 1.5, 5),
            "step_size": {"value": 30},
            "max_epochs": {"value": 25},
        }
    elif dataset == "cifar100":
        params = {
            "lambda_od": uniform_dist(0.0, 0.3, 5),
            "lambda_entropy": uniform_dist(0.0, 0.3, 5),
            "gamma_od": uniform_dist(0.0, 1.0, 5),
            "gamma_entropy": uniform_dist(0.0, 1.5, 5),
            "step_size": {"value": 30},
            "max_epochs": {"value": 80},
            "encoder_lr": {"value": 1e-5},
        }
    else:
        raise ValueError("No config defined for this dataset")
    return {
        "name": f"{dataset}_sweep",
        "method": "grid",
        "metric": {
            "name": "train_target_acc",
            "goal": "maximize",
        },
        "parameters": params,
    }


def combine_args(args, sweep_config):
    for key, val in sweep_config.items():
        setattr(args, key, val)
    config = utils.Config(args)
    return config


def sweep_iteration(args):
    wandb.init(project="fact2021")
    # set up W&B logger
    wandb_logger = WandbLogger()
    # wandb.config holds current hparams
    print("wandb.config", wandb.config)
    config = combine_args(args, wandb.config)
    print("config", config)
    train.main(config, logger=wandb_logger)
    wandb.finish()


def main(args):
    wandb.init(project="fact2021")

    def sweep_iteration_with_args():
        sweep_iteration(args)

    if args.sweep_id is None:
        sweep_config = get_config(args.dataset)
        sweep_id = wandb.sweep(sweep_config, project="fact2021")
        print(f"new sweep. sweep_id: {sweep_id}")
    else:
        sweep_id = args.sweep_id
        print(f"continuing sweep. sweep_id: {sweep_id}")
    wandb.agent(sweep_id, function=sweep_iteration_with_args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
