import pickle
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import sys
import copy
from dotenv import dotenv_values
from pathlib import Path
import warnings
import wandb

warnings.filterwarnings("ignore")
PROJECT_DIR = Path(dotenv_values()["PROJECT_DIR"])
RESULTS_DIR = Path(dotenv_values()["RESULTS_DIR"])
sys.path.insert(0, str(PROJECT_DIR / "src"))

from initializers import (
    get_fodvae,
    get_evaluation_managers,
)

import utils
from dataloaders import (
    load_data,
    dataset_registrar,
)
from defaults import DATASET2DEFAULTS


MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def model_fname(config):
    rel_params = [
        config.dataset,
        config.lambda_od,
        config.lambda_entropy,
        config.gamma_od,
        config.gamma_entropy,
    ]
    process_param = lambda p: p if type(p) == str else str(round(p, 6))
    rel_params = [process_param(param) for param in rel_params]
    return "-".join(rel_params)


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        choices=list(dataset_registrar.keys()),
        help="Dataset to ues",
        required=True,
    )
    parser.add_argument(
        "--eval_on_test",
        type=bool,
        default=True,
        help="Evaluate predictors on test set",
    )
    parser.add_argument(
        "--loss_components",
        type=str,
        default="entropy,kl,orth",
        choices=[
            "none",
            "entropy",
            "kl,orth",
            "entropy,kl",
            "entropy,kl,orth",
        ],
        help="Comma-separated list of components to include in the loss, the representation losses are included by default",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["ablative"],
        help="Context in which the run takes place",
    )
    parser.add_argument(
        "--max_epochs",
        "-e",
        type=int,
        help="Max number of epochs",
    )
    parser.add_argument(
        "--z_dim",
        "-z",
        type=int,
        help="Latent dimensionality",
    )
    parser.add_argument(
        "--lambda_od",
        type=float,
        help="Lambda for OD loss",
    )
    parser.add_argument(
        "--gamma_od",
        type=float,
        help="Gamma for OD loss",
    )
    parser.add_argument(
        "--lambda_entropy",
        type=float,
        help="Lambda for OD loss",
    )
    parser.add_argument(
        "--gamma_entropy",
        type=float,
        help="Gamma for OD loss",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=30,
        help="Number of epochs for which lambda's decay exactly by the corresponding gamma",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--seed",
        "-r",
        type=int,
        default=420,
        help="Random seed",
    )
    parser.add_argument(
        "--predictor_epochs",
        type=int,
        help="Number of epochs for which the predictor should train (if applicable)",
    )
    return parser


def parse_args():
    parser = get_argparser()
    args = parser.parse_args()
    return args


# def set_defaults(args):
#     if args.max_epochs is None:
#         args.max_epochs = DATASET2DEFAULTS[args.dataset]["max_epochs"]
#     if args.z_dim is None:
#         args.z_dim = DATASET2DEFAULTS[args.dataset]["z_dim"]


def evaluate(args, fodvae, logger=None, return_results=False):
    fodvae.eval()

    @torch.no_grad()
    def get_embs(X):
        return fodvae.encode(X.to(utils.current_device()))[0].cpu()

    # Evaluation using predictors
    eval_manager_target, eval_manager_sens = get_evaluation_managers(
        args, get_embs
    )

    eval_manager_target.fit()
    eval_manager_sens.fit()

    with torch.no_grad():
        # If we have wandb logger, or we return results,
        # we want to have the report as a dict.
        output_dict = logger is not None or return_results
        _, report_target, acc_target = eval_manager_target.evaluate(
            output_dict
        )
        _, report_sens, acc_sens = eval_manager_sens.evaluate(output_dict)
        print(report_target)
        print(report_sens)
        if logger is not None:
            logger.log_metrics(
                {
                    "target_classification_report": report_target,
                    "sens_classification_report": report_sens,
                }
            )

        # print("target classification report")
        # print(report_target)
        # print("sensitive classification report")
        # print(report_sens)

        print("~ evaluation results ~~~~~~~~~~~~~")
        print("best target acc:", round(acc_target, 2))
        print("best sens acc:  ", round(acc_sens, 2))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        if return_results:
            return {
                "target": report_target,
                "sensitive": report_sens,
            }


def main(config, logger=None, return_results=False):
    if logger is None:
        pass
        # wandb.init(project="fact2021", config=vars(config))
        # logger = WandbLogger()

    torch.manual_seed(config.seed)
    # Initial model
    fodvae = get_fodvae(config)
    fodvae.set_logger(logger)

    print("FODVAE architecture:")
    print(fodvae)

    # Init dataloaders
    train_dl, val_dl = load_data(
        config.dataset, config.batch_size, num_workers=0
    )
    print(config.key2val, config.max_epochs)
    # Train model
    trainer = pl.Trainer(
        max_epochs=config.max_epochs, logger=logger, gpus=utils.get_n_gpus()
    )
    trainer.fit(fodvae, train_dl, val_dl)

    fodvae_best_version = fodvae.get_best_version()
    # Save best version
    save_path = str(MODELS_DIR / model_fname(config))
    torch.save(fodvae_best_version.state_dict(), save_path)

    return evaluate(config, fodvae_best_version, logger, return_results)


if __name__ == "__main__":
    args = parse_args()
    config = utils.Config(args)
    return_results = config.experiment == "ablative"
    results = main(config, return_results=return_results)
    print(results)
    if return_results:
        with open(RESULTS_DIR / utils.get_result_fname(config), "w") as f:
            f.write(json.dumps(results, indent=2))
            print(
                f"Written results to {(RESULTS_DIR / utils.get_result_fname(args)).relative_to(PROJECT_DIR)}"
            )
