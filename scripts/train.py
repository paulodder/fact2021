import pickle
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
    load_representation_dataloader,
    dataset_registrar,
)


DEFAULT_Z_DIM = None
DEFAULT_INPUT_DIM = 108
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_EPOCHS = None
DEFAULT_PREDICTOR_EPOCHS = 10


def get_argparser():
    parser = utils.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        choices=list(dataset_registrar.keys()),
        help="Dataset to ues",
        required=True,
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["ablative"],
        help="Context in which the run takes place",
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
        "--max_epochs",
        "-e",
        type=int,
        help="Max number of epochs",
        default=DEFAULT_MAX_EPOCHS,
    )
    parser.add_argument(
        "--z_dim",
        "-z",
        type=int,
        default=DEFAULT_Z_DIM,
        help="Latent dimensionality",
    )
    parser.add_argument(
        "--lambda_od",
        type=float,
        default=None,
        help="Lambda for OD loss",
    )
    parser.add_argument(
        "--gamma_od",
        type=float,
        default=None,
        help="Gamma for OD loss",
    )
    parser.add_argument(
        "--lambda_entropy",
        type=float,
        default=None,
        help="Lambda for OD loss",
    )
    parser.add_argument(
        "--gamma_entropy",
        type=float,
        default=None,
        help="Gamma for OD loss",
    )
    parser.add_argument(
        "--eval_on_test",
        type=bool,
        default=True,
        help="Evaluate predictors on test set",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=30,
        help="Number of epochs for which lambda's decay exactly by the corresponding gamma",
    )
    # parser.add_argument(
    #     "--learning_rate",
    #     "-l",
    #     type=float,
    #     default=DEFAULT_LEARNING_RATE,
    #     help="Learning rate",
    # )

    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
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
        default=DEFAULT_PREDICTOR_EPOCHS,
        help="Number of epochs for which the predictor should train (if applicable)",
    )

    return parser


def parse_args():
    parser = get_argparser()
    args = parser.parse_args()
    return args


def set_defaults(args):
    dataset2max_epochs = {
        "adult": 1,
        "german": 12,
    }
    dataset2z_dim = {
        "cifar10": 1,
        "cifar100": 1,
        "adult": 2,
        "german": 2,
        "yaleb": 1,
    }
    if args.max_epochs is None:
        args.max_epochs = dataset2max_epochs[args.dataset]
    if args.z_dim is None:
        args.z_dim = dataset2z_dim[args.dataset]


def get_n_gpus():
    n = torch.cuda.device_count()
    print(f"n. gpus available: {n}")
    return n


def evaluate(args, fodvae, logger=None, return_results=False):
    @torch.no_grad()
    def get_embs(X):
        return fodvae.encode(X)[0]

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
                "sensitive": report_target,
            }


def main(args, logger=None, return_results=False):
    if logger is None:
        pass
        # wandb.init(project="fact2021", config=vars(args))
        # logger = WandbLogger()

    torch.manual_seed(args.seed)
    # Initial model
    fodvae = get_fodvae(args)
    fodvae.set_logger(logger)

    print("FODVAE architecture:")
    print(fodvae)

    # Init dataloaders
    train_dl, val_dl = load_data(args.dataset, args.batch_size, num_workers=0)

    # Train model
    trainer = pl.Trainer(max_epochs=args.max_epochs, logger=logger, gpus=0)
    trainer.fit(fodvae, train_dl, val_dl)

    if return_results:
        return evaluate(args, fodvae, logger, return_results)
    else:
        evaluate(args, fodvae, logger, return_results)


if __name__ == "__main__":
    args = parse_args()
    set_defaults(args)
    # print(dict(vars(args)["namespace"].it))
    # for a in vars(args):
    #     print(a)
    # print(a, getattr(args, a))
    # print arg, getattr(args, arg)
    # print("{:<20}{:>5}".format(k, v))
    return_results = args.experiment == "ablative"
    results = main(args, return_results=return_results)
    if return_results:
        with open(RESULTS_DIR / utils.get_result_fname(args), "w") as f:
            f.write(json.dumps(results, indent=2))
            print(
                f"Written results to {(RESULTS_DIR / utils.get_result_fname(args)).relative_to(PROJECT_DIR)}"
            )
