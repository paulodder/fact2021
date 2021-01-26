from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch

from torch import nn

import matplotlib.pyplot as plt
import sys
from pathlib import Path
from dotenv import dotenv_values

PROJECT_DIR = Path(dotenv_values()["PROJECT_DIR"])

sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts/"))

from predictors import LRPredictorTrainer
from initializers import (
    load_representation_dataloaders,
    get_target_predictor_trainer,
    get_sensitive_predictor_trainer,
    get_evaluation_managers,
)

from dataloaders import load_data, dataset_registrar
from utils import Namespace, Config
import pickle


def evaluate_raw(args):
    dataset = args.dataset
    if dataset == "yaleb":
        batch_size = 16
        z_dim = 32256

    if dataset == "adult":
        batch_size = 64
        z_dim = 108

    if dataset == "german":
        batch_size = 64
        z_dim = 61

    eval_on_test = True
    logger = None
    args.z_dim = z_dim
    args.batch_size = batch_size
    config = Config(args)

    dataset2sens_col = {"adult": 64, "german": 37}

    @torch.no_grad()
    def get_embs(X):
        if args.dataset != "yaleb":
            print((X[:, dataset2sens_col[args.dataset]]).mean())
            X[:, dataset2sens_col[args.dataset]] = 0
        return X

    # Get predictors
    eval_manager_target, eval_manager_sens = get_evaluation_managers(
        config, get_embs
    )

    eval_manager_target.fit()
    eval_manager_sens.fit()

    with torch.no_grad():
        # If we have wandb logger, or we return results,
        # we want to have the report as a dict.
        return_results = True
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

        print("~ evaluation results ~~~~~~~~~~~~~")
        print("best target acc:", round(acc_target, 2))
        print("best sens acc:  ", round(acc_sens, 2))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        return acc_target, acc_sens


if __name__ == "__main__":
    args = Namespace(dataset="german")
    evaluate_raw(args)
