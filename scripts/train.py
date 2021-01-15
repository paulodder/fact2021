import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import torch
import pytorch_lightning as pl
import sys
import copy
from dotenv import dotenv_values
from pathlib import Path
import warnings
import wandb

warnings.filterwarnings("ignore")
PROJECT_DIR = Path(dotenv_values()["PROJECT_DIR"])
sys.path.insert(0, str(PROJECT_DIR / "src"))
from initializers import (
    get_fodvae,
    get_sensitive_discriminator,
    get_target_predictor,
    get_sensitive_predictor,
)

import utils
from dataloaders import load_data, target2sensitive_loader, dataset_registrar

# from predictors import

DEFAULT_Z_DIM = 2
DEFAULT_INPUT_DIM = 108
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_EPOCHS = 1
# DEFAULT_LEARNING_RATE = 10e-4


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
        "--lambda_od", type=float, default=None, help="Lambda for OD loss",
    )
    parser.add_argument(
        "--gamma_od", type=float, default=None, help="Gamma for OD loss",
    )
    parser.add_argument(
        "--lambda_entropy",
        type=float,
        default=None,
        help="Lambda for OD loss",
    )
    parser.add_argument(
        "--gamma_entropy", type=float, default=None, help="Gamma for OD loss",
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
        default=1000,
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
        "--seed", "-r", type=int, default=420, help="Random seed",
    )
    return parser


def parse_args():
    parser = get_argparser()
    args = parser.parse_args()
    return args


def get_classification_report(test, pred, output_dict):
    return classification_report(
        utils.reshape_tensor(test),
        utils.reshape_tensor(pred),
        output_dict=output_dict,
    )


def get_n_gpus():
    return torch.cuda.device_count()


def main(args, logger=None, return_accuracy=False):
    torch.manual_seed(args.seed)
    # Initial model
    fvae = get_fodvae(args)
    fvae.set_logger(logger)
    # Init dataloaders
    train_dl, val_dl = load_data(args.dataset, args.batch_size, num_workers=0)
    # Train model
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, logger=logger, gpus=get_n_gpus()
    )
    trainer.fit(fvae, train_dl, val_dl)
    # Get embeddings for train and test
    @torch.no_grad()
    def get_embs(X):
        return fvae.forward(X)[0]

    train_dl_target_emb, test_dl_target_emb = target2sensitive_loader(
        args.dataset, args.batch_size, get_embs
    )
    # Get predictors
    target_predictor = get_target_predictor(args)
    sensitive_predictor = get_sensitive_predictor(
        get_sensitive_discriminator(args), args
    )

    ## Train target predictor
    target_predictor.fit(train_dl_target_emb)

    ## Train sensitive predictor
    sensitive_predictor.fit(train_dl_target_emb)

    with torch.no_grad():
        if (
            args.eval_on_test
        ):  # test on train DL, should be false except for debugging
            y_test = test_dl_target_emb.dataset.targets
            y_pred = target_predictor.predict(test_dl_target_emb)
        else:
            y_test = train_dl_target_emb.dataset.targets
            y_pred = target_predictor.predict(train_dl_target_emb)

        s_test = test_dl_target_emb.dataset.s
        s_pred = sensitive_predictor.predict(test_dl_target_emb)

        if logger is not None:
            logger.log_metrics(
                {
                    "target_classification_report": get_classification_report(
                        y_test, y_pred, True
                    ),
                    "sens_classification_report": get_classification_report(
                        s_test, s_pred, True
                    ),
                }
            )
        target_classification_report = get_classification_report(
            y_test, y_pred, False
        )
        sens_classification_report = get_classification_report(
            s_test, s_pred, False
        )
        print("target classification report")
        print(target_classification_report)
        print("sensitive classification report")
        print(sens_classification_report)


if __name__ == "__main__":
    args = parse_args()
    main(args)
