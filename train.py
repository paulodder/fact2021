import pickle
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import classification_report
import torch
import pytorch_lightning as pl
import sys
import copy
from dotenv import dotenv_values
from pathlib import Path

PROJECT_DIR = Path(dotenv_values()["PROJECT_DIR"])
sys.path.insert(0, str(PROJECT_DIR / "src"))
from fodvae import get_fodvae, get_sensitive_discriminator
from dataloaders import load_data, target2sensitive_loader, dataset_registrar
from predictors import (
    get_target_predictor,
    get_sensitive_predictor,
)

DEFAULT_Z_DIM = 2
DEFAULT_INPUT_DIM = 108
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_EPOCHS = 1


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
        "--batch_size",
        "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size",
    )
    # parser.add_argument(
    #     "--output_dim_sens", "-o", type=int, help="Latent dimensionality",
    # )
    parser.add_argument(
        "--output_dim_sens",
        "-s",
        type=int,
        default=None,
        help="Latent dimensionality",
    )
    parser.add_argument(
        "--seed", "-r", type=int, default=420, help="Random seed",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    # Init model
    # enc = MLPEncoder(input_dim=args.input_dim, z_dim=args.z_dim)
    # disc_target = MLPDiscriminator(z_dim=args.z_dim, output_dim=1)
    # disc_sens = MLPDiscriminator(z_dim=args.z_dim, output_dim=1)
    # fvae = FODVAE(enc, disc_target, disc_sens)
    fvae = get_fodvae(args)
    # Init dataloaders
    train_dl, val_dl = load_data(args.dataset, args.batch_size, num_workers=0)
    # Train model
    trainer = pl.Trainer(max_epochs=args.max_epochs)
    trainer.fit(fvae, train_dl, val_dl)

    # with open(MODELS_DIR / "adult_model", "wb") as f:
    #     pickle.dump(fvae, f)

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
        y_test = test_dl_target_emb.dataset.targets
        y_pred = target_predictor.predict(test_dl_target_emb)

        s_test = test_dl_target_emb.dataset.s
        s_pred = sensitive_predictor.predict(test_dl_target_emb)

        print("target classification report")
        print(classification_report(y_test, y_pred > 0.5))

        print("sensitive classification report")
        print(classification_report(s_test, s_pred > 0.5))
