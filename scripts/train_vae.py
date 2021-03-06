from pathlib import Path
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
import sys
from dotenv import dotenv_values

PROJECT_DIR = Path(dotenv_values()["PROJECT_DIR"])
sys.path.insert(0, str(PROJECT_DIR / "src/vae"))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from dataloaders import load_data, dataset_registrar
import pickle
from vae import Normal, Encoder, Decoder, VAE, latent_loss
import utils
from train import parse_args


def train(args):
    dataset = args.dataset
    if dataset == "yaleb":
        input_dim = 32256
        batch_size = 32
        z_dim = 100
        hidden = 300
        criterion = nn.BCELoss()
        max_epochs = 100

    if dataset == "adult":
        input_dim = 108
        batch_size = 64
        z_dim = 2
        hidden = 100
        criterion = nn.MSELoss()
        max_epochs = 30

    if dataset == "german":
        input_dim = 61
        batch_size = 64
        z_dim = 2
        hidden = 100
        criterion = nn.MSELoss()
        max_epochs = 30

    dataloader, dataloader_test = load_data(dataset, batch_size)

    encoder = Encoder(input_dim, hidden, hidden)
    decoder = Decoder(z_dim, hidden, input_dim)
    vae = VAE(encoder, decoder, hidden, z_dim)

    optimizer = optim.Adam(vae.parameters(), lr=0.00001)
    l = None
    ls = []
    for epoch in range(max_epochs):
        for i, data in enumerate(dataloader, 0):
            inputs, classes, _ = data
            optimizer.zero_grad()
            dec = vae(inputs)
            ll = latent_loss(vae.z_mean, vae.z_sigma)
            loss = criterion(dec, inputs) + ll
            loss.backward()
            optimizer.step()
            l = loss.data.item()
        ls.append(l)
        print(epoch, l)

    with open(PROJECT_DIR / f"models/{dataset}_vae", "wb") as f:
        torch.save(vae, f)
    plt.plot(ls)
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    config = utils.Config(args)
    train(args)
