import re
import os
import numpy as np
import pickle
from dotenv import dotenv_values, find_dotenv
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
import pandas as pd
from PIL import Image

DOTENV = dotenv_values(find_dotenv())
DATA_DIR = Path(DOTENV["DATA_DIR"])


def categorical_series_labels_to_index(series):
    return series.astype("category").cat.codes


def df_categorical_columns_to_indices(df, columns):
    for column in columns:
        df[column] = categorical_series_labels_to_index(df[column])
    return df


class CIFAR10Dataset(datasets.CIFAR10):
    def __init__(self, train):
        root = DATA_DIR / "cifar10"
        super(CIFAR10Dataset, self).__init__(
            root,
            train=train,
            transform=transforms.ToTensor(),
            target_transform=None,
        )
        self.living_labels = {2, 3, 4, 5, 6, 7}
        self.targets = torch.Tensor(
            [int(target in self.living_labels) for target in self.targets]
        )


class CIFAR100Dataset(datasets.CIFAR100):
    def __init__(self, train):
        root = DATA_DIR / "cifar100"
        super(CIFAR100Dataset, self).__init__(
            root,
            train=train,
            transform=transforms.ToTensor(),
            target_transform=None,
        )

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                self.targets.extend(entry["coarse_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.targets = torch.Tensor(self.targets)


class AdultDataset(Dataset):
    def __init__(self, train):
        super(AdultDataset, self).__init__()
        self.train = train
        self.filepath = (
            DATA_DIR / "adult" / f"{'train' if self.train else 'test'}.csv"
        )
        df = pd.read_csv(
            self.filepath, header=None, skiprows=(0 if self.train else 1)
        )

        # the final column indicates the target var, which is one of "<=50K", ">50K"
        self.y = torch.Tensor((df[14] == ">50K").values)
        # we don't want the target column in our data
        df.drop(columns=14, inplace=True)

        # represent the categorical columns as indices, not strings
        self.cat_columns = [1, 3, 5, 6, 7, 8, 9, 13]
        df = df_categorical_columns_to_indices(df, self.cat_columns)
        self.x = torch.Tensor(df.values)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return self.x.shape[0]


class GermanDataset(Dataset):
    def __init__(self, train, train_fraction=0.8):
        super(GermanDataset, self).__init__()
        self.train = train
        self.filepath = DATA_DIR / "german" / "data.csv"
        df = pd.read_csv(self.filepath, header=None, sep=" ")

        split_index = int(df.shape[0] * train_fraction)
        if train:
            df = df.iloc[:split_index]
        else:
            df = df.iloc[split_index:]

        # the final column indicates the target var, which is either 1 or 2 (neg, pos)
        self.y = torch.Tensor((df[20] == 2).values)
        # we don't want the target column in our data
        df.drop(columns=20, inplace=True)

        # represent the categorical columns as indices, not strings
        self.cat_columns = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]
        df = df_categorical_columns_to_indices(df, self.cat_columns)
        self.x = torch.Tensor(df.values)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return self.x.shape[0]


# def get_yaleb_poses():
#     root = DATA_DIR / "yaleb" / "CroppedYale"
#     filepaths = root.glob("**/*")
#     reg = "A(.\d+)E(.\d+)"
#     poses = []
#     for filepath in filepaths:
#         s = re.search(reg, str(filepath))
#         if s:
#             poses.append(s.groups())
#     return poses
# import matplotlib.pyplot as plt
# for group in set(get_yaleb_poses()):
#     plt.scatter(int(group[0]), int(group[1]))
# plt.show()


class YalebDataset(Dataset):
    people_ids = [f"{n:02}" for n in list(range(1, 14)) + list(range(15, 40))]
    train_positions = {
        (-110, 65),
        (0, 90),
        (110, 65),
        (110, -20),
        (-110, -20),
    }

    x_to_coord = lambda self, n: f"{'+' if n >= 0 else '-'}{abs(n):03}"
    y_to_coord = lambda self, n: f"{'+' if n >= 0 else '-'}{abs(n):02}"

    def __init__(self, train):
        super(YalebDataset, self).__init__()
        self.train = train

        image_paths = []
        train_paths = []
        root = DATA_DIR / "yaleb" / "CroppedYale"
        for people_id in self.people_ids:
            dir_path = root / f"yaleB{people_id}"
            paths = dir_path.glob("*.pgm*")
            paths = [str(path) for path in paths if not "Ambient" in str(path)]
            image_paths.extend(paths)
            for x, y in self.train_positions:
                x_coord = self.x_to_coord(x)
                y_coord = self.y_to_coord(y)
                fname = f"yaleB{people_id}_P00A{x_coord}E{y_coord}.pgm"
                train_paths.append(str(dir_path / fname))

        if train:
            self.paths = train_paths
        else:
            self.paths = [
                path for path in image_paths if path not in train_paths
            ]

        self.targets = torch.from_numpy(
            categorical_series_labels_to_index(
                pd.Series(
                    [
                        re.match(".*/yaleB(\d\d)_P00.*", path).groups()[0]
                        for path in self.paths
                    ]
                )
            ).values
        )

        self.images = []
        for path in self.paths:
            if not os.path.isfile(path):
                path += ".bad"
            self.images.append(np.array(Image.open(path)))
        self.images = torch.from_numpy(np.stack(self.images, axis=0))

    def __getitem__(self, i):
        return self.images[i], self.targets[i]

    def __len__(self):
        return len(self.images)


dataset_registrar = {
    "cifar10": CIFAR10Dataset,
    "cifar100": CIFAR100Dataset,
    "adult": AdultDataset,
    "german": GermanDataset,
    "yaleb": YalebDataset,
}


def load_data(dataset, batch_size, num_workers=0):
    if dataset not in dataset_registrar:
        raise ValueError(f"{dataset} is not a valid dataset")

    dataset_class = dataset_registrar[dataset]
    train_set = dataset_class(train=True)
    valid_set = dataset_class(train=False)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, valid_loader