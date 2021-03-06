import re
import os
import numpy as np
import pickle
from dotenv import dotenv_values, find_dotenv
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
import pandas as pd
from PIL import Image

try:
    import utils
except ModuleNotFoundError:
    import src.utils as utils


DOTENV = dotenv_values()
DATA_DIR = Path(DOTENV["DATA_DIR"])


norm_df = lambda df: (df - df.mean()) / (df.std() + 1e-6)


def norm_non_cat_columns(df, cat_columns):
    """normalize all columns in a dataframe that are not in cat_columns"""
    non_cat_columns = list(set(df.columns) - set(cat_columns))
    df[non_cat_columns] = norm_df(df[non_cat_columns])
    return df


def categorical_series_labels_to_index(series):
    return series.astype("category").cat.codes


def df_categorical_columns_to_indices(df, columns):
    for column in columns:
        df[column] = categorical_series_labels_to_index(df[column])
    return df


def tensor_cols_to_flat_onehot(tensor, cols, cols_n):
    col_mask = torch.zeros(tensor.size(1), dtype=bool)
    col_mask[cols] = True
    n_rows = tensor.size(0)

    # need the iteration to have onehots of different length
    cols_onehot = [
        F.one_hot(tensor[:, col].long(), col_n).view(n_rows, -1)
        for col, col_n in zip(cols, cols_n)
    ]

    # note: this does totally change the column order
    return torch.hstack((tensor[:, ~col_mask], *cols_onehot))


class CIFAR10Dataset(datasets.CIFAR10):
    n_classes = 2
    n_labels = 10
    labels = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    def __init__(self, train):
        root = DATA_DIR / "cifar10"
        super(CIFAR10Dataset, self).__init__(
            root,
            train=train,
            transform=transforms.ToTensor(),
            target_transform=None,
        )

        self.sensitive_attrs = torch.Tensor(self.targets).long()
        # convert to one-hot encoding
        self.sensitive_attrs = F.one_hot(self.sensitive_attrs, self.n_labels)

        self.living_labels = {2, 3, 4, 5, 6, 7}
        self.targets = torch.Tensor(
            [int(target in self.living_labels) for target in self.targets]
        )

    def __getitem__(self, i):
        x, target = super().__getitem__(i)
        sensitive_attr = self.sensitive_attrs[i]
        return x, target, sensitive_attr


class CIFAR100Dataset(datasets.CIFAR100):
    n_fine_labels = 100
    n_coarse_labels = 20

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
        self.sensitive_attrs = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                self.targets.extend(entry["coarse_labels"])
                self.sensitive_attrs.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.targets = torch.Tensor(self.targets).long()
        self.sensitive_attrs = torch.Tensor(self.sensitive_attrs).long()

        # convert to one-hot encoding
        self.sensitive_attrs = F.one_hot(
            self.sensitive_attrs, self.n_fine_labels
        )
        self.targets = F.one_hot(self.targets, self.n_coarse_labels)

    def __getitem__(self, i):
        x, target = super().__getitem__(i)
        sensitive_attr = self.sensitive_attrs[i]
        return x, target, sensitive_attr


class AdultDataset(Dataset):
    def __init__(self, train):
        super(AdultDataset, self).__init__()
        self.train = train
        self.filepath = (
            DATA_DIR / "adult" / f"{'train' if self.train else 'test'}.csv"
        )
        df = pd.read_csv(
            self.filepath,
            header=None,
            skipinitialspace=True,
            skiprows=(0 if self.train else 1),
        )

        # the final column indicates the target var, which is one of "<=50K", ">50K"
        # (in test.csv, these values have a . appended so we remove it)
        self.y = torch.Tensor((df[14].str.replace(".", "") == ">50K").values)
        # we don't want the target column in our data
        df.drop(columns=14, inplace=True)

        # the sensitive attribute is gender, which has index 9 in the data,
        # and is one of "Male", "Female", which we map to 0, 1, respectively.
        # this attribute remains present explicitly in the model input, we don't
        # remove it.
        self.sensitive_attrs = torch.Tensor((df[9] == "Female").values)

        self.cat_columns = [1, 3, 5, 6, 7, 8, 9, 13]
        # n of labels in each categorical column
        self.cat_columns_n = [9, 16, 7, 15, 6, 5, 2, 42]

        # normalize all numerical columns
        df = norm_non_cat_columns(df, self.cat_columns)

        # represent the categorical columns as indices, not strings
        df = df_categorical_columns_to_indices(df, self.cat_columns)
        self.x = torch.Tensor(df.values)

        # ok, but we also want the categorical columns in one-hot encoding
        self.x = tensor_cols_to_flat_onehot(
            self.x, self.cat_columns, self.cat_columns_n
        )

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.sensitive_attrs[i]

    def __len__(self):
        return self.x.shape[0]


class GermanDataset(Dataset):
    def __init__(self, train, train_fraction=0.8):
        super(GermanDataset, self).__init__()
        self.train = train
        self.filepath = DATA_DIR / "german" / "data.csv"
        df = pd.read_csv(
            self.filepath, header=None, skipinitialspace=True, sep=" "
        )

        split_index = int(df.shape[0] * train_fraction)
        if train:
            df = df.iloc[:split_index]
        else:
            df = df.iloc[split_index:]

        # the final column indicates the target var, which is either 1 or 2 (neg, pos)
        self.y = torch.Tensor((df[20] == 2).values)
        # we don't want the target column in our data
        df.drop(columns=20, inplace=True)

        # the sensitive attribute is gender, which is embedded in the 9th column
        # which encodes marital status and sex through the following attributes:
        # A91 : male : divorced/separated
        # A92 : female : divorced/separated/married
        # A93 : male : single
        # A94 : male : married/widowed
        # A95 : female : single
        # we map A92, A95 to 1, and the others to 0.
        # this attribute remains present explicitly in the model input, we don't
        # remove it.
        self.sensitive_attrs = torch.Tensor(df[8].isin(["A92", "A95"]).values)

        self.cat_columns = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]
        self.cat_columns_n = [4, 5, 10, 5, 5, 4, 3, 4, 3, 3, 4, 2, 2]

        # normalize all numerical columns
        df = norm_non_cat_columns(df, self.cat_columns)

        # represent the categorical columns as indices, not strings
        df = df_categorical_columns_to_indices(df, self.cat_columns)

        self.x = torch.Tensor(df.values)
        self.x = tensor_cols_to_flat_onehot(
            self.x, self.cat_columns, self.cat_columns_n
        )

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.sensitive_attrs[i]

    def __len__(self):
        return self.x.shape[0]


class YalebDataset(Dataset):
    # all the ids of people in the dataset
    people_ids = [f"{n:02}" for n in list(range(1, 14)) + list(range(15, 40))]
    # the light source positions corresponding to the train samples
    train_positions = {
        (-110, 65),
        (0, 0),
        (110, 65),
        (110, -20),
        (-110, -20),
    }
    n_lighting_position_clusters = 5

    # convert a numerical coord to the format used by the filenames
    x_to_coord = lambda self, n: f"{'+' if n >= 0 else '-'}{abs(n):03}"
    y_to_coord = lambda self, n: f"{'+' if n >= 0 else '-'}{abs(n):02}"

    def get_cluster(self, path):
        if not hasattr(self, "coords2cluster"):
            self.coords2cluster = {}
            for name, groups in utils.cluster_yaleb_poses().items():
                for group in groups:
                    self.coords2cluster[group[-1]] = name
        return self.coords2cluster[
            re.match(".*/yaleB\d\d_P00A(.\d\d\d)E(.\d\d).*", path).groups()
        ]

    def __init__(self, train):
        super(YalebDataset, self).__init__()
        self.train = train

        image_paths = []
        train_paths = []
        root = DATA_DIR / "yaleb" / "CroppedYale"
        # we index all the images present in the data, and create two lists of
        # filepaths: one, of all images, and two, of the train images subset.
        # NOTE: it's critical that the the .env value PROJECT_DIR is written
        # out as the actual full absolute path (without using ~).
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

        # specify the paths that will be used
        if train:
            self.paths = train_paths
        else:
            self.paths = [
                path for path in image_paths if path not in train_paths
            ]

        # just get the target people_ids from the filepaths
        # to ensure identical order
        self.targets = torch.from_numpy(
            categorical_series_labels_to_index(
                pd.Series(
                    [
                        re.match(".*/yaleB(\d\d)_P00.*", path).groups()[0]
                        for path in self.paths
                    ]
                )
            ).values
        ).long()
        self.targets = F.one_hot(self.targets, len(self.people_ids))

        # sensitive attrs are the cluster labels of the lighting position.
        # this position is embedded in the path/filename, so we convert the
        # path to a cluster label and then turn it into categorical indices.
        self.sensitive_attrs = torch.from_numpy(
            categorical_series_labels_to_index(
                pd.Series([self.get_cluster(path) for path in self.paths])
            ).values
        ).long()

        self.sensitive_attrs = F.one_hot(
            self.sensitive_attrs, self.n_lighting_position_clusters
        )

        # load all the images
        self.images = []
        for path in self.paths:
            # if an image was corrupted, the filename is appended with .bad in
            # the dataset
            if not os.path.isfile(path):
                path += ".bad"
            self.images.append(np.array(Image.open(path)))
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, i):
        img = Image.fromarray(self.images[i])
        img = self.to_tensor(img).view(-1)
        return img, self.targets[i], self.sensitive_attrs[i]

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

    if batch_size == "all":
        batch_size_train = len(train_set)
        batch_size_valid = len(valid_set)
    else:
        batch_size_train = batch_size
        batch_size_valid = batch_size

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size_valid,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, valid_loader


def get_embeddings(dataloader, model):
    embs = []
    targets = []
    sens = []
    for data, target, s in dataloader:
        emb = model(data)
        embs.append(emb)
        targets.append(target)
        sens.append(s)
    return [torch.cat(tensor, dim=0) for tensor in (embs, targets, sens)]


class GenericDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)


@torch.no_grad()
def load_representation_dataloaders(dataset, batch_size, model, num_workers=0):
    train_loader, valid_loader = load_data(
        dataset, "all", num_workers=num_workers
    )

    train_embs, train_targets, train_sens = get_embeddings(train_loader, model)
    valid_embs, valid_targets, valid_sens = get_embeddings(valid_loader, model)

    train_set_target = GenericDataset(train_embs, train_targets)
    valid_set_target = GenericDataset(valid_embs, valid_targets)
    train_set_sens = GenericDataset(train_embs, train_sens)
    valid_set_sens = GenericDataset(valid_embs, valid_sens)

    train_loader_target = DataLoader(
        train_set_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader_target = DataLoader(
        valid_set_target,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    train_loader_sens = DataLoader(
        train_set_sens,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader_sens = DataLoader(
        valid_set_sens,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return (
        train_loader_target,
        valid_loader_target,
        train_loader_sens,
        valid_loader_sens,
    )
