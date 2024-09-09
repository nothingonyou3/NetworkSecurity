from typing import TypeVar
import pandas as pd
from torch.utils.data import random_split
import torch
import socket
import struct
import datetime
import time
import os

from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

pd.set_option('future.no_silent_downcasting', True)


#  - UTILITIES -  #
def split_dataset(dataset, tr_to_tst_ratio=0.8):
    """
    Split a dataset into a train dataset and a test dataset.
    It returns a tuple in the form (train_dataset, test_dataset)
    """
    train_size = int(tr_to_tst_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset


#  - FORMAT CONVERTERS -  #

def ip2int(addr):
    """
    Convert an ip to his integer version
    :param addr:
    :return:
    """
    return struct.unpack("!I", socket.inet_aton(addr))[0]


def int2ip(addr):
    return socket.inet_ntoa(struct.pack("!I", addr))


def replace_negatives(num):
    return max(num, 0)


def datastr2epoch(datastr):
    """
    Convert a string with format %d/%m/%y %H:%M to unix epoch time
    :param datastr:
    :return:
    """
    try:
        dtime_o = datetime.datetime.strptime(datastr, "%d/%m/%Y %H:%M")
    except ValueError:
        dtime_o = datetime.datetime.strptime(datastr, "%d/%m/%Y %H:%M:%S")

    return time.mktime(dtime_o.timetuple())


#  - DATASETS -  #

T = TypeVar('T', bound="KDDDataset")

class KDDDataset(Dataset):
    """
    Dataset class for the KDD dataset
    """
    _categorical_features = ["Protocol Type", "Service", "Flag", "Class"]

    def __init__(self, path: str, cols_to_drop=("Difficulty Level"), overwrite_cache=False, downcast_binary=True):
        if not overwrite_cache and _data_is_cached(path, ".parquet"):
            self.dataframe = _read_cached_data_parquet(path)
        else:
            self.dataframe = self.__class__._import_raw(path, cols_to_drop, downcast_binary)
            _cache_tuned_dataset_parquet(self.dataframe, path)
        self._downcast_binary = downcast_binary

    def __len__(self):
        return self._shape()[0]

    def __getitem__(self, idx):
        # Get dataframe row
        row = self.dataframe.iloc[idx]

        # Separate features and labels
        if self._downcast_binary:
            features = row[:-1].values.astype(float)
            label = row['Class']
        else:
            features = row[:-self.n_labels].values.astype(float)
            label = self.dataframe.filter(like='Class', axis=1).iloc[idx]

        features = torch.tensor(features, dtype=torch.float32)
        features = features.reshape(len(features), 1)

        label = torch.tensor(label, dtype=torch.float32)
        return features, label

    def _shape(self):
        return self.dataframe.shape

    @property
    def n_features(self):
        return self._shape()[1] - self.n_labels  # we subtract labels because the last columns are the labels

    @property
    def n_labels(self):
        if self._downcast_binary:
            return 1
        return self.dataframe.filter(like='Class', axis=1).shape[1]

    def like(self: T, other: T):
        for column in other.dataframe.columns:
            if column not in self.dataframe.columns:
                self.dataframe[column] = [0 for _ in range(len(self))]

    @staticmethod
    def _import_raw(path: str, to_drop, downcast_binary):
        df = pd.read_csv(path, sep=r',', skipinitialspace=True)
        df.columns = [
            "Duration",
            "Protocol Type",
            "Service",
            "Flag",
            "Src Bytes",
            "Dst Bytes",
            "Land",
            "Wrong Fragment",
            "Urgent",
            "Hot",
            "Num Failed Logins",
            "Logged In",
            "Num Compromised",
            "Root Shell",
            "Su Attempted",
            "Num Root",
            "Num File Creations",
            "Num Shells",
            "Num Access Files",
            "Num Outbound Cmds",
            "Is Hot Logins",
            "Is Guest Login",
            "Count",
            "Srv Count",
            "Serror Rate",
            "Srv Serror Rate",
            "Rerror Rate",
            "Srv Rerror Rate",
            "Same Srv Rate",
            "Diff Srv Rate",
            "Srv Diff Host Rate",
            "Dst Host Count",
            "Dst Host Srv Count",
            "Dst Host Same Srv Rate",
            "Dst Host Diff Srv Rate",
            "Dst Host Same Src Port Rate",
            "Dst Host Srv Diff Host Rate",
            "Dst Host Serror Rate",
            "Dst Host Srv Serror Rate",
            "Dst Host Rerror Rate",
            "Dst Host Srv Rerror Rate",
            "Class",
            "Difficulty Level"
        ]
        df = df.dropna()  # drop null records
        if len(to_drop) > 0:
            df = df.drop(to_drop, axis=1)

        for categorical_feature in KDDDataset._categorical_features:
            if categorical_feature != "Class" or not downcast_binary:
                df = pd.get_dummies(df, columns=[categorical_feature], prefix=categorical_feature)

        if downcast_binary:
            df['Class'] = df['Class'].map({'normal': 1}).fillna(0).astype(int)

        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

        return df


#  - UTILITIES -  #

def __convert_path_ext(path: str, extension: str):
    """
    Given a path, for instance ../myfolder/myfile.oldextension it returns '../myfolder/myfile.newextension
    :param path:
    :param extension:
    :return:
    """
    if not extension.startswith("."):
        extension = "." + extension
    path_noex = os.path.splitext(path)[0]
    return path_noex + extension


def _data_is_cached(path: str, extension: str = "h5"):
    """
    It checks if an elaborated (not raw) version of this dataset already exists. It does that checking if a file with
    the same name but a different extension is present, since cached file are saved with the same name but a different
    extension (for instance, CSV vs HDF5)
    :param extension:
    :param path:
    :return:
    """
    return os.path.exists(__convert_path_ext(path, extension))


def _cache_tuned_dataset_parquet(df: pd.DataFrame, path: str):
    df.to_parquet(__convert_path_ext(path, ".parquet"))


def _read_cached_data_parquet(path: str):
    return pd.read_parquet(__convert_path_ext(path, ".parquet"))
