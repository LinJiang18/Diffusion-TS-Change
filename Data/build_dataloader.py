import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import joblib


class Normalizer:
    def __init__(self, scaler, auto_norm=True):
        self.scaler = scaler
        self.auto_norm = auto_norm

    def normalize(self, x):
        # x: (M, C)
        x = self.scaler.transform(x)
        if self.auto_norm:
            x = x * 2.0 - 1.0
        return x

    def unnormalize(self, x):
        # x: (M, C)
        if self.auto_norm:
            x = (x + 1.0) / 2.0
        return self.scaler.inverse_transform(x)


class SplitWindowDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_length):
        """
        data: np.ndarray of shape (M, T)
        seq_length: int
        """
        self.data = torch.from_numpy(data).float()
        self.seq_length = seq_length

        # number of full chunks
        self.num_chunks = self.data.shape[0] // seq_length

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length
        return self.data[start:end]

def split_to_chunks(data, seq_length):
    """
    data: np.ndarray, shape (M, C)
    return: np.ndarray, shape (N, seq_length, C)
    """
    M, C = data.shape
    num_chunks = M // seq_length
    trimmed = data[: num_chunks * seq_length]
    chunks = trimmed.reshape(num_chunks, seq_length, C)
    return chunks

def build_dataloader(config, args):

    # -------- config --------
    npy_path = config["train"]["params"]["data_path"]
    seq_length = config["train"]["params"]["seq_length"]
    batch_size = config["train"]["params"]["batch_size"]

    # -------- load raw data --------
    data = np.load(npy_path)
    if data.ndim == 1:
        data = data[:, None]
    # (M, C)

    # -------- fit scaler on raw data --------
    scaler = MinMaxScaler()
    scaler.fit(data)

    joblib.dump(
        scaler,
        os.path.join(args.output_dir, "scaler.pkl")
    )

    normalizer = Normalizer(scaler, auto_norm=True)

    # -------- split unnormalized data --------
    origin_unnorm = split_to_chunks(data, seq_length)   # (N, L, C)

    # -------- normalize full sequence --------
    data_norm = normalizer.normalize(data)               # (M, C)

    # -------- split normalized data --------
    origin_norm = split_to_chunks(data_norm, seq_length) # (N, L, C)

    # -------- save both --------
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "origin_data_unnorm.npy"), origin_unnorm)
    np.save(os.path.join(args.output_dir, "origin_data_norm.npy"), origin_norm)

    # -------- training dataset uses normalized data --------
    dataset = SplitWindowDataset(data_norm, seq_length)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    return dataloader


