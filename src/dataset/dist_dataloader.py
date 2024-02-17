"""Dataloader for distributed training."""
import os

import numpy as np
import torch
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler


def prepare_dataloader(config: dict) -> tuple:
    """Prepare the data loader for training, validation, and testing.

    Args:
    ----
        config (dict): The config dictionary.

    Returns:
    -------
        train_loader (torch.utils.data.DataLoader): The data loader for training.
        valid_loader (torch.utils.data.DataLoader): The data loader for validation.
        test_loader (torch.utils.data.DataLoader): The data loader for testing.
        config (dict): The config dictionary.

    """
    x = np.load(os.path.join(config["dataset"]["file_dir"], "in_f.npy"))
    y = np.load(os.path.join(config["dataset"]["file_dir"], "out_f.npy"))
    grid = np.load(os.path.join(config["dataset"]["file_dir"], "grid.npy"))
    grid_in = grid.copy()
    grid_out = grid.copy()

    # data split
    x_train = x[
        : config["dataset"]["ntrain"],
        :: config["dataset"]["sub"],
        :: config["dataset"]["sub"],
        :: config["dataset"]["sub"],
    ]
    x_valid = x[
        config["dataset"]["ntrain"] : config["dataset"]["ntrain"] + config["dataset"]["nvalid"],
        :: config["dataset"]["sub"],
        :: config["dataset"]["sub"],
        :: config["dataset"]["sub"],
    ]
    x_test = x[
        config["dataset"]["ntrain"] + config["dataset"]["nvalid"] : config["dataset"]["ntrain"]
        + config["dataset"]["nvalid"]
        + config["dataset"]["ntest"],
        :: config["dataset"]["sub"],
        :: config["dataset"]["sub"],
        :: config["dataset"]["sub"],
    ]
    y_train = y[
        : config["dataset"]["ntrain"],
        :: config["dataset"]["sub"],
        :: config["dataset"]["sub"],
        :: config["dataset"]["sub"],
    ]
    y_valid = y[
        config["dataset"]["ntrain"] : config["dataset"]["ntrain"] + config["dataset"]["nvalid"],
        :: config["dataset"]["sub"],
        :: config["dataset"]["sub"],
        :: config["dataset"]["sub"],
    ]
    y_test = y[
        config["dataset"]["ntrain"] + config["dataset"]["nvalid"] : config["dataset"]["ntrain"]
        + config["dataset"]["nvalid"]
        + config["dataset"]["ntest"],
        :: config["dataset"]["sub"],
        :: config["dataset"]["sub"],
        :: config["dataset"]["sub"],
    ]
    grid_in = grid_in[:: config["dataset"]["sub"], :: config["dataset"]["sub"], :: config["dataset"]["sub"], :]
    grid_out = grid_out[:: config["dataset"]["sub"], :: config["dataset"]["sub"], :: config["dataset"]["sub"], :]
    config["model"]["grid_in"] = grid_in
    config["model"]["grid_out"] = grid_out

    J1_in, J2_in, J3_in = x_train.shape[1], x_train.shape[2], x_train.shape[3]
    J1_out, J2_out, J3_out = y_train.shape[1], y_train.shape[2], y_train.shape[3]
    config["dataset"]["J1_in"] = J1_in
    config["dataset"]["J2_in"] = J2_in
    config["dataset"]["J3_in"] = J3_in
    config["dataset"]["J1_out"] = J1_out
    config["dataset"]["J2_out"] = J2_out
    config["dataset"]["J3_out"] = J3_out

    x_train = torch.from_numpy(x_train).float()
    x_valid = torch.from_numpy(x_valid).float()
    x_test = torch.from_numpy(x_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_valid = torch.from_numpy(y_valid).float()
    y_test = torch.from_numpy(y_test).float()
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=config["dataset"]["batch_size"],
        shuffle=False,
        sampler=DistributedSampler(torch.utils.data.TensorDataset(x_train, y_train)),
    )
    valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_valid, y_valid),
        batch_size=config["dataset"]["batch_size"],
        shuffle=False,
        sampler=DistributedSampler(torch.utils.data.TensorDataset(x_valid, y_valid), shuffle=False),
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=config["dataset"]["batch_size"],
        shuffle=False,
    )
    return (train_loader, valid_loader, test_loader, config)
