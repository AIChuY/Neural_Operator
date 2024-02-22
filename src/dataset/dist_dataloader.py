"""Dataloader for distributed training."""
import os

import numpy as np
import torch
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler


def prepare_dataloader(config: dict, world_size: int) -> tuple:
    """Prepare the data loader for training, validation, and testing.

    Args:
    ----
        config (dict): The config dictionary.
        world_size (int): The total number of processes.

    Returns:
    -------
        train_loader (torch.utils.data.DataLoader): The data loader for training.
        valid_loader (torch.utils.data.DataLoader): The data loader for validation.
        test_loader (torch.utils.data.DataLoader): The data loader for testing.
        config (dict): The config dictionary.

    """
    x = np.load(os.path.join(config["file_dir"], "in_f.npy"))
    y = np.load(os.path.join(config["file_dir"], "out_f.npy"))
    grid = np.load(os.path.join(config["file_dir"], "grid.npy"))
    grid_in = grid.copy()
    grid_out = grid.copy()

    # data split
    x_train = x[
        : config["ntrain"],
        :: config["sub"],
        :: config["sub"],
        :: config["sub"],
    ]
    x_valid = x[
        config["ntrain"] : config["ntrain"] + config["nvalid"],
        :: config["sub"],
        :: config["sub"],
        :: config["sub"],
    ]
    x_test = x[
        config["ntrain"] + config["nvalid"] : config["ntrain"] + config["nvalid"] + config["ntest"],
        :: config["sub"],
        :: config["sub"],
        :: config["sub"],
    ]
    y_train = y[
        : config["ntrain"],
        :: config["sub"],
        :: config["sub"],
        :: config["sub"],
    ]
    y_valid = y[
        config["ntrain"] : config["ntrain"] + config["nvalid"],
        :: config["sub"],
        :: config["sub"],
        :: config["sub"],
    ]
    y_test = y[
        config["ntrain"] + config["nvalid"] : config["ntrain"] + config["nvalid"] + config["ntest"],
        :: config["sub"],
        :: config["sub"],
        :: config["sub"],
    ]
    grid_in = grid_in[:: config["sub"], :: config["sub"], :: config["sub"], :]
    grid_out = grid_out[:: config["sub"], :: config["sub"], :: config["sub"], :]
    config["grid_in"] = grid_in
    config["grid_out"] = grid_out

    J1_in, J2_in, J3_in = x_train.shape[1], x_train.shape[2], x_train.shape[3]
    J1_out, J2_out, J3_out = y_train.shape[1], y_train.shape[2], y_train.shape[3]
    config["J1_in"] = J1_in
    config["J2_in"] = J2_in
    config["J3_in"] = J3_in
    config["J1_out"] = J1_out
    config["J2_out"] = J2_out
    config["J3_out"] = J3_out

    x_train = torch.from_numpy(x_train).float()
    x_valid = torch.from_numpy(x_valid).float()
    x_test = torch.from_numpy(x_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_valid = torch.from_numpy(y_valid).float()
    y_test = torch.from_numpy(y_test).float()

    batch_size = round(config["batch_size"] / world_size)
    # batch_size = config["batch_size"]
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=False,
        sampler=DistributedSampler(torch.utils.data.TensorDataset(x_train, y_train)),
    )
    valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_valid, y_valid),
        batch_size=batch_size,
        shuffle=False,
        sampler=DistributedSampler(torch.utils.data.TensorDataset(x_valid, y_valid), shuffle=False),
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )
    return (train_loader, valid_loader, test_loader, config)
