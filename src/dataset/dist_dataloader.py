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
    if os.path.exists(os.path.join(config["file_dir"], "grid.npy")):
        grid = np.load(os.path.join(config["file_dir"], "grid.npy"))
        grid_in = grid.copy()
        grid_out = grid.copy()
    else:
        grid_in = np.load(os.path.join(config["file_dir"], "grid_in.npy"))
        grid_out = np.load(os.path.join(config["file_dir"], "grid_out.npy"))

    # data split and subsampling
    subsampling_x = (slice(None),) + tuple(slice(None, None, config["sub"]) for _ in range(1, x.ndim))
    subsampling_y = (slice(None),) + tuple(slice(None, None, config["sub"]) for _ in range(1, y.ndim))
    subsampling_grid_in = tuple(slice(None, None, config["sub"]) for _ in range(1, grid_in.ndim)) + (slice(None),)
    subsampling_grid_out = tuple(slice(None, None, config["sub"]) for _ in range(1, grid_out.ndim)) + (slice(None),)
    x_train = x[: config["ntrain"]][subsampling_x]
    x_valid = x[config["ntrain"] : config["ntrain"] + config["nvalid"]][subsampling_x]
    x_test = x[-config["ntest"] :][subsampling_x]
    y_train = y[: config["ntrain"]][subsampling_y]
    y_valid = y[config["ntrain"] : config["ntrain"] + config["nvalid"]][subsampling_y]
    y_test = y[-config["ntest"] :][subsampling_y]
    grid_in = grid_in[subsampling_grid_in]
    grid_out = grid_out[subsampling_grid_out]
    config["grid_in"] = grid_in
    config["grid_out"] = grid_out

    config["input_size"] = np.prod(x_train.shape[1:])
    config["output_size"] = np.prod(y_train.shape[1:])

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
