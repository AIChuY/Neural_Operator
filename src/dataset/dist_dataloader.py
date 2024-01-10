"""Dataloader for distributed training."""
import os

import numpy as np
import torch
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler


def prepare_dataloader(batch_size, opt):
    """Prepare the data loader for training, validation, and testing.

    Args:
    ----
        batch_size (int): The batch size for the data loader.
        opt (object): The options object containing the dataset information.

    Returns:
    -------
        train_loader (torch.utils.data.DataLoader): The data loader for training.
        valid_loader (torch.utils.data.DataLoader): The data loader for validation.
        test_loader (torch.utils.data.DataLoader): The data loader for testing.
    """
    x = np.load(os.path.join("datasets", "in_f.npy"))
    y = np.load(os.path.join("datasets", "out_f.npy"))
    grid = np.load(os.path.join("datasets", "grid.npy"))
    grid_in = grid.copy()
    grid_out = grid.copy()

    # data split
    x_train = x[: opt.ntrain, :: opt.sub, :: opt.sub, :: opt.sub]
    x_valid = x[opt.ntrain : opt.ntrain + opt.nvalid, :: opt.sub, :: opt.sub, :: opt.sub]
    x_test = x[
        opt.ntrain + opt.nvalid : opt.ntrain + opt.nvalid + opt.ntest,
        :: opt.sub,
        :: opt.sub,
        :: opt.sub,
    ]
    y_train = y[: opt.ntrain, :: opt.sub, :: opt.sub, :: opt.sub]
    y_valid = y[opt.ntrain : opt.ntrain + opt.nvalid, :: opt.sub, :: opt.sub, :: opt.sub]
    y_test = y[
        opt.ntrain + opt.nvalid : opt.ntrain + opt.nvalid + opt.ntest,
        :: opt.sub,
        :: opt.sub,
        :: opt.sub,
    ]
    grid_in = grid_in[:: opt.sub, :: opt.sub, :: opt.sub, :]
    grid_out = grid_out[:: opt.sub, :: opt.sub, :: opt.sub, :]
    J1_in, J2_in, J3_in = x_train.shape[1], x_train.shape[2], x_train.shape[3]
    J1_out, J2_out, J3_out = y_train.shape[1], y_train.shape[2], y_train.shape[3]
    x_train = torch.from_numpy(x_train).float()
    x_valid = torch.from_numpy(x_valid).float()
    x_test = torch.from_numpy(x_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_valid = torch.from_numpy(y_valid).float()
    y_test = torch.from_numpy(y_test).float()
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
    return (
        train_loader,
        valid_loader,
        test_loader,
        (J1_in, J2_in, J3_in),
        (J1_out, J2_out, J3_out),
    )
