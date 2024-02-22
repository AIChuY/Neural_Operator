"""Experiment configuration for Navier Stokes problem."""
from datetime import datetime

import torch.nn.functional as F

_base_ = ["./config/base_train.py", "./config/base_model.py", "./config/navier_stokes_data.py"]

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

CONFIG = {
    "log_dir": f"./log/navier_stokes/{timestamp}/",
    "epochs": 10,
    "batch_size": 32,
    "middle_hidden": [512, 512, 512, 512],
    "nbasis_in": 100,
    "nbasis_out": 100,
    "activation": F.gelu,
}
