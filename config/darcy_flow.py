"""Experiment configuration for Darcy Flow problem."""
from datetime import datetime

import torch.nn.functional as F

_base_ = ["./config/base_train.py", "./config/base_model.py", "./config/darcy_flow_data.py"]

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

CONFIG = {
    "log_dir": f"./log/darcy_flow/{timestamp}/",
    "nbasis_in": 50,
    "nbasis_out": 30,
    "activation": F.gelu,
}
