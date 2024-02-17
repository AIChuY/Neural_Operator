"""Experiment configuration for Darcy Flow problem."""
import torch.nn.functional as F

_base_ = ["./config/base_train.py", "./config/base_model.py", "./config/darcy_flow_data.py"]

CONFIG = {
    "train": {
        "log_file": "darcy_flow.log",
    },
    "model": {
        "nbasis_in": 50,
        "nbasis_out": 30,
        "activation": F.gelu,
    },
}
