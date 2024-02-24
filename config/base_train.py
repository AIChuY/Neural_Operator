"""Basic setting for training."""
from datetime import datetime

from src.train.losses import LpLoss

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

CONFIG = {
    "learning_rate": 0.001,
    "epochs": 1000,
    "lambda_in": 1,
    "lambda_out": 1,
    "device": "cuda",
    "loss_function": LpLoss(size_average=False),
    "log_dir": f"./{timestamp}/",
    "early_stop": 100,
}
