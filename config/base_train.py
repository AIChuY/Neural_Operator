"""Basic setting for training."""

from src.train.losses import LpLoss

CONFIG = {
    "train": {
        "learning_rate": 0.001,
        "epochs": 10,
        "lambda_in": 1,
        "lambda_out": 1,
        "device": "cuda",
        "loss_function": LpLoss(size_average=False),
    }
}
