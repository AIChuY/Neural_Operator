"""Trainer for multiple GPU training."""
from typing import List

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .training_logger import TrainingLogger


class Trainer(object):
    """Trainer for multiple GPU training."""

    # TODO: refactor this class
    def __init__(  # noqa: PLR0913
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        rank: int,
        config: dict,
    ):
        """Initialize the trainer.

        Args:
        ----
            model (torch.nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler.
            train_loader (torch.utils.data.DataLoader): The data loader for training.
            valid_loader (torch.utils.data.DataLoader): The data loader for validation.
            test_loader (torch.utils.data.DataLoader): The data loader for testing.
            rank (int): The rank of the current process.
            config (dict): The config dictionary.

        """
        self.model = model.to(rank)
        self.model = DDP(self.model, device_ids=[rank])
        self.loss_function = config["train"]["loss_function"]
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.rank = rank
        self.config = config
        self.logger = TrainingLogger(config["train"]["log_file"])

    def _train_epoch(self, epoch: int):
        self.model.train()
        self.train_loader.sampler.set_epoch(epoch)  # type: ignore
        train_loss_total = 0
        train_l2_loss_operator = 0
        train_l2_loss_autoencoder_in = 0
        train_l2_loss_autoencoder_out = 0
        for x, y in self.train_loader:
            x = x.to(self.rank)  # noqa: PLW2901
            y = y.to(self.rank)  # noqa: PLW2901
            out, aec_in, aec_out = self.model(x, y)
            loss_l2_operator = self.loss_function(
                out,
                y.reshape(
                    -1,
                    self.config["dataset"]["J1_out"]
                    * self.config["dataset"]["J2_out"]
                    * self.config["dataset"]["J3_out"],
                ),
            )
            loss_l2_autoencoder_in = self.loss_function(
                aec_in,
                x.reshape(
                    -1,
                    self.config["dataset"]["J1_in"] * self.config["dataset"]["J2_in"] * self.config["dataset"]["J3_in"],
                ),
            )
            loss_l2_autoencoder_out = self.loss_function(
                aec_out,
                y.reshape(
                    -1,
                    self.config["dataset"]["J1_out"]
                    * self.config["dataset"]["J2_out"]
                    * self.config["dataset"]["J3_out"],
                ),
            )
            loss_total = (
                loss_l2_operator
                + self.config["train"]["lambda_in"] * loss_l2_autoencoder_in
                + self.config["train"]["lambda_out"] * loss_l2_autoencoder_out
            )
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()

            train_loss_total += loss_total.item()
            train_l2_loss_operator += loss_l2_operator.item()
            train_l2_loss_autoencoder_in += loss_l2_autoencoder_in.item()
            train_l2_loss_autoencoder_out += loss_l2_autoencoder_out.item()
        self.scheduler.step()

        train_loss_total /= self.config["dataset"]["ntrain"]
        train_l2_loss_operator /= self.config["dataset"]["ntrain"]
        train_l2_loss_autoencoder_in /= self.config["dataset"]["ntrain"]
        train_l2_loss_autoencoder_out /= self.config["dataset"]["ntrain"]
        self.logger.log_train(
            epoch,
            [
                train_loss_total,
                train_l2_loss_operator,
                train_l2_loss_autoencoder_in,
                train_l2_loss_autoencoder_out,
            ],
        )

    def _validate_epoch(self, epoch: int):
        self.model.eval()
        valid_loss_total = 0
        valid_l2_loss_operator = 0
        valid_l2_loss_autoencoder_in = 0
        valid_l2_loss_autoencoder_out = 0
        with torch.no_grad():
            for x, y in self.valid_loader:
                x = x.to(self.rank)  # noqa: PLW2901
                y = y.to(self.rank)  # noqa: PLW2901
                out, aec_in, aec_out = self.model(x, y)
                loss_l2_operator = self.loss_function(
                    out,
                    y.reshape(
                        -1,
                        self.config["dataset"]["J1_out"]
                        * self.config["dataset"]["J2_out"]
                        * self.config["dataset"]["J3_out"],
                    ),
                )
                loss_l2_autoencoder_in = self.loss_function(
                    aec_in,
                    x.reshape(
                        -1,
                        self.config["dataset"]["J1_in"]
                        * self.config["dataset"]["J2_in"]
                        * self.config["dataset"]["J3_in"],
                    ),
                )
                loss_l2_autoencoder_out = self.loss_function(
                    aec_out,
                    y.reshape(
                        -1,
                        self.config["dataset"]["J1_out"]
                        * self.config["dataset"]["J2_out"]
                        * self.config["dataset"]["J3_out"],
                    ),
                )
                loss_total = (
                    loss_l2_operator
                    + self.config["train"]["lambda_in"] * loss_l2_autoencoder_in
                    + self.config["train"]["lambda_out"] * loss_l2_autoencoder_out
                )

                valid_loss_total += loss_total.item()
                valid_l2_loss_operator += loss_l2_operator.item()
                valid_l2_loss_autoencoder_in += loss_l2_autoencoder_in.item()
                valid_l2_loss_autoencoder_out += loss_l2_autoencoder_out.item()

        valid_loss_total /= self.config["dataset"]["ntrain"]
        valid_l2_loss_operator /= self.config["dataset"]["ntrain"]
        valid_l2_loss_autoencoder_in /= self.config["dataset"]["ntrain"]
        valid_l2_loss_autoencoder_out /= self.config["dataset"]["ntrain"]
        self.logger.log_val(
            epoch,
            [valid_loss_total, valid_l2_loss_operator, valid_l2_loss_autoencoder_in, valid_l2_loss_autoencoder_out],
        )

    def _save_checkpoint(self, epoch: int):
        pass

    def train(self, max_epochs: int):
        """Train the model.

        Args:
        ----
            max_epochs (int): The maximum number of epochs.

        """
        for epoch in range(max_epochs):
            self._train_epoch(epoch)
            self._validate_epoch(epoch)
            self._save_checkpoint(epoch)

    def test(self):
        """Test the model."""
        self.model.eval()
        test_mse_loss_operator = 0
        test_l2_loss_operator = 0
        test_l2_loss_autoencoder_in = 0
        test_l2_loss_autoencoder_out = 0
        test_record = []
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.rank)  # noqa: PLW2901
                y = y.to(self.rank)  # noqa: PLW2901
                out, aec_in, aec_out = self.model(x, y)
                test_record.append(out)
                loss_mse_operator = (
                    self.mse_loss(
                        out,
                        y.reshape(
                            -1,
                            self.config["dataset"]["J1_out"]
                            * self.config["dataset"]["J2_out"]
                            * self.config["dataset"]["J3_out"],
                        ),
                    )
                    * x.shape[0]
                )
                loss_l2_operator = self.l2_rel_loss(
                    out,
                    y.reshape(
                        -1,
                        self.config["dataset"]["J1_out"]
                        * self.config["dataset"]["J2_out"]
                        * self.config["dataset"]["J3_out"],
                    ),
                )
                loss_l2_autoencoder_in = self.l2_rel_loss(
                    aec_in,
                    x.reshape(
                        -1,
                        self.config["dataset"]["J1_in"]
                        * self.config["dataset"]["J2_in"]
                        * self.config["dataset"]["J3_in"],
                    ),
                )
                loss_l2_autoencoder_out = self.l2_rel_loss(
                    aec_out,
                    y.reshape(
                        -1,
                        self.config["dataset"]["J1_out"]
                        * self.config["dataset"]["J2_out"]
                        * self.config["dataset"]["J3_out"],
                    ),
                )

                test_l2_loss_operator += loss_l2_operator.item()
                test_mse_loss_operator += loss_mse_operator.item()
                test_l2_loss_autoencoder_in += loss_l2_autoencoder_in.item()
                test_l2_loss_autoencoder_out += loss_l2_autoencoder_out.item()

        test_mse_loss_operator /= self.config["dataset"]["ntrain"]
        test_l2_loss_operator /= self.config["dataset"]["ntrain"]
        test_l2_loss_autoencoder_in /= self.config["dataset"]["ntrain"]
        test_l2_loss_autoencoder_out /= self.config["dataset"]["ntrain"]

        self.logger.debug(
            "test_mse_op:{:.8f}\ttest_l2_op:{:.6f}\ttest_l2_aec_in:{:.6f}\ttest_l2_aec_out:{:.6f}".format(
                test_mse_loss_operator,
                test_l2_loss_operator,
                test_l2_loss_autoencoder_in,
                test_l2_loss_autoencoder_out,
            )
        )
