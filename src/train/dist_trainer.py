"""Trainer for multiple GPU training."""
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .training_logger import TrainingLogger

class Trainer(object):
    # TODO: refactor this class
    def __init__(
        self,
        model,
        mse_loss,
        l2_rel_loss,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
        test_loader,
        rank,
        opt,
    ):
        self.model = model.to(rank)
        self.model = DDP(self.model, device_ids=[rank])
        self.mse_loss = mse_loss
        self.l2_rel_loss = l2_rel_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.rank = rank
        self.opt = opt
        self.logger = TrainingLogger()

    def _train_epoch(self, epoch: int):
        self.model.train()
        self.train_loader.sampler.set_epoch(epoch)
        train_loss_total = 0
        train_l2_loss_operator = 0
        train_l2_loss_autoencoder_in = 0
        train_l2_loss_autoencoder_out = 0
        for x, y in self.train_loader:
            x = x.to(self.rank)
            y = y.to(self.rank)
            out, aec_in, aec_out = self.model(x, y)
            loss_l2_operator = self.l2_rel_loss(out, y.reshape(-1, self.opt.J1_out * self.opt.J2_out * self.opt.J3_out))
            loss_l2_autoencoder_in = self.l2_rel_loss(
                aec_in, x.reshape(-1, self.opt.J1_in * self.opt.J2_in * self.opt.J3_in)
            )
            loss_l2_autoencoder_out = self.l2_rel_loss(
                aec_out,
                y.reshape(-1, self.opt.J1_out * self.opt.J2_out * self.opt.J3_out),
            )
            loss_total = (
                loss_l2_operator
                + self.opt.lambda_in * loss_l2_autoencoder_in
                + self.opt.lambda_out * loss_l2_autoencoder_out
            )
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()

            train_loss_total += loss_total.item()
            train_l2_loss_operator += loss_l2_operator.item()
            train_l2_loss_autoencoder_in += loss_l2_autoencoder_in.item()
            train_l2_loss_autoencoder_out += loss_l2_autoencoder_out.item()
        self.scheduler.step()

        train_loss_total /= self.opt.ntrain
        train_l2_loss_operator /= self.opt.ntrain
        train_l2_loss_autoencoder_in /= self.opt.ntrain
        train_l2_loss_autoencoder_out /= self.opt.ntrain
        self.logger.debug(
            "ep:{:.0f} | train_l2_total:{:.6f} | train_l2_op:{:.6f} | train_l2_aec_in:{:6f} | train_l2_aec_out:{:6f}".format(
                epoch,
                train_loss_total,
                train_l2_loss_operator,
                train_l2_loss_autoencoder_in,
                train_l2_loss_autoencoder_out,
            )
        )

    def _validate_epoch(self, epoch: int):
        self.model.eval()
        valid_loss_total = 0
        valid_l2_loss_operator = 0
        valid_l2_loss_autoencoder_in = 0
        valid_l2_loss_autoencoder_out = 0
        with torch.no_grad():
            for x, y in self.valid_loader:
                x = x.to(self.rank)
                y = y.to(self.rank)
                out, aec_in, aec_out = self.model(x, y)
                loss_l2_operator = self.l2_rel_loss(
                    out,
                    y.reshape(-1, self.opt.J1_out * self.opt.J2_out * self.opt.J3_out),
                )
                loss_l2_autoencoder_in = self.l2_rel_loss(
                    aec_in,
                    x.reshape(-1, self.opt.J1_in * self.opt.J2_in * self.opt.J3_in),
                )
                loss_l2_autoencoder_out = self.l2_rel_loss(
                    aec_out,
                    y.reshape(-1, self.opt.J1_out * self.opt.J2_out * self.opt.J3_out),
                )
                loss_total = (
                    loss_l2_operator
                    + self.opt.lambda_in * loss_l2_autoencoder_in
                    + self.opt.lambda_out * loss_l2_autoencoder_out
                )

                valid_loss_total += loss_total.item()
                valid_l2_loss_operator += loss_l2_operator.item()
                valid_l2_loss_autoencoder_in += loss_l2_autoencoder_in.item()
                valid_l2_loss_autoencoder_out += loss_l2_autoencoder_out.item()

        valid_loss_total /= self.opt.nvalid
        valid_l2_loss_operator /= self.opt.nvalid
        valid_l2_loss_autoencoder_in /= self.opt.nvalid
        valid_l2_loss_autoencoder_out /= self.opt.nvalid
        self.logger.debug(
            f"ep:{epoch} | "
            f"valid_l2_total:{valid_loss_total:.6f} | "
            f"valid_l2_op:{valid_l2_loss_operator:.6f} | "
            f"valid_l2_aec_in:{valid_l2_loss_autoencoder_in:.6f} | "
            f"valid_l2_aec_out:{valid_l2_loss_autoencoder_out:.6f}"
        )

    def _save_checkpoint(self, epoch: int):
        pass

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._train_epoch(epoch)
            self._validate_epoch(epoch)
            self._save_checkpoint(epoch)

    def test(self):
        self.model.eval()
        test_mse_loss_operator = 0
        test_l2_loss_operator = 0
        test_l2_loss_autoencoder_in = 0
        test_l2_loss_autoencoder_out = 0
        test_record = []
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.rank)
                y = y.to(self.rank)
                out, aec_in, aec_out = self.model(x, y)
                test_record.append(out)
                loss_mse_operator = (
                    self.mse_loss(
                        out,
                        y.reshape(-1, self.opt.J1_out * self.opt.J2_out * self.opt.J3_out),
                    )
                    * x.shape[0]
                )
                loss_l2_operator = self.l2_rel_loss(
                    out,
                    y.reshape(-1, self.opt.J1_out * self.opt.J2_out * self.opt.J3_out),
                )
                loss_l2_autoencoder_in = self.l2_rel_loss(
                    aec_in,
                    x.reshape(-1, self.opt.J1_in * self.opt.J2_in * self.opt.J3_in),
                )
                loss_l2_autoencoder_out = self.l2_rel_loss(
                    aec_out,
                    y.reshape(-1, self.opt.J1_out * self.opt.J2_out * self.opt.J3_out),
                )

                test_l2_loss_operator += loss_l2_operator.item()
                test_mse_loss_operator += loss_mse_operator.item()
                test_l2_loss_autoencoder_in += loss_l2_autoencoder_in.item()
                test_l2_loss_autoencoder_out += loss_l2_autoencoder_out.item()

        test_mse_loss_operator /= self.opt.ntest
        test_l2_loss_operator /= self.opt.ntest
        test_l2_loss_autoencoder_in /= self.opt.ntest
        test_l2_loss_autoencoder_out /= self.opt.ntest

        self.logger.debug(
            "test_mse_op:{:.8f}\ttest_l2_op:{:.6f}\ttest_l2_aec_in:{:.6f}\ttest_l2_aec_out:{:.6f}".format(
                test_mse_loss_operator,
                test_l2_loss_operator,
                test_l2_loss_autoencoder_in,
                test_l2_loss_autoencoder_out,
            )
        )
