"""Script for distributed training."""
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from config_parse import parse_config

from src.dataset.dist_dataloader import prepare_dataloader
from src.model.basis_o_net import BasisONet
from src.train.dist_trainer import Trainer


def set_up_process(rank: int, world_size: int, backend: str = "nccl"):
    """Set up the distributed process group.

    Args:
    ----
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
        backend (str): The backend used for distributed training.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up the distributed process group."""
    dist.destroy_process_group()


def main(rank: int, world_size: int, config_file: str):
    """Excute distributed training.

    Args:
    ----
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
        config_file (str): The path to the config file.
    """
    config = parse_config(config_file)
    set_up_process(rank, world_size)

    train_loader, valid_loader, test_loader = prepare_dataloader(config)

    model = BasisONet(config)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.9)

    loss_functions = config.loss_functions

    trainer = Trainer(
        model,
        loss_functions,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
        test_loader,
        rank,
        config,
    )

    trainer.train(config.train_epoch)

    if rank == 0:
        trainer.test()

    cleanup()


if __name__ == "__main__":
    start_time = time.time()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)

    finish_time = time.time()
    # print time in hour:min:sec
    print(
        "time elapsed: {:.0f}:{:.0f}:{:.0f}".format(
            (finish_time - start_time) // 3600,
            ((finish_time - start_time) % 3600) // 60,
            ((finish_time - start_time) % 3600) % 60,
        )
    )
