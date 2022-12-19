import logging
import os
from pathlib import Path
from typing import Optional
import torch.distributed as dist
import random


def distributed_setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def distributed_cleanup():
    dist.destroy_process_group()


def setup_logging(path: Optional[Path] = None):
    handlers = [logging.StreamHandler()]
    if path is not None:
        path.mkdir(exist_ok=True)
        path = path / "log"
        path.unlink(missing_ok=True)
        handlers.append(logging.FileHandler(path))

    logging.basicConfig(level=logging.INFO, handlers=handlers)
    logging.info(f"Logfile: {path}")
