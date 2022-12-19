import os
import torch.distributed as dist


def distributed_setup(rank, world_size, port=12345):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def distributed_cleanup():
    dist.destroy_process_group()
