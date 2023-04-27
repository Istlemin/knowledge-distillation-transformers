import logging
import os
from pathlib import Path
from typing import Optional
import torch.distributed as dist
import random
import numpy 
import torch
from datetime import datetime
import time

from transformers.optimization import get_linear_schedule_with_warmup

def distributed_setup(rank, world_size, port):
    if port is None:
        port = int(time.time()*1000)%10000+10000
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def distributed_cleanup():
    dist.destroy_process_group()


def setup_logging(path: Optional[Path] = None, logfile_name : str = "log"):
    handlers = [logging.StreamHandler()]
    if path is not None:
        path.mkdir(exist_ok=True, parents=True)
        path = path / logfile_name
        #path.unlink(missing_ok=True)
        handlers.append(logging.FileHandler(path))

    logging.basicConfig(level=logging.INFO, handlers=handlers)
    logging.info(f"Logfile: {path}")

def set_random_seed(seed):
    random.seed(seed)                          
    numpy.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           

def get_optimizer(model, lr, weight_decay=0.01):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    return torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=lr)

def get_scheduler(optimizer,total_steps, schedule="linear_warmup", warmup_proportion=0.1):
    if schedule == "linear_warmup":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps*warmup_proportion),
            num_training_steps=total_steps,
        )
    
    if schedule == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps,
        )
    
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1)

def matthews_correlation(predictions,labels):
    predictions = predictions.bool()
    labels = labels.bool()
    n00 = torch.sum(~predictions & ~labels)
    n01 = torch.sum(~predictions & labels)
    n10 = torch.sum(predictions & ~labels)
    n11 = torch.sum(predictions & labels)
    
    n_1 = n11+n01
    n_0 = n10+n00
    n1_ = n11+n10
    n0_ = n01+n00
    return (n11*n00-n10*n01)/torch.sqrt(n_1*n_0*n1_*n0_)

def F1_score(predictions,labels):
    predictions = predictions.bool()
    labels = labels.bool()
    true_positive = torch.sum(predictions & labels)
    false_positive = torch.sum(predictions & ~labels)
    false_negative = torch.sum(~predictions & labels)

    precision = true_positive/(true_positive+false_positive)
    recall = true_positive/(true_positive+false_negative)

    return 2/(1/precision+1/recall)