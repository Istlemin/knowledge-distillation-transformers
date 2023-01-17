from glob import glob
import json
from pathlib import Path
from tokenize import Token
from typing import Optional
from datasets.dataset_dict import DatasetDict
import torch
from torch.cuda import Device
from torch.utils.data import DataLoader
from torch.optim import Adam
import time
from pathlib import Path
import argparse
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import random
import numpy as np
from load_glue import load_tokenized_glue_dataset
from utils import distributed_cleanup, distributed_setup, set_random_seed, setup_logging
from model import (
    BertForSequenceClassificationWithLoss,
    get_bert_config,
    load_pretrained_bert_base,
    load_model_from_disk,
    ModelWithLoss,
)
import logging
from tqdm.auto import tqdm
from utils import get_optimizer, get_scheduler
from args import FinetuneArgs

class Args(FinetuneArgs):
    modelpath:Path

def run_epoch(
    model,
    dataloader: DataLoader,
    device: Device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR = None,
    epoch=-1,
):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    progress_bar = tqdm(range(len(dataloader)))
    losses = []
    correct_predictions = 0
    total_predictions = 0
    for batch in dataloader:
        input_ids = batch.input_ids.to(device)
        token_type_ids = batch.token_type_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        labels = None if batch.labels is None else batch.labels.to(device)

        if optimizer is not None:
            optimizer.zero_grad()
        loss, curr_correct_predictions = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        correct_predictions += curr_correct_predictions
        total_predictions += len(batch.input_ids)

        loss = torch.mean(loss)
        loss.backward()
        losses.append(loss.detach())
        if optimizer is not None:
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        progress_bar.update(1)
        progress_description = f"Loss: {sum(losses)/len(losses):.4f}, Acc: {correct_predictions / total_predictions*100:.2f}%, device:{device}"
        if scheduler is not None:
            progress_description += f", lr:{scheduler.get_last_lr()[0]:.2e}"
        progress_bar.set_description(
            progress_description,
            refresh=True
        )
        break
    loss = torch.mean(torch.stack(losses))
    return loss, correct_predictions / total_predictions


def finetune(
    gpu_idx: int,
    model: ModelWithLoss,
    tokenized_datasets: DatasetDict,
    args : Args,
):

    set_random_seed(args.seed)
    if gpu_idx != -1:
        distributed_setup(gpu_idx, args.num_gpus, args.port)
        model = model.to(gpu_idx)
        model = DDP(model, device_ids=[gpu_idx], find_unused_parameters=True)
        device = torch.device(gpu_idx)
    else:
        device = torch.device("cpu")

    setup_logging(args.outputdir)
    logging.info(args.get_reproducibility_info())

    def get_dataloader(dataset):
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=args.num_gpus,
            rank=gpu_idx,
            shuffle=True,
        )
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.batch_size // args.num_gpus,
        )

    train_dataloader = get_dataloader(tokenized_datasets.train)
    dev_dataloader = get_dataloader(tokenized_datasets.dev)

    total_train_steps = len(tokenized_datasets.train) * args.num_epochs // args.batch_size

    optimizer = get_optimizer(model, args.lr)
    scheduler = get_scheduler(optimizer, total_train_steps, schedule=args.scheduler,warmup_proportion=0.1)

    args.outputdir.mkdir(parents=True, exist_ok=True)

    for epoch in range(0, args.num_epochs):
        logging.info(f"EPOCH {epoch}:")

        train_loss, train_accuracy = run_epoch(
            model, train_dataloader, optimizer=optimizer, scheduler=scheduler, device=device, epoch=epoch
        )
        torch.distributed.all_reduce(train_loss, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(train_accuracy, op=torch.distributed.ReduceOp.SUM)
        train_loss /= args.num_gpus
        train_accuracy /= args.num_gpus

        dev_loss, dev_accuracy = run_epoch(model, dev_dataloader, device=device)
        torch.distributed.all_reduce(dev_loss, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(dev_accuracy, op=torch.distributed.ReduceOp.SUM)
        dev_loss /= args.num_gpus
        dev_accuracy /= args.num_gpus

        logging.info(f"Train loss: {train_loss}\t\tTrain accuracy: {train_accuracy}")
        logging.info(f"Dev loss: {dev_loss}\t\tdev accuracy: {dev_accuracy}")

        if gpu_idx == 0:
            checkpoint = {
                "epochs": epoch + 1,
                "train_loss": train_loss.item(),
                "dev_loss": dev_loss.item(),
                "train_accuracy": train_accuracy.item(),
                "dev_accuracy": dev_accuracy.item(),
                "lr": args.lr,
                "batch_size": args.batch_size,
                "seed": args.seed
            }
            logging.info(json.dumps(checkpoint,indent=4))

            best_accuracy = 0
            if (args.outputdir/"best.json").exists():
                best_checkpoint = json.loads((args.outputdir/"best.json").read_text())
                best_accuracy = best_checkpoint["dev_accuracy"]
            
            if dev_accuracy>best_accuracy:
                logging.info("Saving model...")
                (args.outputdir/"best.json").write_text(json.dumps(checkpoint))
                torch.save(model,args.outputdir/"bestmodel")

    distributed_cleanup()


def main():
    args = Args().parse_args()

    set_random_seed(args.seed)

    datasets = load_tokenized_glue_dataset(args.gluepath, args.dataset,augmented=args.use_augmented_data)

    if args.modelpath is None:
        model = load_pretrained_bert_base()
    else:
        model = load_model_from_disk(args.modelpath)

    torch.multiprocessing.spawn(
        finetune,
        args=(BertForSequenceClassificationWithLoss(model), datasets, args),
        nprocs=args.num_gpus,
        join=True,
    )


if __name__ == "__main__":
    main()
