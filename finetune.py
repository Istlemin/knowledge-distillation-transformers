from glob import glob
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
from dataset_loading import load_glue_sentence_classification, load_tokenized_dataset
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
        batch = {k: v.to(device) for k, v in batch.items()}
        if optimizer is not None:
            optimizer.zero_grad()
        loss, curr_correct_predictions = model(**batch)

        correct_predictions += curr_correct_predictions
        total_predictions += len(batch["input_ids"])

        loss = torch.mean(loss)
        loss.backward()
        losses.append(loss)
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

    loss = torch.mean(torch.stack(losses))
    return loss, correct_predictions / total_predictions


def finetune(
    gpu_idx: int,
    model: ModelWithLoss,
    tokenized_datasets: DatasetDict,
    args,
):

    set_random_seed(args.seed)
    if gpu_idx != -1:
        distributed_setup(gpu_idx, args.num_gpus, args.port)
        model = model.to(gpu_idx)
        model = DDP(model, device_ids=[gpu_idx], find_unused_parameters=True)
        device = torch.device(gpu_idx)
    else:
        device = torch.device("cpu")

    setup_logging(args.checkpoint_path)

    def get_dataloader(dataset):
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=args.num_gpus,
            rank=gpu_idx,
            shuffle=False,
        )
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.batch_size // args.num_gpus,
        )

    train_dataloader = get_dataloader(tokenized_datasets["train"])
    eval_dataloader = get_dataloader(tokenized_datasets["dev"])

    total_train_steps = len(tokenized_datasets["train"]) * args.num_epochs // args.batch_size

    optimizer = get_optimizer(model, args.lr)
    scheduler = get_scheduler(optimizer, total_train_steps, schedule=args.scheduler,warmup_proportion=0.1)

    start_epoch = 0

    if args.checkpoint_path is not None:
        args.checkpoint_path.mkdir(parents=True, exist_ok=True)
        if args.resume:
            checkpoints = glob(str(args.checkpoint_path / "checkpoint_*"))
            if len(checkpoints):
                latest_checkpoint = sorted(checkpoints)[-1]
                print("Resuming from checkpoint: ", latest_checkpoint)
                checkpoint = torch.load(latest_checkpoint)
                start_epoch = checkpoint["epochs"]
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    for epoch in range(start_epoch, args.num_epochs):
        logging.info(f"EPOCH {epoch}:")

        train_loss, train_accuracy = run_epoch(
            model, train_dataloader, optimizer=optimizer, scheduler=scheduler, device=device, epoch=epoch
        )
        torch.distributed.all_reduce(train_loss, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(train_accuracy, op=torch.distributed.ReduceOp.SUM)
        train_loss /= args.num_gpus
        train_accuracy /= args.num_gpus

        test_loss, test_accuracy = run_epoch(model, eval_dataloader, device=device)
        torch.distributed.all_reduce(test_loss, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(test_accuracy, op=torch.distributed.ReduceOp.SUM)
        test_loss /= args.num_gpus
        test_accuracy /= args.num_gpus

        logging.info(f"Train loss: {train_loss}\t\tTrain accuracy: {train_accuracy}")
        logging.info(f"Test loss: {test_loss}\t\tTest accuracy: {test_accuracy}")

        if gpu_idx == 0 and args.checkpoint_path is not None:
            logging.info("Saving checkpoint...")

            if type(model) == torch.nn.DataParallel:
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()

            torch.save(
                {
                    "epochs": epoch + 1,
                    "model_state_dict": model_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                },
                args.checkpoint_path / f"checkpoint_epoch{epoch:0>3}",
            )

    distributed_cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest="dataset_path", type=Path, required=True)
    parser.add_argument("--model", dest="model_path", type=Path)
    parser.add_argument("--checkpoint_path", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--num_epochs", type=int, default=6)
    parser.add_argument("--device_ids", nargs="+", type=int, default=None)
    args = parser.parse_args()

    set_random_seed(args.seed)

    datasets = load_tokenized_dataset(
        args.dataset_path, load_glue_sentence_classification
    )

    if args.model_path is None:
        model = load_pretrained_bert_base()
    else:
        model = load_model_from_disk(args.model_path)

    torch.multiprocessing.spawn(
        finetune,
        args=(BertForSequenceClassificationWithLoss(model), datasets, args),
        nprocs=args.num_gpus,
        join=True,
    )


if __name__ == "__main__":
    main()
