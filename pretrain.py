from collections import namedtuple
from glob import glob
import logging
from pathlib import Path
from typing import Optional
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
import torch
from torch.cuda import Device
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time
from pathlib import Path
import argparse
import torch
import random
import numpy as np
from transformers import (
    AutoModelForMaskedLM,
    BertForMaskedLM,
    get_linear_schedule_with_warmup,
)

from dataset_loading import (
    load_glue_sentence_classification,
    load_tokenized_dataset,
    load_batched_dataset,
)
from model import (
    BertForMaskedLMWithLoss,
    ModelWithLoss,
    get_bert_config,
    load_pretrained_bert_base,
    load_model_from_disk,
    load_untrained_bert_base,
)

from tqdm.auto import tqdm

from typing import NamedTuple
from distributed import distributed_setup, distributed_cleanup


class MLMBatch(NamedTuple):
    tokens: torch.tensor
    masked_tokens: torch.tensor
    is_masked: torch.tensor


class MLMDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.dataset: Dataset = Dataset.load_from_disk(path)
        self.dataset.set_format("torch")
        self.tokens = self.dataset["tokens"]
        self.masked_tokens = self.dataset["masked_tokens"]
        self.is_masked = self.dataset["is_masked"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return MLMBatch(
            tokens=self.tokens[item],
            masked_tokens=self.masked_tokens[item],
            is_masked=self.is_masked[item],
        )


def run_epoch(
    model: ModelWithLoss,
    dataloader: DataLoader,
    device: Device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR = None,
):
    model.train()

    progress_bar = tqdm(range(len(dataloader)))

    losses = []
    correct_predictions = 0
    total_predictions = 0

    for i, batch in enumerate(dataloader):
        tokens = batch.tokens[:, :].to(device)
        masked_tokens = batch.masked_tokens[:, :].to(device)
        is_masked = batch.is_masked[:, :].to(device)
        batch_size = len(tokens)
        if optimizer is not None:
            optimizer.zero_grad()

        loss, batch_correct_predictions = model(
            input_ids=masked_tokens, is_masked=is_masked, output_ids=tokens
        )
        loss = loss.mean()
        batch_correct_predictions = batch_correct_predictions.sum()

        total_predictions += torch.sum(is_masked)
        correct_predictions += batch_correct_predictions

        loss.backward()
        losses.append(loss.detach().cpu())

        if optimizer is not None:
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        progress_bar.update(1)
        progress_bar.set_description(
            f"Loss: {sum(losses)/len(losses):.4f}, Acc: {correct_predictions / total_predictions*100:.2f}% device:{device}",
            refresh=True,
        )

    loss = torch.mean(torch.stack(losses))
    return loss, correct_predictions / total_predictions


def pretrain(
    gpu_idx: int,
    model: ModelWithLoss,
    args,
):
    if gpu_idx != -1:
        distributed_setup(gpu_idx, args.num_gpus)
        model = model.to(gpu_idx)
        model = DDP(model, device_ids=[gpu_idx], find_unused_parameters=True)
        device = torch.device(gpu_idx)
    else:
        device = torch.device("cpu")

    optimizer = Adam(
        model.parameters(), lr=args.lr, betas=[0.9, 0.999], weight_decay=0.01
    )

    scheduler = None
    if args.scheduler == "linear_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=10000,
            num_training_steps=100000 * 60 * args.num_epochs,
        )

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
        for dataset_part_idx in range(args.dataset_parts):
            dataset = MLMDataset(args.dataset_path / str(dataset_part_idx))

            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=args.num_gpus, rank=gpu_idx, shuffle=True
            )

            train_dataloader = DataLoader(
                dataset,
                shuffle=(train_sampler is None),
                batch_size=(args.batch_size // args.num_gpus),
                num_workers=4,
                pin_memory=True,
                sampler=train_sampler,
            )
            print(f"EPOCH {epoch}, PART {dataset_part_idx}:")

            model.train()
            train_loss, train_accuracy = run_epoch(
                model,
                train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )

            if gpu_idx == 0:
                print("Train loss:", train_loss, "\t\tTrain accuracy:", train_accuracy)
                if args.checkpoint_path is not None:
                    print("Saving checkpoint...")

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
                            "train_accuracy": train_accuracy,
                        },
                        args.checkpoint_path / f"checkpoint_epoch{epoch:0>3}",
                    )

    if gpu_idx != -1:
        distributed_cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest="dataset_path", type=Path, required=True)
    parser.add_argument("--model", dest="model_path", type=Path)
    parser.add_argument("--checkpoint_path", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--scheduler", type=str)
    parser.add_argument("--dataset_parts", type=int, default=60)
    args = parser.parse_args()

    if args.checkpoint_path is not None:
        print("Checkpoint path:", args.checkpoint_path)
        args.checkpoint_path.mkdir(exist_ok=True)
        logging.basicConfig(filename=args.checkpoint_path / "log", level=logging.DEBUG)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.model_path is None:
        model = AutoModelForMaskedLM.from_config(get_bert_config("tiny"))
    else:
        model = load_model_from_disk(args.model_path)

    # model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased", num_labels=5)

    torch.multiprocessing.spawn(
        pretrain,
        args=(BertForMaskedLMWithLoss(model), args),
        nprocs=args.num_gpus,
        join=True,
    )


if __name__ == "__main__":
    main()
