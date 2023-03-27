from glob import glob
import logging
from pathlib import Path
from socket import IP_DEFAULT_MULTICAST_TTL
from typing import Optional
from datasets.arrow_dataset import Dataset
import torch
from torch.cuda import Device
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import argparse
import torch
from transformers import (
    AutoModelForPreTraining
)

from model import (
    BertForPreTrainingWithLoss,
    ModelWithLoss,
    get_bert_config,
    load_model_from_disk,
)
from utils import get_optimizer, get_scheduler

from tqdm.auto import tqdm
import numpy as np
from typing import NamedTuple
from utils import distributed_setup, distributed_cleanup, set_random_seed, setup_logging


class MLMBatch(NamedTuple):
    tokens: torch.tensor
    masked_tokens: torch.tensor
    is_masked: torch.tensor
    segment_ids: torch.tensor
    is_random_next: torch.tensor


class MLMDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.dataset: Dataset = Dataset.load_from_disk(path)
        self.dataset.set_format("torch")
        # self.tokens = self.dataset["tokens"]
        # self.masked_tokens = self.dataset["masked_tokens"]
        # self.is_masked = self.dataset["is_masked"]

        self.tokens = self.dataset["token_ids"]
        self.masked_tokens = self.dataset["masked_token_ids"]
        self.is_masked = self.dataset["masked_positions"]
        self.segment_ids = self.dataset["segment_ids"]
        self.is_random_next = self.dataset["is_random_next"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return MLMBatch(
            tokens=self.tokens[item],
            masked_tokens=self.masked_tokens[item],
            is_masked=self.is_masked[item],
            segment_ids=self.segment_ids[item],
            is_random_next=self.is_random_next[item],
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
    correct_word_predictions = []
    correct_next_predictions = []

    for i, batch in enumerate(dataloader):
        tokens = batch.tokens[:, :].to(device)
        masked_tokens = batch.masked_tokens[:, :].to(device)
        is_masked = batch.is_masked[:, :].to(device)
        segment_ids = batch.segment_ids[:, :].to(device)
        is_random_next = batch.is_random_next[:].to(device)
        if i==0:
            print(tokens[0].tolist())
        batch_size = len(tokens)
        if optimizer is not None:
            optimizer.zero_grad()

        batch_losses, batch_word_correct_predictions, batch_next_correct_predictions = model(
            input_ids=masked_tokens, is_masked=is_masked, segment_ids=segment_ids, output_ids=tokens, is_next_sentence=~is_random_next
        )
        loss = batch_losses.mean()
        for j in range(len(is_masked)):
            correct_word_predictions += batch_word_correct_predictions[j][is_masked[j]].tolist()
        correct_next_predictions += batch_next_correct_predictions.tolist()
        
        loss.backward()
        losses.append(batch_losses.detach().cpu().tolist())

        if optimizer is not None:
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        desc = f"Loss: " + "\t".join([f"{x:.4f}" for x  in torch.mean(torch.tensor(losses,device=device),dim=0).tolist()]) + "\t"
        desc += f"Word Acc: {np.mean(correct_word_predictions)*100:.2f}%,\t" + \
            f"Next Acc: {np.mean(correct_next_predictions)*100:.2f}%,\t" + \
            f"device:{device},\tlr:{scheduler.get_last_lr()[0]:.2e}"
        progress_bar.set_description(desc)
        progress_bar.update(1)

        if i%200==0:
            torch.cuda.empty_cache()

    loss = torch.mean(torch.tensor(losses,device=device),dim=0)
    return loss, np.mean(correct_word_predictions), np.mean(correct_next_predictions)


def pretrain(
    gpu_idx: int,
    model: ModelWithLoss,
    args,
):
    dataset_size = sum(
        len(Dataset.load_from_disk(args.dataset_path / f"part_{i}"))
        for i in range(args.dataset_parts)
    )

    set_random_seed(args.seed)
    if gpu_idx != -1:
        model = model.to(gpu_idx)
        device = torch.device(gpu_idx)

        distributed_setup(gpu_idx, args.num_gpus, args.port)
        model = DDP(model, device_ids=[gpu_idx], find_unused_parameters=True)
        
        #model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus)))
    else:
        device = torch.device("cpu")

    setup_logging(args.checkpoint_path)

    total_train_steps = dataset_size * args.num_epochs // args.batch_size

    optimizer = get_optimizer(model, args.lr)
    scheduler = get_scheduler(optimizer, total_train_steps, schedule=args.scheduler)

    start_epoch = 0
    start_part = 0
    if args.checkpoint_path is not None:
        args.checkpoint_path.mkdir(parents=True, exist_ok=True)
        if args.resume:
            checkpoints = glob(str(args.checkpoint_path / "checkpoint_*"))
            if len(checkpoints):
                latest_checkpoint = sorted(checkpoints)[-1]
                logging.info(f"Resuming from checkpoint: {latest_checkpoint}")
                checkpoint = torch.load(latest_checkpoint)
                start_epoch = checkpoint["epochs"]-1
                if "parts" in checkpoint:
                    start_part = checkpoint["parts"]
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    for epoch in range(start_epoch, args.num_epochs):
        for dataset_part_idx in range(start_part, args.dataset_parts):
            dataset = MLMDataset(args.dataset_path / f"part_{dataset_part_idx}")

            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=args.num_gpus, rank=gpu_idx, shuffle=True
            )
            #train_sampler = None

            train_dataloader = DataLoader(
                dataset,
                shuffle=(train_sampler is None),
                batch_size=(args.batch_size // args.num_gpus),
                num_workers=4,
                pin_memory=True,
                sampler=train_sampler,
            )
            logging.info(f"EPOCH {epoch}, PART {dataset_part_idx}:")

            model.train()
            train_loss, train_accuracy, _ = run_epoch(
                model,
                train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )

            if gpu_idx == 0:
                logging.info(
                    f"Train losses: {train_loss}"
                )
                train_loss = train_loss.sum()
                logging.info(
                    f"Train loss: {train_loss}\t\tTrain accuracy: {train_accuracy}"
                )
                if args.checkpoint_path is not None:
                    logging.info("Saving checkpoint...")

                    if type(model) == torch.nn.DataParallel:
                        model_state_dict = model.module.state_dict()
                    else:
                        model_state_dict = model.state_dict()

                    torch.save(
                        {
                            "epochs": epoch + 1,
                            "parts": dataset_part_idx + 1,
                            "model_state_dict": model_state_dict,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "train_loss": train_loss,
                            "train_accuracy": train_accuracy,
                        },
                        args.checkpoint_path / f"checkpoint_epoch{epoch:0>3}",
                    )
        start_part = 0

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
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--scheduler", type=str)
    parser.add_argument("--dataset_parts", type=int, default=59)
    args = parser.parse_args()

    set_random_seed(args.seed)
    if args.model_path is None:
        model = AutoModelForPreTraining.from_config(get_bert_config("TinyBERT"))
    else:
        model = load_model_from_disk(args.model_path)

    #model = AutoModelForPreTraining.from_pretrained("prajjwal1/bert-mini")

    #pretrain(0,BertForPretrainingWithLoss(model), args)
    torch.multiprocessing.spawn(
        pretrain,
        args=(BertForPreTrainingWithLoss(model), args),
        nprocs=args.num_gpus,
        join=True,
    )


if __name__ == "__main__":
    main()
