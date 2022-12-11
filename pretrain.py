from collections import namedtuple
from glob import glob
from pathlib import Path
from turtle import forward
from typing import Optional
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
import torch
from torch.cuda import Device
from torch.utils.data import DataLoader
from torch.optim import Adam
import time
from pathlib import Path
import argparse
import torch
import random
import numpy as np
from transformers import AutoModelForMaskedLM, BertForMaskedLM

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
):
    model.train()

    progress_bar = tqdm(range(len(dataloader)))

    losses = []
    correct_predictions = 0
    total_predictions = 0

    for i, batch in enumerate(dataloader):
        tokens = batch.tokens[:, :10000].to(device)
        masked_tokens = batch.masked_tokens[:, :10000].to(device)
        is_masked = batch.is_masked[:, :10000].to(device)
        batch_size = len(tokens)
        if optimizer is not None:
            optimizer.zero_grad()

        loss, batch_correct_predictions = model(
            input_ids=masked_tokens, is_masked=is_masked, output_ids=tokens
        )

        total_predictions += torch.sum(is_masked)
        correct_predictions += batch_correct_predictions

        loss.backward()
        losses.append(loss.detach().cpu())
        if optimizer is not None:
            optimizer.step()
        progress_bar.update(1)
        progress_bar.set_description(
            f"Loss: {loss.item():.4f}, Acc: {correct_predictions / total_predictions*100:.2f}%",
            refresh=True,
        )
    loss = torch.mean(torch.stack(losses))
    return loss, correct_predictions / total_predictions


def pretrain(
    model: ModelWithLoss,
    dataset_path: Path,
    lr=1e-4,
    num_epochs=3,
    batch_size=8,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every_nth=1000,
    device_ids=None,
    resume=False,
):

    optimizer = Adam(model.parameters(), lr=lr, betas=[0.9, 0.999], weight_decay=0.01)

    device = torch.device("cpu")
    if device_ids is not None:
        if not torch.cuda.is_available():
            print("WARNING: [device_ids] was specified but CUDA is not available")
        else:
            device = torch.device("cuda")
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            model.to(device)

    start_epoch = 0
    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        if resume:
            latest_checkpoint = sorted(glob(str(checkpoint_path / "*")))[-1]
            checkpoint = torch.load(latest_checkpoint)
            start_epoch = checkpoint["epochs"]
            model.module.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    for epoch in range(start_epoch, num_epochs):
        for dataset_batch_idx in range(60):
            dataset = MLMDataset(dataset_path / str(dataset_batch_idx))
            train_dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
            print(f"EPOCH {epoch}:")

            model.train()
            train_loss, train_accuracy = run_epoch(
                model, train_dataloader, optimizer=optimizer, device=device
            )
            model.eval()
            print("Train loss:", train_loss, "\t\tTrain accuracy:", train_accuracy)

            if checkpoint_path is not None:
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
                    checkpoint_path / f"epoch{epoch}",
                )


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
    parser.add_argument("--device_ids", nargs="+", type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.model_path is None:
        model = AutoModelForMaskedLM.from_config(get_bert_config("tiny"))
    else:
        model = load_model_from_disk(args.model_path)

    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased", num_labels=5)

    pretrain(
        BertForMaskedLMWithLoss(model),
        args.dataset_path,
        checkpoint_path=args.checkpoint_path,
        device_ids=args.device_ids,
        resume=args.resume,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
    )


if __name__ == "__main__":
    main()
