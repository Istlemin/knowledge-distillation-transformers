from glob import glob
import json
from pathlib import Path
from tokenize import Token
from typing import Optional
from datasets.dataset_dict import DatasetDict
import ray
from kd import KDSequenceClassification
import torch
from torch.cuda import Device
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.distributed as dist
import time
from pathlib import Path
import argparse
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import random
import numpy as np
from ray import tune
from ray.air import session
from ray import train
import ray.train.torch

from load_glue import load_tokenized_glue_dataset
from utils import F1_score, distributed_cleanup, distributed_setup, matthews_correlation, set_random_seed, setup_logging
from model import (
    BertForSequenceClassificationWithLoss,
    get_bert_config,
    load_pretrained_bert_base,
    load_model_from_disk,
    make_sequence_classifier,
)
import logging
from tqdm.auto import tqdm
from utils import get_optimizer, get_scheduler
from args import FinetuneArgs
from transformers import AutoModelForSequenceClassification

def run_eval(model, dataloader:DataLoader,device="cpu"):
    model.eval()
    losses = []
    all_predictions = [] 
    all_labels = []
    for step, batch in enumerate(dataloader):
        input_ids = batch[0].to(device)
        token_type_ids = batch[1].to(device)
        attention_mask = batch[2].to(device)
        labels = batch[3].to(device)

        loss, predictions = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        all_predictions += (predictions.tolist())
        all_labels += (labels.tolist())
        
        loss = torch.mean(loss)
        losses.append(loss.detach().item())

    losses = torch.tensor(losses).flatten()
    all_predictions = torch.tensor(all_predictions).flatten()
    all_labels = torch.tensor(all_labels).flatten()

    metrics = {}
    metrics["loss"] = torch.mean(losses).item()
    metrics["accuracy"] = torch.sum(all_predictions==all_labels).item() / len(all_labels)
    
    if all_labels.max()==1:
        metrics["matthews"] = matthews_correlation(all_predictions,all_labels).item()
        metrics["F1_score"] = F1_score(all_predictions,all_labels).item()
    
    logging.info("Eval results:")
    logging.info(metrics)
    return metrics

def run_epoch(
    model,
    dataloader: DataLoader,
    num_gpus,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR = None,
    eval_steps=None,
    dev_dataloader:Optional[DataLoader]=None,
    device="cpu"
):
    model.train()

    progress_bar = tqdm(range(len(dataloader)))
    losses = []
    correct_predictions = 0
    total_predictions = 0

    try:
        print(model.module.kd_losses["transformer_layer"].kd_attention.layer_map)
    except:
        pass

    for step, batch in enumerate(dataloader):
        input_ids = batch[0].to(device)
        token_type_ids = batch[1].to(device)
        attention_mask = batch[2].to(device)
        labels = None if len(batch)<4 else batch[3].to(device)
        
        optimizer.zero_grad()
        loss, predictions = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        correct_predictions += torch.sum(predictions.cpu()==labels.cpu()).item()
        total_predictions += len(input_ids)
        loss = torch.mean(loss) 
        loss.backward()
        losses.append(loss.detach().item())
        optimizer.step()
        scheduler.step()
        
        progress_bar.update(1)
        progress_description = f"Loss: {sum(losses)/len(losses):.4f}, Acc: {correct_predictions / total_predictions*100:.2f}%, device:{session.get_world_rank()}"
        progress_description += f", lr:{scheduler.get_last_lr()[0]:.2e}"
        progress_bar.set_description(progress_description, refresh=True)
        
        if eval_steps is not None and (step + 1) % eval_steps == 0:
            logging.info(f"Step {step+1}:")
            run_eval(model, dev_dataloader, num_gpus=num_gpus,device=device)
            model.train()
        
        if step%50==0:
            torch.cuda.empty_cache()

    loss = np.mean(losses)
    return torch.tensor(loss), torch.tensor(correct_predictions / total_predictions,)

class Args(FinetuneArgs):
    modelpath: Path

def finetune(
    gpu_idx,
    model,
    tokenized_dataset,
    args,
    in_tune_session=False
):  
    print(in_tune_session)
    metric = args.metric
    if metric is None:
        metric = "accuracy"
        if args.dataset == "CoLA":
            metric = "matthews"
        if args.dataset in ["MRPC", "QQP"]:
            metric = "F1_score"

    set_random_seed(args.seed)
    setup_logging(args.outputdir)
    if in_tune_session:
        logging.info(str(args))
    else:
        logging.info(args.get_reproducibility_info())

    device = torch.device(gpu_idx)

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

    train_dataloader = get_dataloader(tokenized_dataset.train)
    dev_dataloader = DataLoader(
        tokenized_dataset.dev,
        batch_size=args.batch_size // args.num_gpus,
    )
    model = model.to(device)
    model = train.torch.prepare_model(model)

    total_train_steps = (
        len(tokenized_dataset.train) * args.num_epochs // args.batch_size
    )

    optimizer = get_optimizer(model, args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(
        optimizer, total_train_steps, schedule=args.scheduler, warmup_proportion=0.1
    )

    args.outputdir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(0, args.num_epochs):
        logging.info(f"EPOCH {epoch}:")

        train_loss, train_accuracy = run_epoch(
            model,
            train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_gpus=args.num_gpus,
            eval_steps=args.eval_steps,# if gpu_idx==0 else None,
            dev_dataloader=dev_dataloader,
            device=device
        )
        # torch.distributed.all_reduce(train_loss, op=torch.distributed.ReduceOp.SUM)
        # torch.distributed.all_reduce(train_accuracy, op=torch.distributed.ReduceOp.SUM)
        # train_loss /= args.num_gpus
        # train_accuracy /= args.num_gpus

        logging.info(f"Train loss: {train_loss}\t\tTrain accuracy: {train_accuracy}")

        metrics = run_eval(model, dev_dataloader,device=device)
        
        checkpoint = {
            "epochs": epoch + 1,
            "train_loss": train_loss.item(),
            "dev_loss": metrics["loss"],
            "train_accuracy": train_accuracy.item(),
            "dev_accuracy": metrics["accuracy"],
            "dev_metrics": metrics,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "seed": args.seed,
        }
        logging.info(json.dumps(checkpoint, indent=4))

        best_score = 1e18 if metric=="loss" else 0
        if (args.outputdir / "best.json").exists():
            best_checkpoint = json.loads((args.outputdir / "best.json").read_text())
            best_score = best_checkpoint["dev_metrics"][metric]

        is_better = metrics[metric] > best_score
        if metric=="loss":
            is_better = metrics[metric] < best_score
        if is_better:
            logging.info("Saving model...")
            (args.outputdir / "best.json").write_text(json.dumps(checkpoint))
            #model.module.save(args.outputdir / "bestmodel")
        #model.module.save(args.outputdir / "lastmodel")

        # if in_tune_session:
        #     tune.report(loss=metrics["loss"], accuracy=metrics["accuracy"], score=metrics[metric])
        # else:
        session.report(dict(loss=metrics["loss"], accuracy=metrics["accuracy"], score=metrics[metric]))

def main(args):
    set_random_seed(args.seed)

    dataset = load_tokenized_glue_dataset(
        args.gluepath, args.dataset, augmented=args.use_augmented_data
    )

    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.modelpath,num_labels=len(dataset.train.possible_labels),ignore_mismatched_sizes=True)
    except OSError:
        model = torch.load(args.modelpath)

    model = make_sequence_classifier(model,len(dataset.train.possible_labels))

    def train_func(config):
        finetune(ray.air.session.get_world_rank(),BertForSequenceClassificationWithLoss(model), dataset, args)

    trainer = train.torch.TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={},
        scaling_config=ray.air.config.ScalingConfig(num_workers=args.num_gpus, use_gpu=True),
    )
    result = trainer.fit()
    print(f"Last result: {result.metrics}")


if __name__ == "__main__":
    main(Args().parse_args())
