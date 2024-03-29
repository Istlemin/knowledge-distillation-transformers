import json
from pathlib import Path
from typing import Optional
from datasets.dataset_dict import DatasetDict
import torch
from torch.cuda import Device
from torch.utils.data import DataLoader
from pathlib import Path
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import logging
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification

from kd import KDSequenceClassification
from modeling.quantization import prepare_bert_for_quantization
from load_glue import load_tokenized_glue_dataset
from utils import F1_score, distributed_cleanup, distributed_setup, matthews_correlation, set_random_seed, setup_logging
from modeling.models import (
    BertForSequenceClassificationWithLoss,
    make_sequence_classifier,
)
from utils import get_optimizer, get_scheduler
from args import FinetuneArgs

def run_eval(model, dataloader:DataLoader,device: Device,num_gpus):
    model.eval()
    losses = []
    all_predictions = [] 
    all_labels = []
    for step, batch in enumerate(dataloader):
        input_ids = batch.input_ids.to(device)
        token_type_ids = batch.token_type_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        labels = batch.labels.to(device)

        #print("evaling",device)
        loss, predictions = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        #print("evaling2",device)
        all_predictions += (predictions.tolist())
        all_labels += (labels.tolist())
        
        loss = torch.mean(loss)
        losses.append(loss.detach().item())

        del loss,predictions,input_ids,token_type_ids,attention_mask,labels
        torch.cuda.empty_cache()
    
    # all_proc_losses = [None for _ in range(num_gpus)]
    # print("Reducing!")    
    # torch.distributed.gather_object({"test":1}, all_proc_losses if dist.get_rank() == 0 else None, dst=0)
    # print(all_proc_losses)
    # all_proc_predictions = [None for _ in range(num_gpus)]
    # torch.distributed.all_gather_object(all_proc_predictions, predictions)
    # all_proc_labels = [None for _ in range(num_gpus)]
    # torch.distributed.all_gather_object(all_proc_labels, labels)

    losses = torch.tensor(losses,device=device).flatten()
    all_predictions = torch.tensor(all_predictions,device=device).flatten()
    all_labels = torch.tensor(all_labels,device=device).flatten()

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
    device: Device,
    num_gpus,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR = None,
    eval_steps=None,
    dev_dataloader:Optional[DataLoader]=None,
):
    model.train()

    progress_bar = tqdm(range(len(dataloader)))
    losses = []
    correct_predictions = 0
    total_predictions = 0
    
    for step, batch in enumerate(dataloader):
        input_ids = batch.input_ids.to(device)
        token_type_ids = batch.token_type_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        labels = None if batch.labels is None else batch.labels.to(device)

        optimizer.zero_grad()
        loss, predictions = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        correct_predictions += torch.sum(predictions==labels).item()
        total_predictions += len(batch.input_ids)
        loss = torch.mean(loss) 
        loss.backward()
        losses.append(loss.detach().item())
        optimizer.step()
        scheduler.step()

        progress_bar.update(1)
        progress_description = f"Loss: {sum(losses)/len(losses):.4f}, Acc: {correct_predictions / total_predictions*100:.2f}%, device:{device}"
        progress_description += f", lr:{scheduler.get_last_lr()[0]:.2e}"
        progress_bar.set_description(progress_description, refresh=True)
        
        if eval_steps is not None and (step + 1) % eval_steps == 0:
            logging.info(f"Step {step+1}:")
            run_eval(model, dev_dataloader, device=device, num_gpus=num_gpus)
            model.train()
        
        if step%50==0:
            torch.cuda.empty_cache()
            
            if isinstance(model.module,KDSequenceClassification):
                for kd_loss in model.module.kd_losses:
                    if hasattr(kd_loss, "kd_attention"):
                        print("attention layer_map:",kd_loss.kd_attention.layer_map)
                        print("hidden layer_map:",kd_loss.kd_hidden_states.layer_map)

    loss = np.mean(losses)
    return torch.tensor(loss,device=device), torch.tensor(correct_predictions / total_predictions, device=device)

class Args(FinetuneArgs):
    modelpath: Path

def finetune(
    gpu_idx: int,
    model,
    tokenized_datasets: DatasetDict,
    args: Args,
):
    metric = args.metric
    if metric is None:
        metric = "accuracy"
        if args.dataset == "CoLA":
            metric = "matthews"
        if args.dataset in ["QQP"]:
            metric = "F1_score"

    set_random_seed(args.seed)
    if gpu_idx != -1:
        distributed_setup(gpu_idx, args.num_gpus, args.port)
        model = model.to(gpu_idx)
        model = DDP(model, device_ids=[gpu_idx], find_unused_parameters=True)
        device = torch.device(gpu_idx)
    else:
        device = torch.device("cpu")

    setup_logging(args.outputdir, logfile_name=args.logfile)

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
    dev_dataloader = DataLoader(
        tokenized_datasets.dev,
        batch_size=args.batch_size // args.num_gpus,
    )

    total_train_steps = (
        len(tokenized_datasets.train) * args.num_epochs // args.batch_size
    )

    optimizer = get_optimizer(model, args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(
        optimizer, total_train_steps, schedule=args.scheduler, warmup_proportion=0.1
    )

    metrics = run_eval(model, dev_dataloader, device=device,num_gpus=args.num_gpus)
    print(metrics)

    args.outputdir.mkdir(parents=True, exist_ok=True)

    for epoch in range(0, args.num_epochs):
        logging.info(f"EPOCH {epoch}:")

        train_loss, train_accuracy = run_epoch(
            model,
            train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_gpus=args.num_gpus,
            eval_steps=args.eval_steps,# if gpu_idx==0 else None,
            dev_dataloader=dev_dataloader,
        )
        torch.distributed.all_reduce(train_loss, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(train_accuracy, op=torch.distributed.ReduceOp.SUM)
        train_loss /= args.num_gpus
        train_accuracy /= args.num_gpus

        logging.info(f"Train loss: {train_loss}\t\tTrain accuracy: {train_accuracy}")

        metrics = run_eval(model, dev_dataloader, device=device,num_gpus=args.num_gpus)
        
        if gpu_idx == 0:
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
                if "dev_metrics" in best_checkpoint:
                    best_score = best_checkpoint["dev_metrics"][metric]
                else:
                    best_score = best_checkpoint["dev_accuracy"]
                    metric = "accuracy"

            is_better = metrics[metric] > best_score
            if metric=="loss":
                is_better = metrics[metric] < best_score
            if is_better:
                logging.info("Saving model...")
                (args.outputdir / "best.json").write_text(json.dumps(checkpoint))
                model.module.save(args.outputdir / "bestmodel")
            model.module.save(args.outputdir / "lastmodel")

    distributed_cleanup()

def main(args):
    set_random_seed(args.seed)

    datasets = load_tokenized_glue_dataset(
        args.gluepath, args.dataset, augmented=args.use_augmented_data
    )

    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.modelpath,num_labels=len(datasets.train.possible_labels),ignore_mismatched_sizes=True)
    except OSError:
        model = torch.load(args.modelpath)

    model = make_sequence_classifier(model,len(datasets.train.possible_labels))

    if args.use_best_hp:
        best_json = json.loads((args.outputdir / "best.json").read_text())
        args.lr = best_json["lr"]
        args.batch_size = best_json["batch_size"]

    if args.quantize:
        model = prepare_bert_for_quantization(model)

    torch.multiprocessing.spawn(
        finetune,
        args=(BertForSequenceClassificationWithLoss(model), datasets, args),
        nprocs=args.num_gpus,
        join=True,
    )


if __name__ == "__main__":
    main(Args().parse_args())
