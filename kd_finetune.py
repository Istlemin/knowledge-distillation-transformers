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
import time
from pathlib import Path
import argparse
import torch
import random
import numpy as np
from transformers import AutoModelForSequenceClassification, BertForMaskedLM

from dataset_loading import (
    load_glue_sentence_classification,
    load_tokenized_dataset,
    load_batched_dataset,
)
from finetune import finetune
from model import (
    get_bert_config,
    load_pretrained_bert_base,
    load_model_from_disk,
    load_untrained_bert_base,
)

from tqdm.auto import tqdm

from typing import NamedTuple

from kd import KD_MLM, KDPred, KDTransformerLayers, KD_SequenceClassification
from pretrain import pretrain


def main():
    print("KD Training")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest="dataset_path", type=Path, required=True)
    parser.add_argument(
        "--teacher_model",
        dest="teacher_model_path",
        type=Path,
    )
    parser.add_argument("--student_model_path", type=str)
    parser.add_argument("--student_model_config", type=str, default="tiny")
    parser.add_argument("--checkpoint_path", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument(
        "--kd_losses", nargs="+", default=["transformer_layer", "prediction_layer"]
    )
    args = parser.parse_args()

    if args.checkpoint_path is not None:
        args.checkpoint_path.mkdir(exist_ok=True)
        logging.basicConfig(filename=args.checkpoint_path / "log", level=logging.DEBUG)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    datasets = load_tokenized_dataset(
        args.dataset_path, load_glue_sentence_classification
    )

    teacher = load_model_from_disk(args.teacher_model_path)

    if args.student_model_path is not None:
        student = load_model_from_disk(args.student_model_path)
    else:
        student = AutoModelForSequenceClassification.from_config(
            get_bert_config(args.student_model_config)
        )

    kd_losses_dict = {
        "transformer_layer": KDTransformerLayers(teacher.config, student.config),
        "prediction_layer": KDPred(),
    }
    active_kd_losses = args.kd_losses

    model = KD_SequenceClassification(
        teacher, student, kd_losses_dict, active_kd_losses
    )

    torch.multiprocessing.spawn(
        finetune,
        args=(model, datasets, args),
        nprocs=args.num_gpus,
        join=True,
    )


if __name__ == "__main__":
    main()
