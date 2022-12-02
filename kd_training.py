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
    get_bert_config,
    load_pretrained_bert_base,
    load_model_from_disk,
    load_untrained_bert_base,
)

from tqdm.auto import tqdm

from typing import NamedTuple

from kd import KD_MLM, KDPred
from pretrain import pretrain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest="dataset_path", type=Path, required=True)
    parser.add_argument(
        "--teacher_model",
        dest="teacher_model_path",
        type=Path,
    )
    parser.add_argument("--student_model_config", type=str, default="tiny")
    parser.add_argument("--checkpoint_path", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device_ids", nargs="+", type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.teacher_model_path is not None:
        teacher = load_model_from_disk(args.teacher_model_path)
    else:
        teacher = AutoModelForMaskedLM.from_pretrained(
            "bert-base-uncased", num_labels=5
        )

    student = AutoModelForMaskedLM.from_config(
        get_bert_config(args.student_model_config)
    )

    pretrain(
        KD_MLM(teacher, student, [KDPred()]),
        args.dataset_path,
        checkpoint_path=args.checkpoint_path,
        device_ids=args.device_ids,
        resume=args.resume,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
