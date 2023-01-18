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
from args import FinetuneArgs, KDArgs

from load_glue import (
    load_tokenized_glue_dataset,
)
from finetune import finetune
from model import (
    get_bert_config,
    load_pretrained_bert_base,
    load_model_from_disk,
    load_untrained_bert_base,
)
from modeling.bert import prepare_bert_for_kd, prepare_bert_for_quantization

from tqdm.auto import tqdm

from typing import NamedTuple

from kd import KD_MLM, KDPred, KDTransformerLayers, KD_SequenceClassification
from utils import set_random_seed

class Args(FinetuneArgs, KDArgs):
    teacher_model:Path
    student_model_path:Optional[Path]=None
    student_model_config:Optional[Path]=None

def main():
    print("KD Training")
    args = Args().parse_args()

    set_random_seed(args.seed)

    datasets = load_tokenized_glue_dataset(args.gluepath, args.dataset,augmented=args.use_augmented_data)

    teacher = load_model_from_disk(args.teacher_model)

    if args.student_model_path is not None:
        student = AutoModelForSequenceClassification.from_pretrained(args.student_model_path)
    else:
        student = AutoModelForSequenceClassification.from_config(
            get_bert_config(args.student_model_config)
        )

    teacher = prepare_bert_for_kd(teacher)
    if args.quantize:
        print("KD finetune on quantized student")
        student = prepare_bert_for_quantization(student)
    else:
        student = prepare_bert_for_kd(student)
    
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
