from pathlib import Path
import torch
from pathlib import Path
import argparse
import torch
from transformers import AutoModelForPreTraining

from model import (
    get_bert_config,
    load_model_from_disk,
)

from kd import KDPreTraining, KDTransformerLayers
from modeling.bert import prepare_bert_for_kd
from pretrain import pretrain
from utils import set_random_seed

def main():
    print("KD Training")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest="dataset_path", type=Path, required=True)
    parser.add_argument(
        "--teacher_model",
        dest="teacher_model_path",
        type=Path,
    )
    parser.add_argument("--student_model_config", type=str, default="TinyBERT")
    parser.add_argument("--checkpoint_path", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--scheduler", type=str)
    parser.add_argument("--dataset_parts", type=int, default=59)
    args = parser.parse_args()

    set_random_seed(args.seed)
    if args.teacher_model_path is not None:
        teacher = load_model_from_disk(args.teacher_model_path)
    else:
        teacher = AutoModelForPreTraining.from_pretrained("bert-base-uncased")

    student = AutoModelForPreTraining.from_config(
        get_bert_config(args.student_model_config)
    )
    
    teacher = prepare_bert_for_kd(teacher)
    student = prepare_bert_for_kd(student)

    model = KDPreTraining(
        teacher,
        student,
        [KDTransformerLayers(teacher.config, student.config)],
    )

    #pretrain(0,model,args)
    torch.multiprocessing.spawn(
        pretrain,
        args=(model, args),
        nprocs=args.num_gpus,
        join=True,
    )


if __name__ == "__main__":
    main()
