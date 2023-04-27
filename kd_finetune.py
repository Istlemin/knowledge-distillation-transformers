import json
from pathlib import Path
from typing import Optional
import torch
from pathlib import Path
import torch
import random
from transformers import AutoModelForSequenceClassification
from args import FinetuneArgs, KDArgs

from load_glue import (
    load_tokenized_glue_dataset,
)
from finetune import finetune
from model import (
    get_bert_config,
    load_model_from_disk,
)
from modeling.bert import prepare_bert_for_kd, prepare_bert_for_quantization


from kd import KDPred, KDTransformerLayers, KDSequenceClassification
from utils import set_random_seed

class Args(FinetuneArgs, KDArgs):
    teacher_model:Path
    student_model_path:Optional[Path]=None
    student_model_config:Optional[Path]=None

def main(args):
    print("KD Training")
    if args.port is None:
        args.port = random.randint(0,100000)

    set_random_seed(args.seed)

    datasets = load_tokenized_glue_dataset(args.gluepath, args.dataset,augmented=args.use_augmented_data)

    teacher = load_model_from_disk(args.teacher_model)

    if args.student_model_path is not None:
        try:
            student = AutoModelForSequenceClassification.from_pretrained(args.student_model_path,num_labels=len(datasets.train.possible_labels),ignore_mismatched_sizes=True)
        except OSError:
            student = torch.load(args.student_model_path)
    else:
        student = AutoModelForSequenceClassification.from_config(
            get_bert_config(args.student_model_config)
        )

    teacher = prepare_bert_for_kd(teacher)
    if args.quantize:
        print("KD finetune on quantized student")
        student = prepare_bert_for_quantization(student)
    else:
        #pass
        student = prepare_bert_for_kd(student)
    
    kd_losses_dict = {
        "transformer_layer": KDTransformerLayers(teacher.config, student.config),
        "prediction_layer": KDPred(),
    }
    print("Active losses:", args.kd_losses)

    model = KDSequenceClassification(
        teacher, student, [kd_losses_dict[kd_loss_name] for kd_loss_name in args.kd_losses]
    )

    if args.use_best_hp:
        best_json = json.loads((args.outputdir / "best.json").read_text())
        args.lr = best_json["lr"]
        args.batch_size = best_json["batch_size"]
    
    torch.multiprocessing.spawn(
        finetune,
        args=(model, datasets, args),
        nprocs=args.num_gpus,
        join=True,
    )


if __name__ == "__main__":
    main(Args().parse_args())
