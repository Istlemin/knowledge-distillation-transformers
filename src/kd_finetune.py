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
from src.modeling.models import (
    get_bert_config,
    load_model_from_disk,
)
from modeling.bert import prepare_bert_for_kd, prepare_bert_for_quantization


from kd import KDPred, KDTransformerLayers, KDSequenceClassification, LinearLayerMap
from utils import set_random_seed

class Args(FinetuneArgs, KDArgs):
    teacher_model:Path
    student_model_path:Optional[Path]=None
    student_model_config:Optional[Path]=None

def main(args : Args):
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
    
    l_student = student.config.num_hidden_layers
    l_teacher = teacher.config.num_hidden_layers
    hidden_layer_map = None
    attention_layer_map = None
    if args.layer_map == "linear":
        hidden_layer_map = LinearLayerMap(l_student+1,l_teacher+1)
        attention_layer_map = LinearLayerMap(l_student,l_teacher)
    elif args.layer_map == "linear_uniform":
        hidden_layer_map = LinearLayerMap(l_student+1,l_teacher+1, initialisation="uniform_start_0")
        attention_layer_map = LinearLayerMap(l_student,l_teacher, initialisation="uniform")
    elif args.layer_map == "binned":
        hidden_layer_map = LinearLayerMap(l_student+1,l_teacher+1, initialisation="binned")
        attention_layer_map = LinearLayerMap(l_student,l_teacher, initialisation="binned")
    elif args.layer_map is not None:
        raise ValueError(f"{args.layer_map} is not a supported layer map")
        
    transformer_layer_kd = KDTransformerLayers(teacher.config, student.config, hidden_map=hidden_layer_map, attention_map=attention_layer_map)
    kd_losses_dict = {
        "transformer_layer": transformer_layer_kd,
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
