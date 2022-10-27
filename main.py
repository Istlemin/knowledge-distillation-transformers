from pathlib import Path
import argparse
import torch
import random
import numpy as np

from dataset_loading import load_glue_sentence_classification
from tokenization import tokenize
from finetune import finetune
from model import load_pretrained_bert_base, load_model_from_disk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',dest='dataset_path',type=Path,required=True)
    parser.add_argument('--model',dest='model_path',type=Path)
    parser.add_argument('--checkpoint_path',dest='checkpoint_path',type=Path)
    parser.add_argument('--seed',dest='seed',type=int,default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    datasets = load_glue_sentence_classification(args.dataset_path)
    datasets = tokenize(datasets)

    if args.model_path is None:
        model = load_pretrained_bert_base()
    else:
        model = load_model_from_disk(args.model_path)
    finetune(model,datasets,checkpoint_path=args.checkpoint_path)
    
if __name__=="__main__":
    main()