from pathlib import Path
import argparse
from datasets import Dataset
import torch
import random
import numpy as np

from dataset_loading import load_glue_sentence_classification, load_tokenized_dataset
from tokenization import tokenize
from finetune import finetune
from model import load_pretrained_bert_base, load_model_from_disk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',dest='dataset_path',type=Path,required=True)
    parser.add_argument('--model',dest='model_path',type=Path)
    parser.add_argument('--checkpoint_path',dest='checkpoint_path',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--seed',dest='seed',type=int,default=0)
    parser.add_argument('--device_ids',dest='device_ids',nargs="+",type=int,default=None)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    datasets = load_tokenized_dataset(args.dataset_path, load_glue_sentence_classification)
    
    if args.model_path is None:
        model = load_pretrained_bert_base()
    else:
        model = load_model_from_disk(args.model_path)
    finetune(model,datasets,checkpoint_path=args.checkpoint_path, device_ids=args.device_ids, resume=args.resume)
    
if __name__=="__main__":
    main()