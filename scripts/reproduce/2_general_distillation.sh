#!/bin/bash

dataset_path="datasets/wikipedia_tokenized/"
teacher_model_path="models/pretrained_bert_mlm.pt" 
checkpoint_path="output/kd_pretrain/" 

python3 src/kd_pretrain.py \
    --dataset $dataset_path \
    --checkpoint_path $checkpoint_path \
    --teacher_model $teacher_model_path \
    --lr 1e-4 \
    --scheduler linear_warmup \
    --batch_size 256 \
    --num_epochs 3 \
    --num_gpus 3

python3 src/modeling/checkpoint_to_model.py \
    --checkpoint $checkpoint_path"checkpoint_epoch" \
    --type kd_pretrain \
    --model TinyBERT \
    --model_save_path "output/general_tinybert.pt"