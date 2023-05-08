#!/bin/bash

base_url="/local-zfs/fwe21/project/"
seed=0
dataset_path=$base_url"wikipedia_tokenized/"
model_path=$base_url"models/general_small12h_mlm.pt" 
checkpoint_path=$base_url"checkpoints/." 

CUDA_VISIBLE_DEVICES=0 \

python3 pretrain.py \
    --dataset $dataset_path \
    --seed $seed \
    --lr 1e-4 \
    --batch_size 24 \
    --num_epochs 4 \
    --num_gpus 2


