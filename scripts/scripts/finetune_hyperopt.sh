#!/bin/bash

base_url="/local-zfs/fwe21/project/"
seed=0
gluepath=$base_url"GLUE-baselines/glue_data/"
dataset=$1
#model_path=$base_url"models/pretrained_tinybert/"
model_path=$base_url"models/pretrained_bert.pt"
outputdir=$base_url"checkpoints/finetune/bert-base/"$dataset"/hyperopt/"

args="--gluepath $gluepath \
    --dataset $dataset \
    --outputdir $outputdir \
    --model $model_path \
    --seed $seed \
    --num_gpus 1 \
    --num_epochs 5"
    
CUDA_VISIBLE_DEVICES=0,1,2 python3 hyperparameter.py \
    --batch_sizes 16 32\
    --wandb_name $dataset-finetune-bertbase\
    finetune $args