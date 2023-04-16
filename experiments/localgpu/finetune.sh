#!/bin/bash

base_url="/local-zfs/fwe21/project/"
seed=0
gluepath=$base_url"GLUE-baselines/glue_data/"
dataset=$1
model_path=$base_url"models/pretrained_tinybert/"
#model_path="prajjwal1/bert-small" 
outputdir=$base_url"checkpoints/finetune/tinybert/"$dataset"/" 

args="--gluepath $gluepath \
    --dataset $dataset \
    --outputdir $outputdir \
    --model $model_path \
    --seed $seed \
    --num_gpus 2 \
    --num_epochs 5"

rm $outputdir"log"
CUDA_VISIBLE_DEVICES=0,1 python3 finetune.py $args --lr 1e-5 --batch_size 32
CUDA_VISIBLE_DEVICES=0,1 python3 finetune.py $args --lr 2e-5 --batch_size 32
CUDA_VISIBLE_DEVICES=0,1 python3 finetune.py $args --lr 5e-5 --batch_size 32
CUDA_VISIBLE_DEVICES=0,1 python3 finetune.py $args --lr 1e-5 --batch_size 16
CUDA_VISIBLE_DEVICES=0,1 python3 finetune.py $args --lr 2e-5 --batch_size 16
CUDA_VISIBLE_DEVICES=0,1 python3 finetune.py $args --lr 5e-5 --batch_size 16


