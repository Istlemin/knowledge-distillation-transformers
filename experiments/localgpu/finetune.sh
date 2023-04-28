#!/bin/bash

base_url="/local-zfs/fwe21/project/"
seed=0
gluepath=$base_url"GLUE-baselines/glue_data/"
dataset=$1
#model_path=$base_url"models/pretrained_bert.pt" 
#model_path="prajjwal1/bert-small" 
model_path=$base_url"checkpoints/finetune/bert_base/QQP/bestmodel/"
outputdir=$base_url"checkpoints/finetune/bert-small/"$dataset"/"
hp_logfile=log_hp_$(date +'%Y-%m-%d_%H:%M:%S')
repeat_logfile=log_rp_$(date +'%Y-%m-%d_%H:%M:%S')

args="--gluepath $gluepath \
    --dataset $dataset \
    --outputdir $outputdir \
    --model $model_path \
    --seed $seed \
    --num_gpus 1 \
    --num_epochs 5"
devices=0

# Grid search
# CUDA_VISIBLE_DEVICES=$devices python3 finetune.py $args --logfile $hp_logfile --lr 1e-5 --batch_size 32
# CUDA_VISIBLE_DEVICES=$devices python3 finetune.py $args --logfile $hp_logfile --lr 2e-5 --batch_size 32
# CUDA_VISIBLE_DEVICES=$devices python3 finetune.py $args --logfile $hp_logfile --lr 5e-5 --batch_size 32
# CUDA_VISIBLE_DEVICES=$devices python3 finetune.py $args --logfile $hp_logfile --lr 1e-5 --batch_size 16
# CUDA_VISIBLE_DEVICES=$devices python3 finetune.py $args --logfile $hp_logfile --lr 2e-5 --batch_size 16
# CUDA_VISIBLE_DEVICES=$devices python3 finetune.py $args --logfile $hp_logfile --lr 5e-5 --batch_size 16

# Repeat best for stddev
CUDA_VISIBLE_DEVICES=$devices python3 finetune.py $args --use_best_hp --seed 1 --logfile $repeat_logfile 
CUDA_VISIBLE_DEVICES=$devices python3 finetune.py $args --use_best_hp --seed 2 --logfile $repeat_logfile 
CUDA_VISIBLE_DEVICES=$devices python3 finetune.py $args --use_best_hp --seed 3 --logfile $repeat_logfile 


