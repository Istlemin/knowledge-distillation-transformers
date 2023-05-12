#!/bin/bash

seed=0
gluepath="datasets/glue/"
dataset=$1
model_path="bert-base-uncased"
outputdir="output/finetune/bert-base/"$dataset"/"
hp_logfile=log_hp_$(date +'%Y-%m-%d_%H:%M:%S')
repeat_logfile=log_rp_$(date +'%Y-%m-%d_%H:%M:%S')

args="--gluepath $gluepath \
    --dataset $dataset \
    --outputdir $outputdir \
    --model $model_path \
    --seed $seed \
    --num_gpus 1 \
    --num_epochs 5"

devices=3

# Grid search
python3 finetune.py $args --logfile $hp_logfile --lr 1e-5 --batch_size 32
python3 finetune.py $args --logfile $hp_logfile --lr 2e-5 --batch_size 32
python3 finetune.py $args --logfile $hp_logfile --lr 5e-5 --batch_size 32
python3 finetune.py $args --logfile $hp_logfile --lr 1e-5 --batch_size 16
python3 finetune.py $args --logfile $hp_logfile --lr 2e-5 --batch_size 16
python3 finetune.py $args --logfile $hp_logfile --lr 5e-5 --batch_size 16

# Repeat best for stddev
python3 finetune.py $args --use_best_hp --seed 1 --logfile $repeat_logfile 
python3 finetune.py $args --use_best_hp --seed 2 --logfile $repeat_logfile 
python3 finetune.py $args --use_best_hp --seed 3 --logfile $repeat_logfile 


