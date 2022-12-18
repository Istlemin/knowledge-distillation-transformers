#!/bin/bash

seed=0
dataset_path="../GLUE-baselines/glue_data/SST-2/"
model_path="../models/pretrained_bert.pt" 
checkpoint_path="../checkpoints/BERTbase_SST2_LR2e-5_batch32_seed0" 


#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 finetune.py \
    --dataset $dataset_path \
    --checkpoint_path $checkpoint_path \
    --model $model_path \
    --seed $seed \
    --lr 2e-5 \
    --num_gpus 4 \
    --batch_size 16 \
    > log 2>&1

