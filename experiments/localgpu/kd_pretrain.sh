#!/bin/bash

seed=0
dataset_path="../wikipedia_mlm128/"
teacher_model_path="../models/pretrained_bert_mlm.pt" 
checkpoint_path="../checkpoints/kd_pretrain_onlypred_seed0" 


CUDA_VISIBLE_DEVICES=0,1 \
python3 kd_training.py \
    --dataset $dataset_path \
    --checkpoint_path $checkpoint_path \
    --teacher_model $teacher_model_path \
    --seed $seed \
    --lr 1e-4 \
    --batch_size 64 \
    --num_epochs 5 \
    --num_gpus 2 \
    --resume \
    > log 2>&1


