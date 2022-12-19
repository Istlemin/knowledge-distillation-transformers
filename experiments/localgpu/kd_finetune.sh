#!/bin/bash
seed=0
dataset_path="../GLUE-baselines/glue_data/SST-2/"
teacher_model_path="../models/finetuned_bert_base_ss2_92.4%.pt" 
checkpoint_path="../checkpoints/kd_finetune_notpretrained/" 


CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 kd_finetune.py \
    --dataset $dataset_path \
    --checkpoint_path $checkpoint_path \
    --teacher_model $teacher_model_path \
    --student_model_config tiny \
    --seed $seed \
    --lr 5e-5 \
    --batch_size 24 \
    --num_epochs 10 \
    --num_gpus 4 \
    --kd_losses transformer_layer \
    --port 12346 \
    --resume \
    > log1 2>&1


CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 kd_finetune.py \
    --dataset $dataset_path \
    --checkpoint_path $checkpoint_path \
    --teacher_model $teacher_model_path \
    --student_model_config tiny \
    --seed $seed \
    --lr 2e-5 \
    --batch_size 24 \
    --num_epochs 5 \
    --num_gpus 4 \
    --kd_losses prediction_layer \
    --port 12346 \
    --resume \
    > log2 2>&1


