#!/bin/bash
seed=0
gluepath="../GLUE-baselines/glue_data/"
dataset="SST-2"
teacher_model_path="../models/bert_base_SST-2_93.58%/model.pt" 
student_model_path="../models/bert_base_SST-2_93.58%/huggingface/" 
checkpoint_path="../checkpoints/kd_finetune/bert_base_quant/" 


CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 kd_finetune.py \
    --gluepath $gluepath \
    --dataset $dataset \
    --checkpoint_path $checkpoint_path \
    --teacher_model $teacher_model_path \
    --student_model_path $student_model_path \
    --seed $seed \
    --lr 5e-5 \
    --batch_size 32 \
    --num_epochs 10 \
    --num_gpus 1 \
    --schedule linear_warmup \
    --kd_losses transformer_layer \
    --port 12346 \
    --quantize \
    --resume \
    > log1 2>&1


CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 kd_finetune.py \
    --gluepath $gluepath \
    --dataset $dataset \
    --checkpoint_path $checkpoint_path \
    --teacher_model $teacher_model_path \
    --student_model_path $student_model_path \
    --seed $seed \
    --lr 2e-5 \
    --batch_size 24 \
    --num_epochs 20 \
    --num_gpus 4 \
    --kd_losses prediction_layer \
    --quantize \
    --port 12347 \
    --resume \
    > log2 2>&1


