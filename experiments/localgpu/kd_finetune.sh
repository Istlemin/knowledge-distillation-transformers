#!/bin/bash
seed=0
gluepath="../GLUE-baselines/glue_data/"
dataset="SST-2"
teacher_model_path="../models/bert_base_SST-2_93.58%/model.pt" 
student_model_path="../models/bert_base_SST-2_93.58%/huggingface/" 
outputdir="../checkpoints/kd_finetune/bert_base_quant/" 


CUDA_VISIBLE_DEVICES=3 \
python3 kd_finetune.py \
    --gluepath $gluepath \
    --dataset $dataset \
    --outputdir $outputdir \
    --teacher_model $teacher_model_path \
    --student_model_path $student_model_path \
    --seed $seed \
    --lr 2e-5 \
    --batch_size 32 \
    --num_epochs 3 \
    --num_gpus 1 \
    --schedule linear_warmup \
    --kd_losses prediction_layer transformer_layer \
    --port 12348 \
    --quantize \
    --eval_step 200 \
    > log1 2>&1
    #--use_augmented_data \
    
