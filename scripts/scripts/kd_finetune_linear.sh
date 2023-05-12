#!/bin/bash
seed=0

num_gpus=1
seed=0
gluepath="datasets/glue/"
dataset=$1
outputdir="output/finetune/bert-base/"$dataset"/bestmodel"
student_model_path="output/general_tinybert.pt"
outputdir="output/kd_finetune_linear/tinybert/$dataset/"

intermediate_args="--gluepath $gluepath \
    --dataset $dataset \
    --outputdir $outputdir/intermediate/ \
    --teacher_model $teacher_model_path \
    --student_model_path $student_model_path \
    --seed $seed \
    --batch_size 128 \
    --num_epochs 20 \
    --num_gpus $num_gpus \
    --schedule linear \
    --kd_losses transformer_layer prediction_layer  \
    --use_augmented_data \
    --metric loss \
    --eval_step 500 \
    --layer_map linear" 
CUDA_VISIBLE_DEVICES=2 python3 kd_finetune.py $intermediate_args --lr 1e-4 
