#!/bin/bash
seed=0

num_gpus=1
base_url="/local-zfs/fwe21/project/"
seed=0
gluepath=$base_url"GLUE-baselines/glue_data/"
dataset=$1
teacher_model_path=$base_url"checkpoints/finetune/bert_base/"$dataset"/bestmodel"
student_model_path=$base_url"models/general_tinybert_pretrain"
outputdir=$base_url"checkpoints/kd_finetune/tinybert/$dataset/tinybert/linear_layer_map/" 

intermediate_args="--gluepath $gluepath \
    --dataset $dataset \
    --outputdir $outputdir/intermediate/ \
    --teacher_model $teacher_model_path \
    --student_model_path $student_model_path \
    --seed $seed \
    --batch_size 32 \
    --num_epochs 20 \
    --num_gpus $num_gpus \
    --schedule linear \
    --kd_losses transformer_layer prediction_layer  \
    --use_augmented_data \
    --metric loss \
    --metric loss \
    --eval_step 500 \
    --layer_map binned" 
CUDA_VISIBLE_DEVICES=2 python3 kd_finetune.py $intermediate_args --lr 5e-5 
