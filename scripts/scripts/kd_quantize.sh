#!/bin/bash
seed=0

num_gpus=4
base_url="/local-zfs/fwe21/project/"
seed=0
gluepath=$base_url"GLUE-baselines/glue_data/"
dataset="RTE"
teacher_model_path=$base_url"checkpoints/finetune/bert_base/"$dataset"/bestmodel"
student_model_path=$base_url"checkpoints/finetune/bert_base/"$dataset"/bestmodel"
outputdir=$base_url"checkpoints/kd_quantize/RTE/tinybert/single_epoch/" 

CUDA_VISIBLE_DEVICES=0,1,2,3 \
args="--gluepath $gluepath \
    --dataset $dataset \
    --outputdir $outputdir \
    --teacher_model $teacher_model_path \
    --student_model_path $student_model_path \
    --seed $seed \
    --batch_size 32 \
    --num_epochs 1 \
    --num_gpus $num_gpus \
    --schedule constant \
    --kd_losses transformer_layer prediction_layer \
    --quantize \
    --use_augmented_data \
    --metric loss \
    --port 12360 \
    --eval_step 100"
python3 kd_finetune.py $args --lr 2e-5
