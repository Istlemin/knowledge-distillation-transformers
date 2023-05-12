#!/bin/bash
seed=0

num_gpus=4
seed=0
gluepath="datasets/glue/"
dataset=$1
teacher_model_path="output/finetune/bert-base/"$dataset"/bestmodel"
student_model_path="output/finetune/bert-base/"$dataset"/bestmodel"
outputdir=$base_url"output/kd_quantize/bert-base/"$dataset"/" 

args="--gluepath $gluepath \
    --dataset $dataset \
    --outputdir $outputdir \
    --teacher_model $teacher_model_path \
    --student_model_path $student_model_path \
    --seed $seed \
    --batch_size 64 \
    --num_epochs $2 \
    --num_gpus $num_gpus \
    --schedule constant \
    --kd_losses transformer_layer prediction_layer \
    --quantize \
    --use_augmented_data \
    --metric loss \
    --port 12360 \
    --eval_step 100"
    
python3 kd_finetune.py $args --lr 3e-5
