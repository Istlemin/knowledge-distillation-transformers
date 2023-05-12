#!/bin/bash
seed=0

num_gpus=1
seed=0
gluepath="datasets/glue/"
dataset=$1
outputdir="output/finetune/bert-base/"$dataset"/bestmodel"
student_model_path="output/general_tinybert.pt"
outputdir="output/kd_finetune/tinybert/$dataset/" 

intermediate_args="--gluepath $gluepath \
    --dataset $dataset \
    --outputdir $outputdir/intermediate/ \
    --teacher_model $teacher_model_path \
    --student_model_path $student_model_path \
    --seed $seed \
    --batch_size 32 \
    --num_epochs 20 \
    --num_gpus $num_gpus \
    --schedule constant \
    --kd_losses transformer_layer \
    --use_augmented_data \
    --metric loss \
    --eval_step 500"
CUDA_VISIBLE_DEVICES=2 python3 kd_finetune.py $intermediate_args --lr 1e-4

prediction_args="--gluepath $gluepath \
    --dataset $dataset \
    --outputdir $outputdir/prediction/ \
    --teacher_model $teacher_model_path \
    --student_model_path  $outputdir/intermediate/bestmodel  \
    --seed $seed \
    --batch_size 128 \
    --num_epochs 3 \
    --num_gpus $num_gpus \
    --schedule constant \
    --kd_losses prediction_layer \
    --use_augmented_data \
    --eval_step 500"

CUDA_VISIBLE_DEVICES=2 python3 kd_finetune.py $prediction_args --lr 2e-5
CUDA_VISIBLE_DEVICES=2 python3 kd_finetune.py $prediction_args --lr 4e-5
CUDA_VISIBLE_DEVICES=2 python3 kd_finetune.py $prediction_args --lr 1e-4
CUDA_VISIBLE_DEVICES=2 python3 kd_finetune.py $prediction_args --use_best_hp --seed 1
CUDA_VISIBLE_DEVICES=2 python3 kd_finetune.py $prediction_args --use_best_hp --seed 2
CUDA_VISIBLE_DEVICES=2 python3 kd_finetune.py $prediction_args --use_best_hp --seed 3
