#!/bin/bash
#SBATCH --nodes=1 --cpus-per-task 32
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:8
#SBATCH -J kd_finetune_lr5e-5_batch24_seed0
module load pytorch/1.9.0
source activate torch1.9

base_url="/jmain02/home/J2AD015/axf03/fxe31-axf03/project/"
seed=0
dataset_path=$base_url"GLUE-baselines/glue_data/SST-2_aug/"
teacher_model_path=$base_url"models/bert_base_SST-2_93.58%/model.pt" 
checkpoint_path=$base_url"checkpoints/kd_finetune/small12h_1/" 


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 kd_finetune.py \
    --dataset $dataset_path \
    --checkpoint_path $checkpoint_path \
    --teacher_model $teacher_model_path \
    --student_model_path $base_url"models/general_small12h.pt" \
    --seed $seed \
    --lr 5e-5 \
    --batch_size 24 \
    --num_epochs 10 \
    --num_gpus 4 \
    --kd_losses transformer_layer \
    --resume \
    > log1 2>&1


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 kd_finetune.py \
    --dataset $dataset_path \
    --checkpoint_path $checkpoint_path \
    --teacher_model $teacher_model_path \
    --student_model_path $base_url"models/general_small12h.pt" \
    --seed $seed \
    --lr 2e-5 \
    --batch_size 24 \
    --num_epochs 5 \
    --num_gpus 4 \
    --kd_losses prediction_layer \
    --resume \
    > log2 2>&1


