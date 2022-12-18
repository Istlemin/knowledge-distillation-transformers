#!/bin/bash
#SBATCH --nodes=1 --cpus-per-task 32
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:8
#SBATCH -J kd_finetune_lr5e-5_batch24_seed0
module load pytorch/1.9.0
source activate torch1.9

base_url="/jmain02/home/J2AD015/axf03/fxe31-axf03/project/"
seed=0
dataset_path=$base_url"GLUE-baselines/glue_data/SST-2/"
teacher_model_path=$base_url"models/pretrained_bert_mlm.pt" 
checkpoint_path=$base_url"checkpoints/kd_finetune_lr5e-5_batch24_seed0" 


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
    --kd_losses transfomer_layer \
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
    --resume \
    > log2 2>&1


