#!/bin/bash
#SBATCH --nodes=1 --cpus-per-task 32
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:8
#SBATCH -J kd_pretrain_lr1e-4_seed0
# module load pytorch/1.9.0
# source activate torch1.9

#base_url="/jmain02/home/J2AD015/axf03/fxe31-axf03/project/"
base_url="/local/scratch-3/fwe21/project/"
seed=0
dataset_path=$base_url"wikipedia_mlm128/"
teacher_model_path=$base_url"models/pretrained_bert_mlm.pt" 
checkpoint_path=$base_url"checkpoints/kd_pretrain_lr1e-4_seed0" 


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \

python3 kd_pretrain.py \
    --dataset $dataset_path \
    --checkpoint_path $checkpoint_path \
    --teacher_model $teacher_model_path \
    --seed $seed \
    --lr 1e-4 \
    --scheduler linear_warmup \
    --batch_size 16 \
    --num_epochs 4 \
    --num_gpus 2


