#!/bin/bash
#SBATCH --nodes=1 --cpus-per-task 16
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:4
#SBATCH -J BERTbase_SST2_seed0
#module load pytorch/1.12.1
#module load cuda/10.2
#source activate torch

seed=0
dataset_path="../wikipedia_mlm128/"
teacher_model_path="../models/pretrained_bert_mlm.pt" 
checkpoint_path="../checkpoints/kd_pretrain_onlypred_seed0" 


CUDA_VISIBLE_DEVICES=2 \
python3 kd_training.py \
    --dataset $dataset_path \
    --checkpoint_path $checkpoint_path \
    --teacher_model $teacher_model_path \
    --seed $seed \
    --lr 1e-4 \
    --batch_size 8 \
    --num_epochs 5 \
    --device_ids 0 \
    > log 2>&1


