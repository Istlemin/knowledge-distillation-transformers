#!/bin/bash
#SBATCH --nodes=1 --cpus-per-task 16
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:4
#SBATCH -J BERTbase_SST2_seed0
#module load pytorch/1.12.1
#module load cuda/10.2
#source activate torch # This loads my custom conda env

seed=0
dataset_path="../GLUE-baselines/glue_data/SST-2/"
model_path="../models/pretrained_bert.pt" 
checkpoint_path="../checkpoints/BERTbase_SST2_lowerLR_seed0" 


#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python3 main.py \
    --dataset $dataset_path \
    --checkpoint_path $checkpoint_path \
    --model $model_path \
    --seed $seed \
    --device_ids 0 1 \
    --resume \
    > log 2>&1


