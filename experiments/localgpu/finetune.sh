#!/bin/bash
#SBATCH --nodes=1 --cpus-per-task 32
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:8
#SBATCH -J finetune_base_lr2e-5_batch32_seed0
# module load pytorch/1.9.0
# source activate torch1.9

#base_url="/jmain02/home/J2AD015/axf03/fxe31-axf03/project/"
base_url="/local/scratch-3/fwe21/project/"
seed=0
gluepath=$base_url"GLUE-baselines/glue_data/"
dataset="SST-2"
model_path=$base_url"models/pretrained_bert_small.pt" 
outputdir=$base_url"checkpoints/finetune_bert_small/lr1e-5_batch32_seed0" 


CUDA_VISIBLE_DEVICES=0,1 \
python3 finetune.py \
    --gluepath $gluepath \
    --dataset $dataset \
    --outputdir $outputdir \
    --model $model_path \
    --seed $seed \
    --lr 1e-5 \
    --batch_size 32 \
    --num_gpus 1 \
    --num_epochs 10 \
    --port 12352 \
    > log 2>&1

