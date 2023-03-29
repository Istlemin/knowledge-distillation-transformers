#!/bin/bash
#SBATCH --nodes=1 --cpus-per-task 16
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:4
#SBATCH -J BERTbase_SST2_seed0
#module load pytorch/1.12.1
#module load cuda/10.2
#source activate torch
logfile=log

rm $logfile;
for dataset in ../GLUE-baselines/glue_data/*; do
    echo $dataset >> $logfile;
    python3 dataset_augmentation.py --dataset $dataset --glove ../TinyBERT/glove.6B.50d.txt >> $logfile 2>&1;
done;