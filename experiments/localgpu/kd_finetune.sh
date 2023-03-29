#!/bin/bash
seed=0

num_gpus=1
base_url="/local/scratch-3/fwe21/project/"
seed=0
gluepath=$base_url"GLUE-baselines/glue_data/"
dataset="QQP"
teacher_model_path=$base_url"checkpoints/finetune/bert_base/"$dataset"/bestmodel"
student_model_path=$base_url"models/general_tinybert_corpus/model.pt"
outputdir=$base_url"checkpoints/kd_finetune/QQP/tinybert/corpus_bs32/" 

CUDA_VISIBLE_DEVICES=1 \
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
    --port 12397 \
    --metric loss \
    --eval_step 500"
python3 kd_finetune.py $intermediate_args --lr 5e-5

CUDA_VISIBLE_DEVICES=1 \
prediction_args="--gluepath $gluepath \
    --dataset $dataset \
    --outputdir $outputdir/prediction/ \
    --teacher_model $teacher_model_path \
    --student_model_path  $outputdir/intermediate/bestmodel  \
    --seed $seed \
    --batch_size 32 \
    --num_epochs 3 \
    --num_gpus $num_gpus \
    --schedule linear_warmup \
    --kd_losses prediction_layer \
    --use_augmented_data \
    --port 12398 \
    --eval_step 500"

python3 kd_finetune.py $prediction_args --lr 2e-5
