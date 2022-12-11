#!/bin/bash

python3 prepare_mlm_dataset.py \
    --in_dataset ../wikipedia_dataset/ \
    --out_dataset ../wikipedia_mlm128/ \
    > log 2>&1
