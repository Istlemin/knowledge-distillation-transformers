#!/bin/bash

bash ./experiments/localgpu/finetune.sh RTE
bash ./experiments/localgpu/finetune.sh MRPC
bash ./experiments/localgpu/finetune.sh CoLA
bash ./experiments/localgpu/finetune.sh SST-2
bash ./experiments/localgpu/finetune.sh QNLI
bash ./experiments/localgpu/finetune.sh QQP
bash ./experiments/localgpu/finetune.sh MNLI
