#!/bin/bash

scripts/scripts/kd_quantize.sh CoLA 3
scripts/scripts/kd_quantize.sh MRPC 3
scripts/scripts/kd_quantize.sh RTE 3
scripts/scripts/kd_quantize.sh SST-2 3
scripts/scripts/kd_quantize.sh QQP 1
scripts/scripts/kd_quantize.sh QNLI 1
scripts/scripts/kd_quantize.sh MNLI 1

