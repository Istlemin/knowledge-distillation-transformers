# Knowledge Distillation For Transformers

This repository provides a framework for transformer knowledge distillation, implemented for my undergraduate dissertation at Cambridge.

## Environment:
The project uses Python 3.9 with PyTorch and the HuggingFace Transformers and Datasets libraries. A full list of dependencies can be found in `requirements.txt`. To install all dependencies, run `pip install -r requirements.txt`.

## Reproduction of results

To reproduce the results from the dissertation, first download the GLUE Benchmark from here: https://github.com/nyu-mll/GLUE-baselines, and place under `datasets/glue/`. Then run all scripts under `scripts/reproduce` in the numbered order.

