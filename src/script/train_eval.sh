#!/bin/bash
cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 python train_eval.py \
    --exp 'TEMP'