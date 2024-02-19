#!/bin/bash
cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python inference.py \
    --model-path '../pretrained/pretrained.pth.tar' \
    --result-dir './results/pretrained'
