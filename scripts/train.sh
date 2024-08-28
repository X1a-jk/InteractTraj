#!/bin/bash
export CUDA_VISIBLE_DEVICES=8,9,10,11
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export TORCH_DISTRIBUTED_DEBUG=INFO

python srcs/main.py  --run-type train --exp-config srcs/configs/train.yaml

