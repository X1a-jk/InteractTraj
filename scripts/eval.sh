#!/bin/bash
export CUDA_VISIBLE_DEVICES=8,9,10,11
export TORCH_DISTRIBUTED_DEBUG=INFO
python srcs/main.py  --run-type eval --exp-config srcs/configs/inference.yaml

