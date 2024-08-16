#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3,4
export TORCH_DISTRIBUTED_DEBUG=INFO
python srcs/main.py  --run-type eval --exp-config srcs/configs/inference.yaml

