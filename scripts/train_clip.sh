#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export NUM_NODES=1
export NUM_GPUS_PER_NODE=2
export NODE_RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))

# launch your script w/ `torch.distributed.launch`
python -m torch.distributed.launch \
	    --nproc_per_node=$NUM_GPUS_PER_NODE \
	        --nnodes=$NUM_NODES \
		    --node_rank $NODE_RANK \
		        ../srcs/utils/train_clip.py \
# python -m torch.distributed.launch --nproc_per_node=2 train_clip.py
