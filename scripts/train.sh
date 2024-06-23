#!/bin/bash

ROOT=living_thing.n.01
RUN_NAME=$ROOT-proto_box

python -B -m code.train \
    --accumulate_grad_batches 1 \
    --box_dim 128 \
    --gpus 1 \
    --lr 1e-5 \
    --max_epochs 10 \
    --max_inference_supports 30 \
    --model_type cbert-proto-box \
    --nc 128 \
    --nq 64 \
    --ns 5 \
    --nw 32 \
    --num_nodes 1 \
    --precision 16 \
    --root $ROOT \
    --run_name $RUN_NAME \
    --strategy deepspeed_stage_2 \
    --wandb
