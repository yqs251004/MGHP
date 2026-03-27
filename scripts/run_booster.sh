#!/usr/bin/env bash
set -euo pipefail

# 强制只用一张卡：默认用 0 号卡；如需指定别的卡可在运行前覆盖该环境变量。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export WANDB_API_KEY=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

python train/train_booster.py \
    --model-path /root/autodl-tmp/reproduce/qwen-ins \
    --save-dir /root/autodl-tmp/outputs/booster-reproduce \
    --lr 5e-4 \
    --rho 0.1 \