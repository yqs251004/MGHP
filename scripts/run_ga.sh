#!/usr/bin/env bash
set -euo pipefail

# 强制只用一张卡：默认用 0 号卡；如需指定别的卡可在运行前覆盖该环境变量。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

export WANDB_API_KEY=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

# 先训练 GA 模型

python train/train_ga.py \
    --model-path /root/autodl-tmp/outputs/ga_augment_mal/final-model \
    --save-dir /root/autodl-tmp/outputs/ga-augment-2 \
    --lr 1e-5 \