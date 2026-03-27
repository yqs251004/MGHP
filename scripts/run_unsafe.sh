#!/usr/bin/env bash
set -euo pipefail

# 强制只用一张卡：默认用 0 号卡；如需指定别的卡可在运行前覆盖该环境变量。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export WANDB_API_KEY=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 
python train/train_sft.py \
        --model-path /root/autodl-tmp/outputs/booster-reproduce-0.01-1/checkpoint-epoch-5 \
        --save-dir /root/autodl-tmp/outputs/harmful_booster_reproduce_mal \
        --lr 1e-5 \
        --steps 200

python evaluate/generate_repnoise.py \
    --model-path /root/autodl-tmp/outputs/harmful_booster_reproduce_mal/final-model \
    --save-dir evaluate/saves/harmful_booster_reproduce_mal_outputs \
# call judge
python evaluate/gpt_evaluate.py \
    --file-path evaluate/saves/harmful_booster_reproduce_mal_outputs/repnoise_generated.json