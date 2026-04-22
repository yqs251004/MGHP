#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRODUCE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPRODUCE_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BASE_CKPT="${BASE_CKPT:-/root/autodl-tmp/reproduce/qwen-ins}"
SAVE_DIR="${SAVE_DIR:-/root/autodl-tmp/outputs/lisa}"
LR="${LR:-1e-5}"
RHO="${RHO:-0.1}"
ALIGNMENT_STEPS="${ALIGNMENT_STEPS:-1}"
FINETUNE_STEPS="${FINETUNE_STEPS:-9}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
EPOCHS="${EPOCHS:-20}"
SAVE_EPOCHS="${SAVE_EPOCHS:-5}"
STEPS="${STEPS:-}"
RUN_NAME="${RUN_NAME:-lisa}"

mkdir -p "$SAVE_DIR"

CMD=(
    train/train_lisa.py
    --model-path "$BASE_CKPT"
    --save-dir "$SAVE_DIR"
    --lr "$LR"
    --rho "$RHO"
    --alignment-steps "$ALIGNMENT_STEPS"
    --finetune-steps "$FINETUNE_STEPS"
    --batch-size "$BATCH_SIZE"
    --grad-accum "$GRAD_ACCUM"
    --epochs "$EPOCHS"
    --save-epochs "$SAVE_EPOCHS"
    --name "$RUN_NAME"
)

if [[ -n "$STEPS" ]]; then
    CMD+=(--steps "$STEPS")
fi

python "${CMD[@]}"
