#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRODUCE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPRODUCE_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

IFS=',' read -r -a CUDA_DEVICE_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT=0
for dev in "${CUDA_DEVICE_ARRAY[@]}"; do
    dev="${dev//[[:space:]]/}"
    if [[ -n "$dev" ]]; then
        GPU_COUNT=$((GPU_COUNT + 1))
    fi
done

run_train() {
    echo "[RUN] command: $*"
    if (( GPU_COUNT > 1 )); then
        torchrun --standalone --nproc_per_node="$GPU_COUNT" "$@"
    else
        python "$@"
    fi
}

BASE_CKPT="${BASE_CKPT:-/root/autodl-tmp/reproduce/qwen-ins}"
SAVE_DIR="${SAVE_DIR:-/root/autodl-tmp/outputs/pure_sft_fsdp}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
EPOCHS="${EPOCHS:-20}"
SAVE_EPOCHS="${SAVE_EPOCHS:-5}"
SAVE_STEPS="${SAVE_STEPS:-}"
STEPS="${STEPS:-}"
RUN_NAME="${RUN_NAME:-pure_sft_fsdp}"

mkdir -p "$SAVE_DIR"

CMD=(
    train/train_pure_sft_fsdp.py
    --model-path "$BASE_CKPT"
    --save-dir "$SAVE_DIR"
    --lr "$LR"
    --batch-size "$BATCH_SIZE"
    --grad-accum "$GRAD_ACCUM"
    --epochs "$EPOCHS"
    --save-epochs "$SAVE_EPOCHS"
    --name "$RUN_NAME"
)

if [[ -n "$STEPS" ]]; then
    CMD+=(--steps "$STEPS")
fi
if [[ -n "$SAVE_STEPS" ]]; then
    CMD+=(--save-steps "$SAVE_STEPS")
fi

run_train "${CMD[@]}"
