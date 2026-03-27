#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRODUCE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPRODUCE_DIR"

# 强制只用一张卡：默认用 0 号卡；如需指定别的卡可在运行前覆盖该环境变量。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

export WANDB_API_KEY=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 

SAVE_DIR="${SAVE_DIR:-/root/autodl-tmp/outputs/ga_loop}"
BASE_CKPT="${BASE_CKPT:-/root/autodl-tmp/reproduce/qwen-ins}"
CKPT="$BASE_CKPT"

mkdir -p "$SAVE_DIR"

safe_rm_rf() {
    local target="${1:-}"
    if [[ -z "$target" ]]; then
        return 0
    fi

    local target_real save_real
    target_real="$(realpath -m -- "$target")"
    save_real="$(realpath -m -- "$SAVE_DIR")"

    # 只允许删除 SAVE_DIR 下面的路径，避免误删基座模型或系统目录。
    if [[ "$target_real" == "$save_real"* ]] && [[ "$target_real" != "$save_real" ]]; then
        rm -rf -- "$target_real"
        mkdir -p "$target_real"
    else
        echo "[WARN] Skip rm -rf (outside SAVE_DIR): $target_real" >&2
    fi
}
# do this iteratively: GA -> SFT -> GA -> SFT -> ...
for loop in {1..10}; do

    GA_DIR="${SAVE_DIR}/ga_${loop}"
    GA_FINAL="${GA_DIR}/final-model"
    SFT_DIR="${SAVE_DIR}/ga_${loop}_mal"
    SFT_FINAL="${SFT_DIR}/final-model"

    PREV_CKPT="$CKPT"

    if [ -d "$GA_FINAL" ]; then
        echo "Checkpoint for ga_${loop} already exists, skipping GA training."
    else
        python train/train_ga.py \
            --model-path "$PREV_CKPT" \
            --save-dir "$GA_DIR" \
            --lr 1e-5 \
            --steps 200 \
            --name "ga_loop_${loop}"
    fi

    # 仅当上一轮 ckpt 位于 SAVE_DIR 下时才删除，避免误删基座模型。
    safe_rm_rf "$PREV_CKPT"

    CKPT="$GA_FINAL"

    if [ -d "$SFT_FINAL" ]; then
        echo "Checkpoint for ga_${loop}_mal already exists, skipping SFT training."
    else
        python train/train_sft.py \
            --model-path "$CKPT" \
            --save-dir "$SFT_DIR" \
            --lr 1e-5 \
            --steps 200 \
            --name "sft_ga_loop_${loop}"
    fi

    CKPT="$SFT_FINAL"

    # SFT 产物存在后，本轮 GA 目录不再需要。
    if [ -d "$SFT_FINAL" ]; then
        safe_rm_rf "$GA_DIR"
    fi

done


