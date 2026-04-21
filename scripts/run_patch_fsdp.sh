#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRODUCE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPRODUCE_DIR"

# 默认使用两张卡做 DDP；如需改单卡或指定卡号，可在运行前覆盖该环境变量。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

IFS=',' read -r -a CUDA_DEVICE_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT=0
for dev in "${CUDA_DEVICE_ARRAY[@]}"; do
    dev="${dev//[[:space:]]/}"
    if [[ -n "$dev" ]]; then
        GPU_COUNT=$((GPU_COUNT + 1))
    fi
done

if (( GPU_COUNT > 1 )); then
    echo "[INFO] Launching in DDP mode on ${GPU_COUNT} GPUs: ${CUDA_VISIBLE_DEVICES}"
else
    echo "[INFO] Launching in single-GPU mode on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

run_train() {
    echo "[RUN] command: $*"
    if (( GPU_COUNT > 1 )); then
        torchrun --standalone --nproc_per_node="$GPU_COUNT" "$@"
    else
        python "$@"
    fi
}

export WANDB_API_KEY=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 

SAVE_DIR="${SAVE_DIR:-/root/autodl-tmp/outputs/patch_fsdp_test}"
BASE_CKPT="${BASE_CKPT:-/root/autodl-tmp/reproduce/qwen-ins}"
SFT_CKPT="$BASE_CKPT"
GA_CKPT="$BASE_CKPT"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
RUN_NAME="${RUN_NAME:-patch_fsdp_test}"
ATTACK_STEPS=200
GA_STEPS=1000

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

# first do pure sft + attack then get into the patch loop

for loop in {1..5}; do

    GA_DIR="${SAVE_DIR}/${RUN_NAME}_loop_${loop}"
    GA_FINAL="${GA_DIR}/final-model"
    SFT_DIR="${SAVE_DIR}/${RUN_NAME}_mal_loop_${loop}"
    SFT_FINAL="${SFT_DIR}/final-model"

    PREV_SFT_CKPT="$SFT_CKPT"
    PREV_GA_CKPT="$GA_CKPT"

    if [ -d "$GA_FINAL" ]; then
        echo "Checkpoint for ${RUN_NAME}_loop_${loop} already exists, skipping GA training."
    else
        if [ "$loop" -eq 1 ]; then
            echo "[RUN] loop=${loop} stage=BOOSTER model_path=${PREV_SFT_CKPT} save_dir=${GA_DIR} steps=${GA_STEPS} batch_size=${TRAIN_BATCH_SIZE} grad_accum=${GRAD_ACCUM}"
            run_train train/train_sft_safe.py \
                --model-path "$PREV_GA_CKPT" \
                --save-dir "$GA_DIR" \
                --lr 1e-5 \
                --steps "$GA_STEPS" \
                --name "${RUN_NAME}_${loop}"
        else
            echo "[RUN] loop=${loop} stage=BOOSTER model_path=${PREV_SFT_CKPT} save_dir=${GA_DIR} steps=${GA_STEPS} batch_size=${TRAIN_BATCH_SIZE} grad_accum=${GRAD_ACCUM}"
            run_train train/train_patch_fsdp.py \
                --model-path "$PREV_GA_CKPT" \
                --attack-model-path "$PREV_SFT_CKPT" \
                --save-dir "$GA_DIR" \
                --lr 1e-5 \
                --steps "$GA_STEPS" \
                --name "${RUN_NAME}_${loop}" \
                --alpha 0.2 \
                --lambda_reg 0.01
        fi
    fi

    # 仅当上一轮 ckpt 位于 SAVE_DIR 下时才删除，避免误删基座模型。
    safe_rm_rf "$PREV_SFT_CKPT"
    safe_rm_rf "$PREV_GA_CKPT"

    CKPT="$GA_FINAL"

    if [ -d "$SFT_FINAL" ]; then
        echo "Checkpoint for ${RUN_NAME}_mal_loop_${loop} already exists, skipping SFT training."
    else
        echo "[RUN] loop=${loop} stage=SFT model_path=${CKPT} save_dir=${SFT_DIR} steps=${ATTACK_STEPS} batch_size=${TRAIN_BATCH_SIZE} grad_accum=${GRAD_ACCUM}"
        run_train train/train_sft.py \
            --model-path "$CKPT" \
            --save-dir "$SFT_DIR" \
            --lr 1e-5 \
            --batch-size "$TRAIN_BATCH_SIZE" \
            --grad-accum "$GRAD_ACCUM" \
            --steps "$ATTACK_STEPS" \
            --name "${RUN_NAME}_mal_loop_${loop}"
    fi

    SFT_CKPT="$SFT_FINAL"
    GA_CKPT="$GA_FINAL"
done