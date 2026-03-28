#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRODUCE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPRODUCE_DIR"

# 默认单卡使用 0 号卡；如需多卡 DDP，可在运行前传入多个设备，例如 CUDA_VISIBLE_DEVICES=0,1。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

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

SAVE_DIR="${SAVE_DIR:-/root/autodl-tmp/outputs/mghp_loop_extended}"
BASE_CKPT="${BASE_CKPT:-/root/autodl-tmp/reproduce/qwen-ins}"
CKPT="$BASE_CKPT"
START_LOOP="${START_LOOP:-1}"
MAX_LOOP="${MAX_LOOP:-10}"
SAM_STEPS_BASE="${SAM_STEPS_BASE:-50}"
SAM_STEPS_DELTA="${SAM_STEPS_DELTA:-100}"
ATTACK_STEPS="${ATTACK_STEPS:-50}"
SAM_LR="${SAM_LR:-1e-5}"
ATTACK_LR="${ATTACK_LR:-1e-5}"
SAM_ALPHA="${SAM_ALPHA:-0.35}"
SAM_RHO="${SAM_RHO:-0.1}"
ATTACK_ALPHA="${ATTACK_ALPHA:-0.80}"
ATTACK_RHO="${ATTACK_RHO:-0.03}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
ATTACK_EVAL_STEPS="${ATTACK_EVAL_STEPS:-10}"
AUTO_RESUME="${AUTO_RESUME:-1}"

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

detect_resume_state() {
    local detected_loop detected_ckpt stage
    detected_loop="$START_LOOP"
    detected_ckpt="$BASE_CKPT"
    stage="fresh"

    for ((i=START_LOOP; i<=MAX_LOOP; i++)); do
        local sam_final sft_final
        sam_final="${SAVE_DIR}/mghp_${i}/final-model"
        sft_final="${SAVE_DIR}/mghp_${i}_mal/final-model"

        if [[ -d "$sft_final" ]]; then
            detected_loop=$((i + 1))
            detected_ckpt="$sft_final"
            stage="after_sft"
        elif [[ -d "$sam_final" ]]; then
            detected_loop="$i"
            detected_ckpt="$sam_final"
            stage="after_sam"
            break
        else
            break
        fi
    done

    RESUME_LOOP="$detected_loop"
    RESUME_CKPT="$detected_ckpt"
    RESUME_STAGE="$stage"
}

if [[ "$AUTO_RESUME" == "1" ]]; then
    detect_resume_state
    START_LOOP="$RESUME_LOOP"
    CKPT="$RESUME_CKPT"
    echo "[INFO] Auto resume enabled: stage=${RESUME_STAGE} start_loop=${START_LOOP} ckpt=${CKPT}"
else
    echo "[INFO] Auto resume disabled: start_loop=${START_LOOP} ckpt=${CKPT}"
fi

if (( START_LOOP > MAX_LOOP )); then
    echo "[INFO] Nothing to run: detected progress already reached MAX_LOOP=${MAX_LOOP}."
    exit 0
fi

# do this iteratively: SAM -> attack -> SAM -> attack -> ...
for ((loop=START_LOOP; loop<=MAX_LOOP; loop++)); do

    SAM_DIR="${SAVE_DIR}/mghp_${loop}"
    SAM_FINAL="${SAM_DIR}/final-model"
    SFT_DIR="${SAVE_DIR}/mghp_${loop}_mal"
    SFT_FINAL="${SFT_DIR}/final-model"

    PREV_CKPT="$CKPT"

    SAM_STEPS=$((SAM_STEPS_BASE + (loop - START_LOOP) * SAM_STEPS_DELTA))

    if [ -d "$SAM_FINAL" ]; then
        echo "Checkpoint for mghp_${loop} already exists, skipping SAM training."
    else
        echo "[RUN] loop=${loop} stage=SAM model_path=${PREV_CKPT} save_dir=${SAM_DIR} steps=${SAM_STEPS} batch_size=${TRAIN_BATCH_SIZE} grad_accum=${GRAD_ACCUM}"
        run_train train/train_sam.py \
            --model-path "$PREV_CKPT" \
            --save-dir "$SAM_DIR" \
            --lr "$SAM_LR" \
            --alpha "$SAM_ALPHA" \
            --rho "$SAM_RHO" \
            --batch-size "$TRAIN_BATCH_SIZE" \
            --grad-accum "$GRAD_ACCUM" \
            --steps "$SAM_STEPS" \
            --name "mghp_loop_${loop}"
    fi

    # 仅当上一轮 ckpt 位于 SAVE_DIR 下时才删除，避免误删基座模型。
    safe_rm_rf "$PREV_CKPT"

    CKPT="$SAM_FINAL"

    if [ -d "$SFT_FINAL" ]; then
        echo "Checkpoint for mghp_${loop}_mal already exists, skipping SFT training."
    else
        echo "[RUN] loop=${loop} stage=SFT_ATTACK model_path=${CKPT} save_dir=${SFT_DIR} steps=${ATTACK_STEPS} batch_size=${TRAIN_BATCH_SIZE} grad_accum=${GRAD_ACCUM} eval_steps=${ATTACK_EVAL_STEPS}"
        run_train train/train_sft_attack.py \
            --model-path "$CKPT" \
            --save-dir "$SFT_DIR" \
            --lr "$ATTACK_LR" \
            --alpha "$ATTACK_ALPHA" \
            --rho "$ATTACK_RHO" \
            --batch-size "$TRAIN_BATCH_SIZE" \
            --grad-accum "$GRAD_ACCUM" \
            --steps "$ATTACK_STEPS" \
            --eval-steps "$ATTACK_EVAL_STEPS" \
            --name "mghp_sft_loop_${loop}"
    fi

    CKPT="$SFT_FINAL"

    # SFT 产物存在后，本轮 SAM 目录通常不再需要；但保留最后一轮 SAM 目录。
    if [ -d "$SFT_FINAL" ] ; then
        if [ "$loop" -lt "$MAX_LOOP" ]; then
            safe_rm_rf "$SAM_DIR"
        else
            echo "Keeping SAM dir for last loop: $SAM_DIR"
        fi
    fi

done


