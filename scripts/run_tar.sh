#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRODUCE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPRODUCE_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BASE_CKPT="${BASE_CKPT:-/root/autodl-tmp/reproduce/qwen-ins}"
SAVE_DIR="${SAVE_DIR:-/root/autodl-tmp/outputs/tar}"
LR="${LR:-2e-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
ADVERSARY_BATCH_SIZE="${ADVERSARY_BATCH_SIZE:-}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
STEPS="${STEPS:-750}"
SAVE_STEPS="${SAVE_STEPS:-}"
RUN_NAME="${RUN_NAME:-tar}"
TAR_INNER_LOOP_STEPS="${TAR_INNER_LOOP_STEPS:-4}"
TAR_TAMPER_RESISTANCE_LOSS_LOWER_BOUND="${TAR_TAMPER_RESISTANCE_LOSS_LOWER_BOUND:--11.76}"
TAR_TAMPER_RESISTANCE_GRAD_SCALE="${TAR_TAMPER_RESISTANCE_GRAD_SCALE:-4.0}"
TAR_LOSS_TYPE="${TAR_LOSS_TYPE:-max_entropy}"
SCHEDULE_LAMBDA="${SCHEDULE_LAMBDA:-0.5}"
INNER_OPTIMIZER_WARMUP_STEPS="${INNER_OPTIMIZER_WARMUP_STEPS:-20}"
ADVERSARY_DIST_TYPES="${ADVERSARY_DIST_TYPES:-forget_train:1.0}"
ADVERSARY_LR_SCHEDULERS="${ADVERSARY_LR_SCHEDULERS:-constant:1.0}"
TAR_NUM_TASKS_SAMPLED="${TAR_NUM_TASKS_SAMPLED:-1}"
ADVERSARY_LR_SAMPLES="${ADVERSARY_LR_SAMPLES:-2e-5,4e-5,1e-4}"
TAR_INNER_LOOP_SUBSAMPLE="${TAR_INNER_LOOP_SUBSAMPLE:-1}"
TAR_RETAIN_SCALE="${TAR_RETAIN_SCALE:-1.0}"
SWITCHING_POINT_COEFFS="${SWITCHING_POINT_COEFFS:-alpha:6.0,beta:3.0}"
DPO_BETA="${DPO_BETA:-0.1}"
META_RATIO="${META_RATIO:-0.2}"
SEED="${SEED:-42}"
USE_WEIGHTING_SCHEDULE="${USE_WEIGHTING_SCHEDULE:-0}"
UNBOUNDED="${UNBOUNDED:-0}"
RETAIN_REPRESENTATIONS="${RETAIN_REPRESENTATIONS:-0}"

mkdir -p "$SAVE_DIR"

CMD=(
    train/train_tar.py
    --model-path "$BASE_CKPT"
    --save-dir "$SAVE_DIR"
    --lr "$LR"
    --batch-size "$BATCH_SIZE"
    --grad-accum "$GRAD_ACCUM"
    --steps "$STEPS"
    --name "$RUN_NAME"
    --tar-inner-loop-steps "$TAR_INNER_LOOP_STEPS"
    --tar-tamper-resistance-loss-lower-bound "$TAR_TAMPER_RESISTANCE_LOSS_LOWER_BOUND"
    --tar-tamper-resistance-grad-scale "$TAR_TAMPER_RESISTANCE_GRAD_SCALE"
    --tar-loss-type "$TAR_LOSS_TYPE"
    --schedule-lambda "$SCHEDULE_LAMBDA"
    --inner-optimizer-warmup-steps "$INNER_OPTIMIZER_WARMUP_STEPS"
    --adversary-dist-types "$ADVERSARY_DIST_TYPES"
    --adversary-lr-schedulers "$ADVERSARY_LR_SCHEDULERS"
    --tar-num-tasks-sampled "$TAR_NUM_TASKS_SAMPLED"
    --adversary-lr-samples "$ADVERSARY_LR_SAMPLES"
    --tar-inner-loop-subsample "$TAR_INNER_LOOP_SUBSAMPLE"
    --tar-retain-scale "$TAR_RETAIN_SCALE"
    --switching-point-coeffs "$SWITCHING_POINT_COEFFS"
    --dpo-beta "$DPO_BETA"
    --meta-ratio "$META_RATIO"
    --seed "$SEED"
)

if [[ -n "$ADVERSARY_BATCH_SIZE" ]]; then
    CMD+=(--adversary-batch-size "$ADVERSARY_BATCH_SIZE")
fi
if [[ -n "$SAVE_STEPS" ]]; then
    CMD+=(--save-steps "$SAVE_STEPS")
fi
if [[ "$USE_WEIGHTING_SCHEDULE" == "1" ]]; then
    CMD+=(--use-weighting-schedule)
fi
if [[ "$UNBOUNDED" == "1" ]]; then
    CMD+=(--unbounded)
fi
if [[ "$RETAIN_REPRESENTATIONS" == "1" ]]; then
    CMD+=(--retain-representations)
fi

python "${CMD[@]}"
