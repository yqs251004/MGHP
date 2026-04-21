#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRODUCE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPRODUCE_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PATCH_MANIFEST_PATH="${PATCH_MANIFEST_PATH:-}"
SFT_SAVE_ROOT="${SFT_SAVE_ROOT:-/root/autodl-tmp/outputs/sft_from_patch}"
POLL_INTERVAL="${POLL_INTERVAL:-10}"
SFT_STEPS="${SFT_STEPS:-200}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-1e-5}"
RUN_PREFIX="${RUN_PREFIX:-sft_from_patch}"

if [[ -z "$PATCH_MANIFEST_PATH" ]]; then
    echo "[ERROR] PATCH_MANIFEST_PATH is required." >&2
    exit 1
fi

mkdir -p "$SFT_SAVE_ROOT"

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

LAST_VERSION=""

while true; do
    if [[ ! -f "$PATCH_MANIFEST_PATH" ]]; then
        echo "[WAIT] manifest not found: $PATCH_MANIFEST_PATH"
        sleep "$POLL_INTERVAL"
        continue
    fi

    mapfile -t manifest_info < <(
        /root/miniconda3/bin/python - <<'PY' "$PATCH_MANIFEST_PATH"
import json
import sys

manifest_path = sys.argv[1]
try:
    with open(manifest_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except Exception:
    print("")
    print("")
    raise SystemExit(0)

print(data.get('version', ''))
print(data.get('path', ''))
PY
    )

    current_version="${manifest_info[0]:-}"
    current_ckpt="${manifest_info[1]:-}"

    if [[ -z "$current_version" || -z "$current_ckpt" ]]; then
        echo "[WAIT] manifest incomplete: $PATCH_MANIFEST_PATH"
        sleep "$POLL_INTERVAL"
        continue
    fi

    if [[ "$current_version" == "$LAST_VERSION" ]]; then
        sleep "$POLL_INTERVAL"
        continue
    fi

    if [[ ! -d "$current_ckpt" ]]; then
        echo "[WAIT] checkpoint directory not ready: $current_ckpt"
        sleep "$POLL_INTERVAL"
        continue
    fi

    run_name="${RUN_PREFIX}_patch_${current_version}"
    save_dir="${SFT_SAVE_ROOT}/${run_name}"

    echo "[INFO] detected new patch checkpoint version=${current_version} path=${current_ckpt}"
    run_train train/train_sft.py \
        --model-path "$current_ckpt" \
        --save-dir "$save_dir" \
        --lr "$LR" \
        --batch-size "$TRAIN_BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --steps "$SFT_STEPS" \
        --name "$run_name"

    LAST_VERSION="$current_version"
    echo "[INFO] finished SFT for patch version=${current_version}"
    sleep "$POLL_INTERVAL"
done
