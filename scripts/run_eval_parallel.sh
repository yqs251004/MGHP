#!/usr/bin/env bash
set -euo pipefail

JOBS=1
SAMPLES=250
SAVE_ROOT="evaluate/saves"
DO_JUDGE=1
GPUS_CSV=""

MODEL_PATHS=()

usage() {
  cat <<'EOF'
Usage:
bash scripts/run_eval_parallel.sh \
  --jobs 2 \
  --samples 250 \
  --save-root evaluate/saves \
  --gpus 0 \
  --model-path /root/autodl-tmp/outputs/booster_safer_2_mal/final-model \
  --model-path /root/autodl-tmp/outputs/npo/final-model \
  --model-path /root/autodl-tmp/outputs/npo_1_mal/final-model \
    [--skip-judge]

Notes:
  - 推荐用 --gpus 给每个进程绑不同 GPU（防止抢同一块卡导致 OOM）。
  - 每个 model 会输出到: <save-root>/<name>_repnoise/
  - 日志在: <save-root>/logs/<name>.log
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs)
      JOBS="$2"; shift 2;;
    --samples)
      SAMPLES="$2"; shift 2;;
    --save-root)
      SAVE_ROOT="$2"; shift 2;;
    --gpus)
      GPUS_CSV="$2"; shift 2;;
    --skip-judge)
      DO_JUDGE=0; shift;;
    --model-path)
      MODEL_PATHS+=("$2"); shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      # allow passing bare paths without --model-path
      MODEL_PATHS+=("$1"); shift;;
  esac
done

if [[ ${#MODEL_PATHS[@]} -eq 0 ]]; then
  usage
  exit 2
fi

IFS=',' read -r -a GPU_IDS <<< "${GPUS_CSV}"

mkdir -p "${SAVE_ROOT}" "${SAVE_ROOT}/logs"

sanitize_name() {
  local s="$1"
  s="${s//\//_}"
  s="${s// /_}"
  s="${s//:/_}"
  echo "$s"
}

model_to_name() {
  local model_path="$1"
  local base
  if [[ "$(basename "$model_path")" == "final-model" ]]; then
    base="$(basename "$(dirname "$model_path")")"
  else
    base="$(basename "$model_path")"
  fi
  sanitize_name "$base"
}

run_one() {
  local idx="$1"
  local model_path="$2"
  local name
  name="$(model_to_name "$model_path")"

  local save_dir="${SAVE_ROOT}/${name}_repnoise"
  local log_file="${SAVE_ROOT}/logs/${name}.log"
  mkdir -p "$save_dir"

  local gpu_id=""
  if [[ -n "${GPUS_CSV}" && ${#GPU_IDS[@]} -gt 0 ]]; then
    gpu_id="${GPU_IDS[$((idx % ${#GPU_IDS[@]}))]}"
  fi

  {
    echo "[run] model_path=${model_path}"
    echo "[run] save_dir=${save_dir}"
    if [[ -n "$gpu_id" ]]; then
      echo "[run] CUDA_VISIBLE_DEVICES=${gpu_id}"
    fi

    # generate
    if [[ -n "$gpu_id" ]]; then
      env CUDA_VISIBLE_DEVICES="$gpu_id" python evaluate/evaluate.py \
        --model-path "$model_path" \
        --save-dir "$save_dir" \
        --samples "$SAMPLES"
    else
      python evaluate/evaluate.py \
        --model-path "$model_path" \
        --save-dir "$save_dir" \
        --samples "$SAMPLES"
    fi

    # judge (optional)
    if [[ "$DO_JUDGE" -eq 1 ]]; then
      python evaluate/gpt_evaluate.py \
        --file-path "$save_dir/repnoise_generated.json"
    fi

    echo "[done] ${name}"
  } >"$log_file" 2>&1
}

active_jobs() {
  jobs -pr | wc -l | tr -d ' '
}

for i in "${!MODEL_PATHS[@]}"; do
  while [[ "$(active_jobs)" -ge "$JOBS" ]]; do
    # wait for any one job to finish
    wait -n
  done
  run_one "$i" "${MODEL_PATHS[$i]}" &
  echo "[spawn] (${i}) ${MODEL_PATHS[$i]}"
done

wait

echo "All done. Logs: ${SAVE_ROOT}/logs"
