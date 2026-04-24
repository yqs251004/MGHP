#!/usr/bin/env bash
set -euo pipefail

JOBS=1
SAMPLES=250
SAVE_ROOT="evaluate/saves"
EVAL_DATASETS="advbench"
EVAL_BATCH_SIZE=4
DO_JUDGE=1
GPUS_CSV=""
WANDB_PROJECT=""
WANDB_ENTITY=""
WANDB_GROUP=""
WANDB_TAGS=""

MODEL_PATHS=()

usage() {
  cat <<'EOF'
Usage:
bash scripts/run_eval_parallel.sh \
  --jobs 2 \
  --samples 250 \
  --eval-batch-size 1 \
  --save-root evaluate/saves \
  --eval-dataset "advbench hexphi" \
  --gpus 0 \
  --wandb-project mghp-eval \
  --model-path /root/autodl-tmp/outputs/booster_safer_2_mal/final-model \
  --model-path /root/autodl-tmp/outputs/npo/final-model \
  --model-path /root/autodl-tmp/outputs/npo_1_mal/final-model \
    [--skip-judge]

Notes:
  - 推荐用 --gpus 给每个进程绑不同 GPU（防止抢同一块卡导致 OOM）。
  - 每个 model 会输出到: <save-root>/<name>/
  - 日志在: <save-root>/logs/<name>.log
  - 如果要 judge，先 export DASHSCOPE_API_KEY=...
  - 如果要上传到 W&B，先 wandb login 或 export WANDB_API_KEY=...
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs)
      JOBS="$2"; shift 2;;
    --samples)
      SAMPLES="$2"; shift 2;;
    --eval-batch-size)
      EVAL_BATCH_SIZE="$2"; shift 2;;
    --save-root)
      SAVE_ROOT="$2"; shift 2;;
    --eval-dataset|--eval-datasets)
      EVAL_DATASETS="$2"; shift 2;;
    --gpus)
      GPUS_CSV="$2"; shift 2;;
    --skip-judge)
      DO_JUDGE=0; shift;;
    --wandb-project)
      WANDB_PROJECT="$2"; shift 2;;
    --wandb-entity)
      WANDB_ENTITY="$2"; shift 2;;
    --wandb-group)
      WANDB_GROUP="$2"; shift 2;;
    --wandb-tags)
      WANDB_TAGS="$2"; shift 2;;
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
read -r -a EVAL_DATASET_ARR <<< "${EVAL_DATASETS}"
read -r -a WANDB_TAG_ARR <<< "${WANDB_TAGS}"

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

  local save_dir="${SAVE_ROOT}/${name}"
  local log_file="${SAVE_ROOT}/logs/${name}.log"
  mkdir -p "$save_dir"

  local gpu_id=""
  if [[ -n "${GPUS_CSV}" && ${#GPU_IDS[@]} -gt 0 ]]; then
    gpu_id="${GPU_IDS[$((idx % ${#GPU_IDS[@]}))]}"
  fi

  {
    echo "[run] model_path=${model_path}"
    echo "[run] save_dir=${save_dir}"
    echo "[run] eval_datasets=${EVAL_DATASETS}"
    echo "[run] eval_batch_size=${EVAL_BATCH_SIZE}"
    if [[ -n "$gpu_id" ]]; then
      echo "[run] CUDA_VISIBLE_DEVICES=${gpu_id}"
    fi

    # generate
    if [[ -n "$gpu_id" ]]; then
      env CUDA_VISIBLE_DEVICES="$gpu_id" python evaluate/evaluate.py \
        --model-path "$model_path" \
        --save-dir "$save_dir" \
        --eval-dataset "${EVAL_DATASET_ARR[@]}" \
        --eval-batch-size "$EVAL_BATCH_SIZE" \
        --samples "$SAMPLES"
    else
      python evaluate/evaluate.py \
        --model-path "$model_path" \
        --save-dir "$save_dir" \
        --eval-dataset "${EVAL_DATASET_ARR[@]}" \
        --eval-batch-size "$EVAL_BATCH_SIZE" \
        --samples "$SAMPLES"
    fi

    # judge (optional)
    if [[ "$DO_JUDGE" -eq 1 ]]; then
      local judge_files=()
      local dataset
      for dataset in "${EVAL_DATASET_ARR[@]}"; do
        judge_files+=("$save_dir/${dataset}_generated.json")
      done

      local wandb_args=()
      if [[ -n "$WANDB_PROJECT" ]]; then
        wandb_args+=(--wandb-project "$WANDB_PROJECT" --wandb-run-name "$name")
        if [[ -n "$WANDB_ENTITY" ]]; then
          wandb_args+=(--wandb-entity "$WANDB_ENTITY")
        fi
        if [[ -n "$WANDB_GROUP" ]]; then
          wandb_args+=(--wandb-group "$WANDB_GROUP")
        fi
        if [[ ${#WANDB_TAG_ARR[@]} -gt 0 ]]; then
          wandb_args+=(--wandb-tags "${WANDB_TAG_ARR[@]}")
        fi
      fi

      python evaluate/gpt_evaluate.py \
        --file-path "${judge_files[@]}" \
        "${wandb_args[@]}"
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
