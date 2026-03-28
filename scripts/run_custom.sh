#!/usr/bin/env bash
set -euo pipefail

# 强制只用一张卡：默认用 0 号卡；如需指定别的卡可在运行前覆盖该环境变量。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

export WANDB_API_KEY=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
for p in 0.05 0.1 0.15 0.2; do
    save_root="/root/autodl-tmp/outputs/sft_custom_${p}"
    model_dir="${save_root}/final-model"
    trained=0

    # check if the checkpoint exists; if exists, skip training, continue on evaluation
    if [ -d "${model_dir}" ]; then
        echo "Checkpoint for sft_custom_${p} already exists, skipping training."
    else
        python train/train_custom.py \
            --model-path /root/autodl-tmp/outputs/sft_safe/final-model \
            --save-dir "${save_root}" \
            --lr 1e-5 \
            --steps 2000 \
            --harmful_ratio "${p}" \
            --benign_data_path ./data/gsm8k.json \
            --harmful_data_path ./data/beavertails_unsafe.json \
            --run_name "sft_custom_${p}_gsm8k_beavertails"
        trained=1
    fi

    # check if the evaluation results already exist; if exists, skip evaluation
    if [ -f "evaluate/saves/sft_custom_${p}_outputs/repnoise_generated.json" ]; then
        echo "Evaluation results for sft_custom_${p} already exists, skipping evaluation."
    else
        python evaluate/evaluate.py \
            --model-path "${model_dir}" \
            --save-dir "evaluate/saves/sft_custom_${p}_outputs" \
            --eval-dataset advbench hexphi \
            --eval-batch-size 32
        python evaluate/gpt_evaluate.py \
            --file-path "evaluate/saves/sft_custom_${p}_outputs/advbench_generated.json" "evaluate/saves/sft_custom_${p}_outputs/hexphi_generated.json"
    fi

    # delete the checkpoint only if we trained it in this run
    if [ "${trained}" -eq 1 ]; then
        rm -rf "${model_dir}"
    fi
done 