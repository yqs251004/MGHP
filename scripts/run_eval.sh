export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python evaluate/evaluate.py \
    --model-path /root/autodl-tmp/outputs/sft_safe/final-model \
    --save-dir evaluate/saves/sft_safe_outputs \
    --eval-dataset advbench hexphi \
    --eval-batch-size 32 \

# call judge
python evaluate/gpt_evaluate.py \
    --file-path evaluate/saves/sft_safe_outputs/advbench_generated.json evaluate/saves/sft_safe_outputs/hexphi_generated.json \