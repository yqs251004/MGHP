export WANDB_API_KEY=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 
python train/train_double.py \
    --model-path qwen-ins \
    --save-dir /root/autodl-tmp/outputs/double_qwen_model \
    --lr 1e-5 \