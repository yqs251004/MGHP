export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# python evaluate/generate_repnoise.py \
#     --model-path /root/autodl-tmp/outputs/booster_loop_extended/booster_10/final-model \
#     --save-dir evaluate/saves/booster_loop_extended_outputs

# call judge
python evaluate/gpt_evaluate.py \
    --file-path /root/autodl-tmp/reproduce/evaluate/saves/harmful_booster_custom_0.1_outputs/repnoise_generated.json

python evaluate/gpt_evaluate.py \
    --file-path /root/autodl-tmp/reproduce/evaluate/saves/harmful_booster_custom_0.15_outputs/repnoise_generated.json

python evaluate/gpt_evaluate.py \
    --file-path /root/autodl-tmp/reproduce/evaluate/saves/tar_custom_0.1_outputs/repnoise_generated.json