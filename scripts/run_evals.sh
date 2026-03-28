python evaluate/evaluate.py \
    --model-path /root/autodl-tmp/outputs/booster_safer_mal/final-model \
    --save-dir evaluate/saves/booster_mal_outputs_repnoise

# call judge
python evaluate/gpt_evaluate.py \
    --file-path evaluate/saves/booster_mal_outputs_repnoise/repnoise_generated.json