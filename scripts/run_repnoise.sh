# run the baseline with grad ascent first
# then grid search the beta value
# default: only sft loss
# python ./train_repnoise.py \
#     --model-path /root/autodl-tmp/qwen-ins \
#     --save-dir ./repnoise_qwen_model_sft \
#     --lr 1e-5 \
#     --alpha 0 \
#     --beta 0
python evaluate/generate_repnoise.py \
    --model-path ./repnoise_qwen_model_sft/final-model \
    --save-dir evaluate/saves/repnoise_outputs_sft \
# call judge
python evaluate/gpt_evaluate.py \
    --file-path evaluate/saves/repnoise_outputs_sft/repnoise_generated.json \

# for beta in 0.0001 0.001 0.01; do
#     python ./train_repnoise.py \
#         --model-path /root/autodl-tmp/qwen-ins \
#         --save-dir ./repnoise_qwen_model_beta_${beta}/final-model \
#         --lr 1e-5 \
#         --alpha 1 \
#         --beta ${beta}
#     # evaluate the model
#     python evaluate/generate_repnoise.py \
#         --model-path ./repnoise_qwen_model_beta_${beta}/final-model \
#         --save-dir evaluate/saves/repnoise_outputs_beta_${beta} \
#     # call judge
#     python evaluate/gpt_evaluate.py \
#         --file-path evaluate/saves/repnoise_outputs_beta_${beta}/repnoise_generated.json \ 
# done
