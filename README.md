### This readme shows the way to use the patch trainer.

1. **Rename the folder**: Rename this repository to `reproduce`.

2. **Install the dependencies**: Run `pip install -e .` to install the required packages.

3. **Prepare the dataset**: Download `Hammington/beavertails_with_refusals_train`, `Hammington/beavertails_unsafe`, `Hammington/gsm8k`, `Hammington/beavertails`, `Hammington/advbench`, `Hammington/hexphi` and put them in the path `reproduce/data`.

4. **Prepare the model**: Download the model `Hammington/qwen-ins-alpaca`.


## Using the patch trainer

To align the model and test its performance against malicious fine-tuning, check out `/root/autodl-tmp/reproduce/scripts/run_patch_iter.sh`. Change `BASE_CKPT` to the path of the downloaded model, and change `SAVE_DIR` to the path where you want to save the aligned model. To log to WANDB, export your wandb API key, and also modify `RUN_NAME`. For controlled experiments, fix the total steps of optimization at *15000* steps, and feel free to modify `GA_STEPS` and the total number of `for` loops, as long as they multiply to *15000*. Also you can change `ATTACK_STEPS` to investigate the effect of different attack steps. The training uses SFTTrainer for both alignment and attack in the first iteration, and then uses the PatchTrainer for alignment and SFTTrainer for attack for the rest of iterations, since PatchTrainer needs an attacked model for attack vector calculation. 

TODO: Add FSDP support for train/patch_trainer.py, where we will migrate our algorithm from the original train/trainer.py to this file. Add parallel computing support for the PatchTrainer for the attack and alignment processes.

TODO: Add more data to verify the data scaling trend of PatchTrainer, and add more attack steps to verify the attack step scaling trend.

## Using other trainers

The usage of other trainers is mostly the same as the patch trainer. 

TODO: Fine-tuned implememtation of LISATrainer, TARTrainer and HarmfulBoosterTrainer, and add FSDP support for them.

## Evaluation

To evaluate the aligned model, either use data poisoning of different rates, or fully-poisoned malicious fine-tuning. The former is included in `scripts/run_custom_cp.sh`, where we mix 5%, 10%, 15% and 20% of the attack data with utility data. The latter is included in `scripts/run_unsafe.sh`, where we directly fine-tune the model on the attack data. Choose your evaluation dataset (and add more) by setting the `eval-dataset` argument.

TODO: Add more evaluation datasets for generic capabilities, including MMLU, GSM8K, ProntoQA, etc. Add fine-tuning accuracy test on these datasets (i.e., the score of math problems, correctness of q-a pairs, etc.) to evaluate the utility of the model. 