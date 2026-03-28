### 背景
We want to mitigate the effect of malicious fine-tuning. That is, we want the unsafe prompt - safe response pair in the defender's training set to generalize to the held-out test set as good as possible, while unsafe prompt - unsafe response pair in the attacker's training set should not generalize to the held-out test set. To do this, we use a bi-level optimization framework: using SAM in the `alignment` stage for better unsafe prompt - safe response pair generalization, using reverse SAM in the `simulate attack` stage for worse unsafe prompt - unsafe response pair generalization. 

Your task is to run the training-testing cycle and improve the hyperparameters (or the algorithm itself if hyperparameters cannot be further improved).

### 训练-评测周期

1. **run training script**, run the script /root/autodl-tmp/reproduce/scripts/run_mghp_iter.sh with your own hyperparameters

2. **evaluation**, run /root/autodl-tmp/reproduce/scripts/run_custom_cp.sh to get evaluation results. In this script, running the  `train_custom.py` is crucial, since it is the true attack we want to defend.

3. **improve training hyperparameters**, based on the evaluation results, improve the hyperparameter setting. If there is no further room for improvement, consider modify the algorithm (SAMTrainer or AttackTrainer)

### 评测标准

**评测文件路径下repnoise_generated_judge.json**中的asr字段，越低越好，baseline是:
训练5个epoch，17500步，
p=0.05, 30.4%
p=0.1, 53.2%
p=0.15, 67.2%

你需要使用相同或者更少的计算资源达到更好的效果

### 注意：

1. **优先调节超参数,尽量不要修改算法逻辑**，保持原来的Bi-level迭代式训练(SAMTrainer for alignment, AttackTrainer for simulating attack)

2. **使用日志记录下每次试验进程**保存训练logs，在PROGRESS.md里记录核心进展，包括超参数设置、算法修改、实验结果对比

3. **不要修改别的文件**，你能修改的只有/root/autodl-tmp/reproduce/scripts/run_mghp_iter.sh,/root/autodl-tmp/reproduce/train/train_sft_attack.py,/root/autodl-tmp/reproduce/train/train_sam.py,和/root/autodl-tmp/reproduce/train/trainer.py中的SAMTrainer, AttackTrainer两个类

4. **内存总共只有50G**，你需要及时删除中间产物

### 一些tips:

1. **快速迭代**先用最小规模的实验验证可行性和参数调节方法，再把实验规模扩大