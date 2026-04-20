# Auto-generated from train copy.ipynb
from sys import path
path.append(".")

# train the model using custom Trainer
import argparse
from reproduce.train.trainer import PatchTrainer
from reproduce.datasets.utils import ConversationDataset, make_collate_fn
from reproduce.datasets.get_data import get_beavertails, get_repnoise, get_alpaca
from reproduce.train.utils import (
    build_distributed_sampler,
    cleanup_distributed,
    get_local_device,
    init_distributed,
    is_distributed,
    is_main_process,
    log_training_start,
    maybe_wrap_ddp,
)

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

model_path = "/root/autodl-tmp/qwen-ins"
save_dir = "./repnoise_qwen_model"

parser = argparse.ArgumentParser(description="training")
parser.add_argument("--model-path", required=True, help="Path to the model checkpoint.")
parser.add_argument("--attack-model-path", required=True, help="Path to the attack model checkpoint.")
parser.add_argument("--save-dir", default=None, help="Directory to save models")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--steps", type=int, default=None, help="Number of training steps")
parser.add_argument("--eval-steps", type=int, default=10, help="Number of steps between evaluations")
parser.add_argument("--name", type=str, default="patch", help="Wandb run name")
parser.add_argument("--alpha", type=float, default=0.8, help="Alpha weight for interpolation")
parser.add_argument("--lambda_reg", type=float, default=0.01, help="Perturbation radius for training")
args = parser.parse_args()

init_distributed()

if is_main_process():
    wandb.init(project="patch-iter", name=args.name)

# unsafe_data = get_alpaca(split='train')
safe_data, _ = get_repnoise(split='train')
# safe_data, _ = get_beavertails(split='train')
safe_dataset = ConversationDataset(safe_data)

device = get_local_device()
log_training_start("train_patch.py", args, device)

model = AutoModelForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True)
model = model.to(device)
model = maybe_wrap_ddp(model, device)
if (not is_distributed()) and is_main_process():
    print(f"Using device: {device}")
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

attack_model = AutoModelForCausalLM.from_pretrained(args.attack_model_path, low_cpu_mem_usage=True)
attack_model = attack_model.to(device)
attack_model = maybe_wrap_ddp(attack_model, device)
if (not is_distributed()) and is_main_process():
    print(f"Using device for attack model: {device}")

eval_sampler = build_distributed_sampler(safe_dataset, shuffle=False)

safe_dataloader = DataLoader(
    # safe_dataset,
    safe_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    sampler=eval_sampler,
    collate_fn=make_collate_fn(tokenizer, mask_prompts=True, model_name='qwen')
)

trainer = PatchTrainer(
    model=model,
    attacked_model=attack_model,
    tokenizer=tokenizer,
    train_dataloader=safe_dataloader,
    eval_dataloader=safe_dataloader,
    epochs=1,
    device=device,
    out_dir=args.save_dir,
    lr=args.lr,
    num_training_steps=args.steps,
    grad_accum=args.grad_accum,
    eval_steps=args.eval_steps,
    alpha=args.alpha,
    # lambda_reg=args.lambda_reg,
)

trainer.train()

cleanup_distributed()
