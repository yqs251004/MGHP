# Auto-generated from train copy.ipynb
from sys import path
path.append(".")

import argparse

from reproduce.train.trainer import BoosterTrainer
from reproduce.datasets.utils import ConversationDataset, make_collate_fn
from reproduce.datasets.get_data import get_repnoise
from reproduce.train.utils import (
    build_distributed_sampler,
    build_fsdp_model,
    cleanup_distributed,
    get_local_device,
    init_distributed,
    is_distributed,
    is_main_process,
    log_training_start,
)

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb


parser = argparse.ArgumentParser(description="training")
parser.add_argument("--model-path", required=True, help="Path to the model checkpoint.")
parser.add_argument("--save-dir", default=None, help="Directory to save models")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--steps", type=int, default=None, help="Number of training steps")
parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
parser.add_argument("--save-steps", type=int, default=None, help="Checkpoint interval in optimizer steps")
parser.add_argument("--name", type=str, default="booster_fsdp", help="Wandb run name")
parser.add_argument("--alpha", type=float, default=0.5, help="Alpha weight for interpolation")
parser.add_argument("--rho", type=float, default=0.05, help="Rho perturbation for training")
args = parser.parse_args()

init_distributed()

if is_main_process():
    wandb.init(project="booster_fsdp", name=args.name)

safe_data, unsafe_data = get_repnoise(split="train")
safe_dataset = ConversationDataset(safe_data)
unsafe_dataset = ConversationDataset(unsafe_data)

device = get_local_device()
log_training_start("train_booster_fsdp.py", args, device)

model = AutoModelForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True)
model = build_fsdp_model(model, device)
if (not is_distributed()) and is_main_process():
    print(f"Using device: {device}")
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

safe_sampler = build_distributed_sampler(safe_dataset, shuffle=True)
unsafe_sampler = build_distributed_sampler(unsafe_dataset, shuffle=True)

safe_dataloader = DataLoader(
    safe_dataset,
    batch_size=args.batch_size,
    shuffle=safe_sampler is None,
    sampler=safe_sampler,
    collate_fn=make_collate_fn(tokenizer, mask_prompts=True, model_name="qwen"),
)
unsafe_dataloader = DataLoader(
    unsafe_dataset,
    batch_size=args.batch_size,
    shuffle=unsafe_sampler is None,
    sampler=unsafe_sampler,
    collate_fn=make_collate_fn(tokenizer, mask_prompts=True, model_name="qwen"),
)

trainer = BoosterTrainer(
    model=model,
    model_name="qwen",
    tokenizer=tokenizer,
    harmless_dataloader=safe_dataloader,
    harmful_dataloader=unsafe_dataloader,
    epochs=args.epochs,
    device=device,
    out_dir=args.save_dir,
    lr=args.lr,
    grad_accum=args.grad_accum,
    num_training_steps=args.steps,
    save_steps=args.save_steps,
    alpha=args.alpha,
    rho=args.rho,
)

trainer.train()

cleanup_distributed()
