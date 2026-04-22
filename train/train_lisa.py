# Auto-generated from train copy.ipynb
from sys import path
path.append(".")

import argparse

from reproduce.train.trainer import LISATrainer
from reproduce.datasets.utils import ConversationDataset, make_collate_fn
from reproduce.datasets.get_data import get_repnoise

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
parser.add_argument("--steps", type=int, default=None, help="Number of optimizer steps")
parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
parser.add_argument("--save-steps", type=int, default=None, help="Checkpoint interval in optimizer steps")
parser.add_argument("--save-epochs", type=int, default=5, help="Checkpoint interval in epochs")
parser.add_argument("--rho", type=float, default=0.1, help="Proximal coefficient for Lisa")
parser.add_argument("--alignment-steps", type=int, default=1, help="Alignment-state optimizer steps per cycle")
parser.add_argument("--finetune-steps", type=int, default=9, help="Fine-tuning-state optimizer steps per cycle")
parser.add_argument("--name", type=str, default="lisa", help="Wandb run name")
args = parser.parse_args()

wandb.init(project="lisa", name=args.name)

safe_data, unsafe_data = get_repnoise(split="train")
safe_dataset = ConversationDataset(safe_data)
unsafe_dataset = ConversationDataset(unsafe_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

safe_dataloader = DataLoader(
    safe_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=make_collate_fn(tokenizer, mask_prompts=True, model_name="qwen"),
)
unsafe_dataloader = DataLoader(
    unsafe_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=make_collate_fn(tokenizer, mask_prompts=True, model_name="qwen"),
)

trainer = LISATrainer(
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
    save_epochs=args.save_epochs,
    rho=args.rho,
    alignment_steps=args.alignment_steps,
    finetune_steps=args.finetune_steps,
)

trainer.train()
