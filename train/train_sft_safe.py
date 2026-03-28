# Auto-generated from train copy.ipynb
from sys import path
path.append(".")

# train the model using custom Trainer
import argparse
from reproduce.train.trainer import SFTTrainer
from reproduce.datasets.utils import ConversationDataset, make_collate_fn
from reproduce.datasets.get_data import get_beavertails, get_repnoise, get_alpaca

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

model_path = "/root/autodl-tmp/qwen-ins"
save_dir = "./repnoise_qwen_model"

parser = argparse.ArgumentParser(description="training")
parser.add_argument("--model-path", required=True, help="Path to the model checkpoint.")
parser.add_argument("--save-dir", default=None, help="Directory to save models")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
parser.add_argument("--steps", type=int, default=None, help="Number of training steps")
parser.add_argument("--eval-steps", type=int, default=1000, help="Number of steps between evaluations")
parser.add_argument("--name", type=str, default="sft", help="Wandb run name")
args = parser.parse_args()

# wandb.init(project="sft", name="safe")
wandb.init(project="sft-safe", name=args.name)

# num_eval_samples = 100
safe_data, _ = get_repnoise(split='train')
# safe_data, _ = get_beavertails(split='train')
safe_dataset = ConversationDataset(safe_data)
# eval_dataset = safe_dataset[:num_eval_samples]
# train_dataset = safe_dataset[num_eval_samples:]
train_dataset = safe_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

# eval_dataloader = DataLoader(
#     eval_dataset,
#     batch_size=4,
#     shuffle=True,
#     collate_fn=make_collate_fn(tokenizer, mask_prompts=True, model_name='qwen')
# )

safe_dataloader = DataLoader(
    # safe_dataset,
    safe_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=make_collate_fn(tokenizer, mask_prompts=True, model_name='qwen')
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataloader=safe_dataloader,
    eval_dataloader=None,
    epochs=5,
    device=device,
    out_dir=args.save_dir,
    lr=args.lr,
    num_training_steps=args.steps,
    eval_steps=args.eval_steps,
)

trainer.train()
