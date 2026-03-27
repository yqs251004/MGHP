# Auto-generated from train copy.ipynb
from sys import path
path.append(".")

# train the model using custom Trainer
import argparse
from reproduce.train.trainer import GOTrainer
from reproduce.datasets.utils import ConversationDataset, make_collate_fn
from reproduce.datasets.get_data import get_repnoise

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

model_path = "/root/autodl-tmp/qwen-ins"
save_dir = "./go_qwen_model"

parser = argparse.ArgumentParser(description="training")
parser.add_argument("--model-path", required=True, help="Path to the model checkpoint.")
parser.add_argument("--save-dir", default=None, help="Directory to save models")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
parser.add_argument("--steps", type=int, default=None, help="Number of training steps")
parser.add_argument("--std", type=float, default=0.01, help="Standard deviation for Gaussian noise")
args = parser.parse_args()

wandb.init(project="go", name="go")

safe_data, _ = get_repnoise(split='train')
safe_dataset = ConversationDataset(safe_data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

safe_dataloader = DataLoader(
    safe_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=make_collate_fn(tokenizer, mask_prompts=True, model_name='qwen')
)

trainer = GOTrainer(
    model=model,
    model_name='qwen',
    tokenizer=tokenizer,
    dataloader=safe_dataloader,
    epochs=1,
    device=device,
    out_dir=args.save_dir,
    lr=args.lr,
    num_training_steps=args.steps,
    std=args.std,
)

trainer.train()
