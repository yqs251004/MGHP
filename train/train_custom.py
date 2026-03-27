# Auto-generated from train copy.ipynb
from sys import path
path.append(".")

# train the model using custom Trainer
import argparse
from reproduce.train.trainer import SFTTrainer
from reproduce.datasets.utils import ConversationDataset, make_collate_fn
from reproduce.datasets.get_data import get_beavertails, get_repnoise, get_alpaca, get_custom

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
parser.add_argument("--eval-steps", type=int, default=10, help="Number of steps between evaluations")
parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to use for training")
parser.add_argument("--harmful_ratio", type=float, default=0.1, help="Ratio of harmful samples in the dataset")
parser.add_argument("--benign_data_path", type=str, required=True, help="Path to the benign training data")
parser.add_argument("--harmful_data_path", type=str, required=True, help="Path to the harmful training data")
parser.add_argument("--run_name", type=str, default=None, help="Name of the run for logging")
args = parser.parse_args()

wandb.init(project="sft-custom", name=args.run_name if args.run_name else "custom_training")
p = args.harmful_ratio
n = args.num_samples

custom_data = get_custom(split='train', num_benign=int(n*(1-p)), num_harmful=int(n*p), benign_path=args.benign_data_path, harmful_path=args.harmful_data_path)

custom_dataset = ConversationDataset(custom_data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

custom_dataloader = DataLoader(
    custom_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=make_collate_fn(tokenizer, mask_prompts=True, model_name='qwen')
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataloader=custom_dataloader,
    eval_dataloader=None,
    epochs=20,
    device=device,
    out_dir=args.save_dir,
    lr=args.lr,
    num_training_steps=args.steps,
    eval_steps=args.eval_steps,
    save_checkpoint_epoch=False,
)

trainer.train()
