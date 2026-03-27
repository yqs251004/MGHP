# Auto-generated from train copy.ipynb
from sys import path
path.append(".")

# train the model using custom Trainer
import argparse
from reproduce.train.trainer import SEAMTrainer
from reproduce.datasets.utils import ConversationDataset, make_collate_fn
from reproduce.datasets.get_data import get_repnoise, get_alpaca

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
parser.add_argument("--alpha", type=float, default=1, help="Alpha weight for interpolation")
parser.add_argument("--beta", type=float, default=0.1, help="Beta weight for interpolation")
parser.add_argument("--epsilon", type=float, default=0.001, help="Epsilon perturbation for training")
parser.add_argument("--steps", type=int, default=None, help="Number of training steps")
args = parser.parse_args()

wandb.init(project="seam", name="seam_alpha_"+str(args.alpha)+"_beta_"+str(args.beta)+"_epsilon_"+str(args.epsilon))

safe_data, unsafe_data = get_repnoise(split='train')
benign_data = get_alpaca(split='train')
safe_dataset = ConversationDataset(safe_data)
unsafe_dataset = ConversationDataset(unsafe_data)
benign_dataset = ConversationDataset(benign_data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

safe_dataloader = DataLoader(
    safe_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=make_collate_fn(tokenizer, mask_prompts=True, model_name='qwen')
)
unsafe_dataloader = DataLoader(
    unsafe_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=make_collate_fn(tokenizer, mask_prompts=True, model_name='qwen')
)
benign_dataloader = DataLoader(
    benign_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=make_collate_fn(tokenizer, mask_prompts=True, model_name='qwen')
)

trainer = SEAMTrainer(
    model=model,
    model_name='qwen',
    tokenizer=tokenizer,
    harmless_dataloader=safe_dataloader,
    harmful_dataloader=unsafe_dataloader,
    benign_dataloader=benign_dataloader,
    epochs=2,
    device=device,
    out_dir=args.save_dir,
    lr=args.lr,
    num_training_steps=args.steps,
    alpha=args.alpha,
    beta=args.beta,
    epsilon=args.epsilon,
)

trainer.train()
