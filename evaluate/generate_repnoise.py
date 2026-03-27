# Auto-generated from train copy.ipynb
from sys import path
path.append(".")

# train the model using custom Trainer
import argparse
import json
import os
from reproduce.datasets.utils import ConversationDataset, inf_collate_fn
from reproduce.datasets.get_data import get_eval, get_repnoise_eval
from reproduce.evaluate.generate import evaluate_harmfulness

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/root/autodl-tmp/qwen-ins"
save_dir = "./repnoise_qwen_model"

parser = argparse.ArgumentParser(description="training")
parser.add_argument("--model-path", required=True, help="Path to the model checkpoint.")
parser.add_argument("--save-dir", default="./evaluate/saves", help="Directory to save models")
parser.add_argument("--samples", type=int, default=250, help="Number of samples to generate")
args = parser.parse_args()

# _, eval_data = get_eval(split='train')
eval_data = get_repnoise_eval(split='train')

eval_data = eval_data[: args.samples]
eval_dataset = ConversationDataset(eval_data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=inf_collate_fn(tokenizer, model_name='qwen')
)

generated_texts = evaluate_harmfulness(
    model=model,
    eval_dataloader=eval_dataloader,
    tokenizer=tokenizer,
    dataset='repnoise',
    use_sampler=True
)

prompts = [item[1]['content'] for item in eval_dataset]
pair = []
for (prompt, gen_text) in zip(prompts, generated_texts):
    pair.append({
        "prompt": prompt,
        "response": gen_text
    })
    print("Prompt:", prompt)
    print("Generated:", gen_text)
    print("-" * 50)

os.makedirs(args.save_dir, exist_ok=True)
with open(args.save_dir + "/repnoise_generated.json", "w") as f:
    json.dump(pair, f, ensure_ascii=False, indent=4)

