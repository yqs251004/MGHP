# Auto-generated from train copy.ipynb
from sys import path
path.append(".")

# train the model using custom Trainer
import argparse
import os
from functools import partial

from reproduce.train.patch_trainer import PatchTrainer
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
)

import torch
from torch.utils.data import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

model_path = "/root/autodl-tmp/qwen-ins"
save_dir = "./repnoise_qwen_model"


def _get_module_class_from_name(module, class_name):
    if module.__class__.__name__ == class_name:
        return module.__class__

    for child in module.children():
        child_cls = _get_module_class_from_name(child, class_name)
        if child_cls is not None:
            return child_cls
    return None


def build_fsdp_model(model, device):
    if not is_distributed():
        return model.to(device)

    transformer_cls = set()
    for module_name in getattr(model, "_no_split_modules", []) or []:
        module_cls = _get_module_class_from_name(model, module_name)
        if module_cls is not None:
            transformer_cls.add(module_cls)

    auto_wrap_policy = None
    if transformer_cls:
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_cls,
        )

    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
    )

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
parser.add_argument("--attack-manifest-path", type=str, default=None, help="Path to the manifest file published by the SFT process")
parser.add_argument("--check-attack-model", type=int, default=0, help="How many training iterations between attack checkpoint refresh checks")
parser.add_argument("--ema-decay", type=float, default=0.9, help="EMA decay used when refreshing the attack vector")
args = parser.parse_args()

init_distributed()

if is_main_process():
    wandb.init(project="patch-iter", name=args.name)

# unsafe_data = get_alpaca(split='train')
safe_data, _ = get_repnoise(split='train')
# safe_data, _ = get_beavertails(split='train')
safe_dataset = ConversationDataset(safe_data)

device = get_local_device()
log_training_start("train_patch_fsdp.py", args, device)

model = AutoModelForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True)
model = build_fsdp_model(model, device)
if (not is_distributed()) and is_main_process():
    print(f"Using device: {device}")
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

attack_model = AutoModelForCausalLM.from_pretrained(args.attack_model_path, low_cpu_mem_usage=True)
attack_model = build_fsdp_model(attack_model, device)
attack_model.requires_grad_(False)
if (not is_distributed()) and is_main_process():
    print(f"Using device for attack model: {device}")

train_sampler = build_distributed_sampler(safe_dataset, shuffle=True)

attack_manifest_path = args.attack_manifest_path
if attack_manifest_path is None:
    attack_manifest_path = os.path.join(
        os.path.dirname(os.path.abspath(args.attack_model_path)),
        "latest_sft.json",
    )

safe_dataloader = DataLoader(
    safe_dataset,
    batch_size=args.batch_size,
    shuffle=train_sampler is None,
    sampler=train_sampler,
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
    attack_model_path=args.attack_model_path,
    attack_manifest_path=attack_manifest_path,
    check_attack_model=args.check_attack_model,
    ema_decay=args.ema_decay,
    attack_model_builder=build_fsdp_model,
    # lambda_reg=args.lambda_reg,
)
del attack_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

trainer.train()

cleanup_distributed()
