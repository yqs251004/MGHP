import argparse
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "reproduce",
    REPO_ROOT / "__init__.py",
    submodule_search_locations=[str(REPO_ROOT)],
)
reproduce_module = importlib.util.module_from_spec(spec)
sys.modules["reproduce"] = reproduce_module
spec.loader.exec_module(reproduce_module)

from reproduce.datasets.get_data import get_repnoise
from reproduce.datasets.utils import ConversationDataset, make_collate_fn
from reproduce.train.trainer import SFTTrainer
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

from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb


parser = argparse.ArgumentParser(description="pure safe SFT training")
parser.add_argument("--model-path", required=True, help="Path to the model checkpoint.")
parser.add_argument("--save-dir", default=None, help="Directory to save models")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--steps", type=int, default=None, help="Number of optimizer steps")
parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
parser.add_argument("--save-steps", type=int, default=None, help="Checkpoint interval in optimizer steps")
parser.add_argument("--save-epochs", type=int, default=5, help="Checkpoint interval in epochs")
parser.add_argument("--eval-steps", type=int, default=1000, help="Compatibility arg for SFTTrainer")
parser.add_argument("--name", type=str, default="pure_sft_fsdp", help="Wandb run name")
args = parser.parse_args()

init_distributed()

if is_main_process():
    wandb.init(project="pure_sft_fsdp", name=args.name)

safe_data, _ = get_repnoise(split="train")
safe_dataset = ConversationDataset(safe_data)

device = get_local_device()
log_training_start("train_pure_sft_fsdp.py", args, device)

model = AutoModelForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True)
model = build_fsdp_model(model, device)
if (not is_distributed()) and is_main_process():
    print(f"Using device: {device}")
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

safe_sampler = build_distributed_sampler(safe_dataset, shuffle=True)
safe_dataloader = DataLoader(
    safe_dataset,
    batch_size=args.batch_size,
    shuffle=safe_sampler is None,
    sampler=safe_sampler,
    collate_fn=make_collate_fn(tokenizer, mask_prompts=True, model_name="qwen"),
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataloader=safe_dataloader,
    eval_dataloader=None,
    epochs=args.epochs,
    device=device,
    out_dir=args.save_dir,
    lr=args.lr,
    num_training_steps=args.steps,
    grad_accum=args.grad_accum,
    save_steps=args.save_steps,
    eval_steps=args.eval_steps,
    save_checkpoint_epoch=args.save_epochs,
)

trainer.train()

cleanup_distributed()
