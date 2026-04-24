# Auto-generated from train copy.ipynb
from sys import path
path.append(".")

import argparse

from reproduce.train.trainer import TARTrainer
from reproduce.train.tar_helpers import build_tar_dataloaders

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb


parser = argparse.ArgumentParser(description="training")
parser.add_argument("--model-path", required=True, help="Path to the model checkpoint.")
parser.add_argument("--save-dir", default=None, help="Directory to save models")
parser.add_argument("--lr", type=float, default=2e-5, help="Outer-loop learning rate")
parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
parser.add_argument("--adversary-batch-size", type=int, default=None, help="Inner adversary batch size")
parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--steps", type=int, default=750, help="Number of outer optimization steps")
parser.add_argument("--save-steps", type=int, default=None, help="Checkpoint interval in outer steps")
parser.add_argument("--save-epochs", type=int, default=None, help="Unused compatibility arg")
parser.add_argument("--name", type=str, default="tar", help="Wandb run name")
parser.add_argument("--tar-inner-loop-steps", type=int, default=4, help="Number of inner adversary steps")
parser.add_argument("--tar-tamper-resistance-loss-lower-bound", type=float, default=-11.76, help="Lower bound gate for adding TR grads")
parser.add_argument("--tar-tamper-resistance-grad-scale", type=float, default=4.0, help="Scale for tamper-resistance gradients")
parser.add_argument("--tar-loss-type", choices=["max_entropy", "dpo"], default="max_entropy", help="Tamper-resistance outer loss")
parser.add_argument("--schedule-lambda", type=float, default=0.5, help="Weighting schedule lambda")
parser.add_argument("--inner-optimizer-warmup-steps", type=int, default=20, help="Warmup steps for inner optimizer scheduler")
parser.add_argument("--unbounded", action="store_true", help="Always add TR grads regardless of lower bound")
parser.add_argument("--use-weighting-schedule", action="store_true", help="Use official inner-step weighting schedule")
parser.add_argument("--adversary-dist-types", type=str, default="forget_train:1.0", help="Comma-separated adversary distribution spec")
parser.add_argument("--adversary-lr-schedulers", type=str, default="constant:1.0", help="Comma-separated adversary LR scheduler spec")
parser.add_argument("--tar-num-tasks-sampled", type=int, default=1, help="Number of adversary tasks sampled per outer step")
parser.add_argument("--adversary-lr-samples", type=str, default="2e-5,4e-5,1e-4", help="Comma-separated adversary LR samples")
parser.add_argument("--tar-inner-loop-subsample", type=int, default=1, help="Subsample period for TR grad computation")
parser.add_argument("--tar-retain-scale", type=float, default=1.0, help="Scale for retain loss")
parser.add_argument("--retain-representations", action="store_true", help="Use representation retain loss instead of plain CE")
parser.add_argument("--switching-point-coeffs", type=str, default="alpha:6.0,beta:3.0", help="Beta distribution coeffs for retain_forget_switch")
parser.add_argument("--dpo-beta", type=float, default=0.1, help="DPO beta")
parser.add_argument("--meta-ratio", type=float, default=0.2, help="Held-out meta split ratio for max-entropy TAR")
parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
args = parser.parse_args()

wandb.init(project="tar", name=args.name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True).to(device)

dataloaders = build_tar_dataloaders(
    tokenizer=tokenizer,
    batch_size=args.batch_size,
    adversary_batch_size=args.adversary_batch_size,
    tar_loss_type=args.tar_loss_type,
    model_name="qwen",
    meta_ratio=args.meta_ratio,
    seed=args.seed,
    distributed=False,
)

ref_model = None
retain_model = None
if args.tar_loss_type == "dpo":
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True).to(device)
if args.retain_representations:
    retain_model = AutoModelForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True).to(device)

trainer = TARTrainer(
    model=model,
    model_name="qwen",
    tokenizer=tokenizer,
    dataloaders=dataloaders,
    ref_model=ref_model,
    retain_model=retain_model,
    device=device,
    out_dir=args.save_dir,
    lr=args.lr,
    grad_accum=args.grad_accum,
    num_training_steps=args.steps,
    save_steps=args.save_steps,
    save_epochs=args.save_epochs,
    tar_inner_loop_steps=args.tar_inner_loop_steps,
    tar_tamper_resistance_loss_lower_bound=args.tar_tamper_resistance_loss_lower_bound,
    tar_tamper_resistance_grad_scale=args.tar_tamper_resistance_grad_scale,
    tar_tamper_resistance_loss_type=args.tar_loss_type,
    schedule_lambda=args.schedule_lambda,
    inner_optimizer_warmup_steps=args.inner_optimizer_warmup_steps,
    unbounded=args.unbounded,
    use_weighting_schedule=args.use_weighting_schedule,
    adversary_dist_types=args.adversary_dist_types,
    adversary_lr_schedulers=args.adversary_lr_schedulers,
    tar_num_tasks_sampled=args.tar_num_tasks_sampled,
    adversary_lr_samples=args.adversary_lr_samples,
    tar_inner_loop_subsample=args.tar_inner_loop_subsample,
    tar_retain_scale=args.tar_retain_scale,
    retain_representations=args.retain_representations,
    switching_point_coeffs=args.switching_point_coeffs,
    dpo_beta=args.dpo_beta,
)

trainer.train()
