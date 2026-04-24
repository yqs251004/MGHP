# a naive trainer using repnoise loss and beavertails dataset
import copy
import json
import math
import os
import random

import torch
import torch.distributed as dist
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm

try:
    # PyTorch 2.x
    from torch.func import functional_call as _functional_call
except Exception:
    # Fallback for older versions
    from torch.nn.utils.stateless import functional_call as _functional_call

from reproduce.train.loss import rep_noise_loss, register_activation_hook, contrastive_loss, weighted_ce_loss
try:
    import wandb
except Exception:  # wandb is optional
    wandb = None

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullStateDictConfig,
        StateDictType,
    )
except Exception:
    FSDP = None
    FullStateDictConfig = None
    StateDictType = None


def unwrap_model(model):
    if FSDP is not None and isinstance(model, FSDP):
        wrapped = model.module
        while hasattr(wrapped, "module"):
            wrapped = wrapped.module
        return wrapped
    while hasattr(model, "module"):
        model = model.module
    return model


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return (not is_distributed()) or dist.get_rank() == 0


def set_dataloader_epoch(dataloader, epoch):
    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def save_model_and_tokenizer(model, raw_model, tokenizer, save_path):
    os.makedirs(save_path, exist_ok=True)
    if FSDP is not None and isinstance(model, FSDP):
        save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_cfg):
            state_dict = model.state_dict()
        if is_main_process():
            raw_model.save_pretrained(save_path, state_dict=state_dict)
            tokenizer.save_pretrained(save_path)
            print(f"Model saved to {save_path}")
        if is_distributed():
            dist.barrier()
        return

    if is_main_process():
        raw_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    if is_distributed():
        dist.barrier()


def is_fsdp_model(model):
    return FSDP is not None and isinstance(model, FSDP)


def get_optimizer_model(model, raw_model):
    return model if is_fsdp_model(model) else raw_model


def get_stateless_model(model, raw_model):
    return raw_model


def clip_grad_norm(model, max_grad_norm):
    if is_fsdp_model(model):
        return model.clip_grad_norm_(max_grad_norm)
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)


def reduce_loss(loss):
    if isinstance(loss, torch.Tensor) and loss.ndim > 0:
        return loss.mean()
    return loss


def log_p_loss_from_logits(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def max_entropy_loss_from_logits(logits):
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return -entropy.mean()


def sequence_logps_from_logits(logits, labels):
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits and labels must have matching batch/sequence shapes.")

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:].clone()
    loss_mask = shifted_labels != -100
    shifted_labels = shifted_labels.masked_fill(~loss_mask, 0)
    per_token_logps = torch.gather(
        shifted_logits.log_softmax(-1),
        dim=2,
        index=shifted_labels.unsqueeze(2),
    ).squeeze(2)
    return (per_token_logps * loss_mask).sum(-1)


def dpo_loss_from_logits(
    policy_chosen_logits,
    policy_chosen_labels,
    policy_rejected_logits,
    policy_rejected_labels,
    reference_chosen_logps=None,
    reference_rejected_logps=None,
    beta=0.1,
):
    policy_chosen_logps = sequence_logps_from_logits(policy_chosen_logits, policy_chosen_labels)
    policy_rejected_logps = sequence_logps_from_logits(policy_rejected_logits, policy_rejected_labels)
    if reference_chosen_logps is None:
        reference_chosen_logps = torch.zeros_like(policy_chosen_logps)
    if reference_rejected_logps is None:
        reference_rejected_logps = torch.zeros_like(policy_rejected_logps)

    logits = (policy_chosen_logps - policy_rejected_logps) - (
        reference_chosen_logps - reference_rejected_logps
    )
    losses = -torch.nn.functional.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    reward_acc = (chosen_rewards > rejected_rewards).float().mean()
    return losses.mean(), reward_acc


def _filter_dpo_inputs(inputs, chosen: bool = False):
    prefix = "chosen_" if chosen else "rejected_"
    if f"{prefix}input_ids" not in inputs:
        return inputs
    return {
        "input_ids": inputs[f"{prefix}input_ids"],
        "attention_mask": inputs[f"{prefix}attention_mask"],
        "labels": inputs[f"{prefix}labels"],
    }


def _forward_inputs(inputs):
    return {k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}


def obj_standard_max_next_token(model, inputs, chosen: bool = False):
    filtered_inputs = _filter_dpo_inputs(inputs, chosen=chosen)
    outputs = model(**_forward_inputs(filtered_inputs), output_hidden_states=False)
    return log_p_loss_from_logits(outputs.logits, filtered_inputs["labels"])


def obj_model_mse_representations(model, inputs, base_model):
    forward_inputs = _forward_inputs(inputs)
    with torch.no_grad():
        base_outputs = base_model(**forward_inputs, output_hidden_states=True)
    model_outputs = model(**forward_inputs, output_hidden_states=True)
    loss = log_p_loss_from_logits(model_outputs.logits, inputs["labels"])
    rep_loss = torch.mean(
        torch.stack(
            [
                torch.norm(base_hidden - model_hidden, dim=-1).mean()
                for base_hidden, model_hidden in zip(
                    base_outputs.hidden_states, model_outputs.hidden_states
                )
            ]
        )
    )
    return loss + rep_loss


def get_next_batch(iterator, dataloader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(dataloader)
        batch = next(iterator)
    return batch, iterator


def next_n_batches(iterator, dataloader, n):
    batches = []
    for _ in range(n):
        batch, iterator = get_next_batch(iterator, dataloader)
        batches.append(batch)
    return batches, iterator


def distributed_sample_task(spec: str):
    task_probs = {item.split(":")[0]: float(item.split(":")[1]) for item in spec.split(",")}
    task_type = random.choices(list(task_probs.keys()), weights=list(task_probs.values()), k=1)[0]
    if is_distributed():
        payload = [task_type]
        dist.broadcast_object_list(payload, src=0)
        task_type = payload[0]
    return task_type


def distributed_sample_adversary_lr(adversary_lr_samples):
    if isinstance(adversary_lr_samples, str):
        adversary_lr_samples = [float(lr) for lr in adversary_lr_samples.split(",")]
    idx = random.randrange(len(adversary_lr_samples))
    if is_distributed():
        tensor = torch.tensor([idx], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        dist.broadcast(tensor, src=0)
        idx = int(tensor.item())
    return adversary_lr_samples[idx]


def schedule(i: int, K: int, schedule_lambda: float = 0.5):
    return torch.exp(schedule_lambda * (torch.tensor(i) - (K - 1))).item()


def sample_switching_point(switching_point_coeffs: str, tar_inner_loop_steps: int):
    coeffs = {
        key: float(value)
        for key, value in (item.split(":") for item in switching_point_coeffs.split(","))
    }
    point = int(
        torch.distributions.Beta(coeffs["alpha"], coeffs["beta"]).sample()
        * tar_inner_loop_steps
    )
    if is_distributed():
        payload = [point]
        dist.broadcast_object_list(payload, src=0)
        point = payload[0]
    return point


class ModelStorage:
    def __init__(self):
        self.storage_dict = {"params": {}, "grads": {}}

    def clear_params(self):
        self.storage_dict["params"].clear()

    def clear_grads(self):
        self.storage_dict["grads"].clear()

    def collect_param_or_grad(self, model, to_cpu: bool = False, mode: str = "grads", scale: float = 1.0):
        for idx, param in enumerate(model.parameters()):
            if not param.requires_grad:
                continue
            if mode == "params":
                tensor = param.detach().clone()
                self.storage_dict["params"][idx] = tensor.cpu() if to_cpu else tensor
            elif mode == "grads" and param.grad is not None:
                grad = param.grad.detach().clone() * scale
                if idx not in self.storage_dict["grads"]:
                    self.storage_dict["grads"][idx] = grad.cpu() if to_cpu else grad
                else:
                    existing = self.storage_dict["grads"][idx]
                    if existing.device != grad.device:
                        existing = existing.to(grad.device)
                    existing = existing + grad
                    self.storage_dict["grads"][idx] = existing.cpu() if to_cpu else existing

    def add_from_storage_to_model(self, model, skip_check: bool = False, mode: str = "grads"):
        for idx, param in enumerate(model.parameters()):
            if not param.requires_grad:
                continue
            if mode == "params":
                if idx in self.storage_dict["params"]:
                    param.data.copy_(self.storage_dict["params"][idx].to(param.device))
                continue
            if not skip_check:
                assert (idx in self.storage_dict["grads"]) == (param.grad is not None)
            if idx in self.storage_dict["grads"] and param.grad is not None:
                param.grad += self.storage_dict["grads"][idx].to(param.device)


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved

class SFTTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataloader,
        eval_dataloader=None,
        lr=1e-5,
        num_training_steps=None,
        epochs=None,
        grad_accum=1,
        max_grad_norm=1.0,
        device=None,
        log_steps=1,
        out_dir=None,
        save_steps=None,
        eval_steps=10,
        save_checkpoint_epoch=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.lr = lr
        self.grad_accum = grad_accum
        self.max_grad_norm = max_grad_norm
        self.log_steps = log_steps
        if out_dir is None:
            out_dir = "./sft_checkpoints"
        self.save_checkpoint_epoch = save_checkpoint_epoch
        self.out_dir = out_dir
        self.save_steps = save_steps   
        self.eval_steps = eval_steps

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.raw_model = unwrap_model(self.model)
        self.raw_model.to(self.device)
        self.model.train()

        self.opt_model = get_optimizer_model(self.model, self.raw_model)
        self.stateless_model = get_stateless_model(self.model, self.raw_model)
        self.opt = AdamW(self.opt_model.parameters(), lr=self.lr)

        self.num_training_steps = num_training_steps
        self.epochs = epochs
        if self.epochs is None:
            raise ValueError("You must specify epochs for SFTtrainer.")
        if self.num_training_steps is None:
            self.num_training_steps = self.epochs * len(self.train_dataloader) // self.grad_accum

        self.lr_scheduler = get_scheduler(
            "constant",
            optimizer=self.opt,
        )

        self.global_step = 0
        if wandb is not None and wandb.run is not None:
            bs = getattr(self.train_dataloader, "batch_size", None)
            wandb.config.update(
                {
                    "lr": self.lr,
                    "grad_accum": self.grad_accum,
                    "max_grad_norm": self.max_grad_norm,
                    "log_steps": self.log_steps,
                    "save_steps": self.save_steps,
                    "epochs": self.epochs,
                    "num_training_steps": self.num_training_steps,
                    "batch_size": bs,
                }
            )
        
    def save(self, name):
        save_path = os.path.join(self.out_dir, name)
        save_model_and_tokenizer(self.model, self.raw_model, self.tokenizer, save_path)

    def train(self):
        os.makedirs(self.out_dir, exist_ok=True)

        for epoch in range(self.epochs):
            set_dataloader_epoch(self.train_dataloader, epoch)
            if self.eval_dataloader is not None:
                set_dataloader_epoch(self.eval_dataloader, epoch)
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}", disable=not is_main_process())
            for step, batch in enumerate(pbar):
                
                if self.eval_dataloader is not None and self.global_step % self.eval_steps == 0:
                    eval_pbar = tqdm(self.eval_dataloader, desc="Evaluating", leave=False, disable=not is_main_process())
                    eval_avg_loss = 0.0
                    for eval_step, eval_batch in enumerate(eval_pbar):
                        eval_batch = {k: v.to(self.device) for k, v in eval_batch.items()}
                        with torch.no_grad():
                            eval_outputs = self.model(**eval_batch)
                            eval_loss = eval_outputs.loss
                            eval_avg_loss += eval_loss.item()
                        eval_pbar.set_postfix(eval_loss=f"{eval_loss.item():.4f}")
                    # calculate average loss
                    eval_avg_loss /= (eval_step + 1)
                    print(f"Epoch {epoch+1}, Step {self.global_step}, Eval Loss: {eval_avg_loss:.4f}")
                    if wandb is not None and wandb.run is not None:
                        wandb.log(
                            {
                                "eval/loss": eval_avg_loss,
                            },
                            step=self.global_step,
                        )

                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss
                loss = loss / self.grad_accum
                loss.backward()

                pbar.set_postfix(loss=f"{loss.item() * self.grad_accum:.4f}")

                if (step + 1) % self.grad_accum == 0:
                    # get the grad norm
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()
                    self.global_step += 1

                    if self.global_step % self.log_steps == 0:
                        print(f"Epoch {epoch+1}, Step {self.global_step}, Loss: {loss.item() * self.grad_accum:.4f}")
                        if wandb is not None and wandb.run is not None:
                            wandb.log(
                                {
                                    "loss/total": loss.item() * self.grad_accum,
                                    "grad_norm": grad_norm.item(),
                                },
                                step=self.global_step,
                            )

                if self.save_steps is not None and self.global_step % self.save_steps == 0:
                    self.save(f"checkpoint-step-{self.global_step}")
                
                if self.global_step >= self.num_training_steps:
                    # Save final model
                    self.save("final-model")
                    return
            
            # End of epoch
            if self.save_checkpoint_epoch and (epoch + 1) % self.save_checkpoint_epoch == 0:
                self.save(f"checkpoint-epoch-{epoch+1}")
        return

class RepnoiseTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        harmful_dataloader=None,
        harmless_dataloader=None,
        lr=1e-5,
        # weight_decay=1,
        num_training_steps=None,
        epochs=None,
        grad_accum=1,
        max_grad_norm=1.0,
        device=None,
        log_steps=20,
        out_dir=None,
        save_steps=None,
        beta=0.001,
        alpha=1,
    ):
        self.model = model
        self.tokenizer = tokenizer

        if harmful_dataloader is None or harmless_dataloader is None:
            raise ValueError(
                "RepnoiseTrainer requires paired dataloaders: harmful/harmless (or unsafe/safe)."
            )
        self.harmful_dataloader = harmful_dataloader
        self.harmless_dataloader = harmless_dataloader
        self.lr = lr
        # self.weight_decay = weight_decay
        self.grad_accum = grad_accum
        self.max_grad_norm = max_grad_norm
        self.log_steps = log_steps
        if out_dir is None:
            out_dir = "./repnoise_checkpoints"
        self.out_dir = out_dir
        self.save_steps = save_steps    

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.raw_model = unwrap_model(self.model)
        self.raw_model.to(self.device)
        self.model.train()

        self.opt_model = get_optimizer_model(self.model, self.raw_model)
        self.stateless_model = get_stateless_model(self.model, self.raw_model)
        self.opt = AdamW(self.opt_model.parameters(), lr=self.lr)

        self.num_training_steps = num_training_steps
        self.epochs = epochs
        if self.epochs is None:
            raise ValueError("You must specify epochs for RepnoiseTrainer.")
        if self.num_training_steps is None:
            self.num_training_steps = self.epochs * min(
                len(self.harmful_dataloader), len(self.harmless_dataloader)
            ) // self.grad_accum

        self.lr_scheduler = get_scheduler(
            "cosine",
            optimizer=self.opt,
            num_warmup_steps=self.num_training_steps // 10,
            num_training_steps=self.num_training_steps,  # You might want to set this dynamically
        )

        self.activations, self.hook_handles = register_activation_hook(self.model)

        self.beta = beta
        self.alpha = alpha
        self.global_step = 0

        if wandb is not None and wandb.run is not None:
            safe_bs = getattr(self.harmless_dataloader, "batch_size", None)
            unsafe_bs = getattr(self.harmful_dataloader, "batch_size", None)
            wandb.config.update(
                {
                    "lr": self.lr,
                    "grad_accum": self.grad_accum,
                    "max_grad_norm": self.max_grad_norm,
                    "log_steps": self.log_steps,
                    "save_steps": self.save_steps,
                    "epochs": self.epochs,
                    "num_training_steps": self.num_training_steps,
                    "beta": self.beta,
                    "alpha": self.alpha,
                    "harmless_batch_size": safe_bs,
                    "harmful_batch_size": unsafe_bs,
                }
            )

    def save(self, name):
        save_path = os.path.join(self.out_dir, name)
        save_model_and_tokenizer(self.model, self.raw_model, self.tokenizer, save_path)
    
    def train(self):
        os.makedirs(self.out_dir, exist_ok=True)

        for epoch in range(self.epochs):
            set_dataloader_epoch(self.harmful_dataloader, epoch)
            set_dataloader_epoch(self.harmless_dataloader, epoch)
            total_steps = min(len(self.harmful_dataloader), len(self.harmless_dataloader))
            epoch_iter = zip(self.harmful_dataloader, self.harmless_dataloader)
            pbar = tqdm(epoch_iter, total=total_steps, desc=f"Epoch {epoch+1}", disable=not is_main_process())
            for step, (harmful_batch, harmless_batch) in enumerate(pbar):
                harmful_batch = {k: v.to(self.device) for k, v in harmful_batch.items()}
                harmless_batch = {k: v.to(self.device) for k, v in harmless_batch.items()}

                loss, harmless_losses, noise_loss, harmful_losses = rep_noise_loss(
                    self.model,
                    harmful_batch,
                    harmless_batch,
                    self.activations,
                    self.beta,
                    self.alpha,
                )
                loss = loss / self.grad_accum
                loss.backward()

                pbar.set_postfix(loss=f"{loss.item() * self.grad_accum:.4f}")

                if (step + 1) % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()
                    self.global_step += 1

                    # if self.global_step % self.log_steps == 0:
                    print(f"Epoch {epoch+1}, Step {self.global_step}, Loss: {loss.item() * self.grad_accum:.4f}")
                    if wandb is not None and wandb.run is not None:
                        wandb.log(
                            {
                                "loss/total": loss.item() * self.grad_accum,
                                "loss/harmless": harmless_losses.item(),
                                "loss/noise": noise_loss.item(),
                                "loss/harmful": harmful_losses.item(),
                            },
                            step=self.global_step,
                        )

                    if self.save_steps is not None and self.global_step % self.save_steps == 0:
                        self.save(f"checkpoint-step-{self.global_step}")
                    
                    if self.global_step >= self.num_training_steps:
                        for h in self.hook_handles:
                            h.remove()
                        # Save final model
                        self.save("final-model")
                        return
            
            # End of epoch
            self.save(f"checkpoint-epoch-{epoch+1}")
        for h in self.hook_handles:
            h.remove()
        return

class ContrastiveTrainer:
    def __init__(
        self,
        model,
        model_name,
        tokenizer,
        harmful_dataloader=None,
        harmless_dataloader=None,
        lr=1e-5,
        # weight_decay=1,
        num_training_steps=None,
        epochs=None,
        grad_accum=1,
        max_grad_norm=1.0,
        device=None,
        log_steps=20,
        out_dir=None,
        save_steps=None,
        beta=0.001,
        alpha=1,
    ):
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer

        if harmful_dataloader is None or harmless_dataloader is None:
            raise ValueError(
                "RepnoiseTrainer requires paired dataloaders: harmful/harmless (or unsafe/safe)."
            )
        self.harmful_dataloader = harmful_dataloader
        self.harmless_dataloader = harmless_dataloader
        self.lr = lr
        # self.weight_decay = weight_decay
        self.grad_accum = grad_accum
        self.max_grad_norm = max_grad_norm
        self.log_steps = log_steps
        if out_dir is None:
            out_dir = "./repnoise_checkpoints"
        self.out_dir = out_dir
        self.save_steps = save_steps    

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.train()

        self.opt = AdamW(self.model.parameters(), lr=self.lr)

        self.num_training_steps = num_training_steps
        self.epochs = epochs
        if self.epochs is None:
            raise ValueError("You must specify epochs for RepnoiseTrainer.")
        if self.num_training_steps is None:
            self.num_training_steps = self.epochs * min(
                len(self.harmful_dataloader), len(self.harmless_dataloader)
            ) // self.grad_accum

        self.lr_scheduler = get_scheduler(
            "cosine",
            optimizer=self.opt,
            num_warmup_steps=self.num_training_steps // 10,
            num_training_steps=self.num_training_steps,  # You might want to set this dynamically
        )

        self.activations, self.hook_handles = register_activation_hook(self.model)

        self.beta = beta
        self.alpha = alpha
        self.global_step = 0

        if wandb is not None and wandb.run is not None:
            safe_bs = getattr(self.harmless_dataloader, "batch_size", None)
            unsafe_bs = getattr(self.harmful_dataloader, "batch_size", None)
            wandb.config.update(
                {
                    "lr": self.lr,
                    "grad_accum": self.grad_accum,
                    "max_grad_norm": self.max_grad_norm,
                    "log_steps": self.log_steps,
                    "save_steps": self.save_steps,
                    "epochs": self.epochs,
                    "num_training_steps": self.num_training_steps,
                    "beta": self.beta,
                    "alpha": self.alpha,
                    "harmless_batch_size": safe_bs,
                    "harmful_batch_size": unsafe_bs,
                }
            )

    def save(self, name):
        save_path = os.path.join(self.out_dir, name)
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    
    def train(self):
        os.makedirs(self.out_dir, exist_ok=True)

        for epoch in range(self.epochs):
            total_steps = min(len(self.harmful_dataloader), len(self.harmless_dataloader))
            epoch_iter = zip(self.harmful_dataloader, self.harmless_dataloader)
            pbar = tqdm(epoch_iter, total=total_steps, desc=f"Epoch {epoch+1}")
            for step, (harmful_batch, harmless_batch) in enumerate(pbar):
                harmful_batch = {k: v.to(self.device) for k, v in harmful_batch.items()}
                harmless_batch = {k: v.to(self.device) for k, v in harmless_batch.items()}

                loss, harmless_losses, noise_loss, harmful_losses = contrastive_loss(
                    self.model,
                    self.model_name,
                    harmful_batch,
                    harmless_batch,
                    self.activations,
                    self.beta,
                    self.alpha,
                )
                loss = loss / self.grad_accum
                loss.backward()

                pbar.set_postfix(loss=f"{loss.item() * self.grad_accum:.4f}")

                if (step + 1) % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()
                    self.global_step += 1

                    # if self.global_step % self.log_steps == 0:
                    print(f"Epoch {epoch+1}, Step {self.global_step}, Loss: {loss.item() * self.grad_accum:.4f}")
                    if wandb is not None and wandb.run is not None:
                        wandb.log(
                            {
                                "loss/total": loss.item() * self.grad_accum,
                                "loss/harmless": harmless_losses.item(),
                                "loss/noise": noise_loss.item(),
                                "loss/harmful": harmful_losses.item(),
                            },
                            step=self.global_step,
                        )

                    if self.save_steps is not None and self.global_step % self.save_steps == 0:
                        self.save(f"checkpoint-step-{self.global_step}")
                    
                    if self.global_step >= self.num_training_steps:
                        for h in self.hook_handles:
                            h.remove()
                        # Save final model
                        self.save("final-model")
                        return
            
            # End of epoch
            self.save(f"checkpoint-epoch-{epoch+1}")
        for h in self.hook_handles:
            h.remove()
        return

class SAMTrainer:
    def __init__(
        self,
        model,
        model_name,
        tokenizer,
        dataloader,
        lr=1e-5,
        # weight_decay=1,
        num_training_steps=None,
        epochs=None,
        grad_accum=1,
        max_grad_norm=1.0,
        device=None,
        log_steps=20,
        out_dir=None,
        save_steps=None,
        alpha=0.8,
        rho=0.05,
        save_checkpoint_epoch=None,
    ):
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.lr = lr
        # self.weight_decay = weight_decay
        self.grad_accum = grad_accum
        self.max_grad_norm = max_grad_norm
        self.log_steps = log_steps
        if out_dir is None:
            out_dir = "./repnoise_checkpoints"
        self.out_dir = out_dir
        self.save_steps = save_steps    
        self.rho = rho
        self.alpha = alpha
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.raw_model = unwrap_model(self.model)
        if not is_fsdp_model(self.model):
            self.raw_model.to(self.device)
        self.model.train()

        self.opt_model = get_optimizer_model(self.model, self.raw_model)
        self.stateless_model = get_stateless_model(self.model, self.raw_model)
        self.opt = AdamW(self.opt_model.parameters(), lr=self.lr)

        self.num_training_steps = num_training_steps
        self.epochs = epochs
        self.save_checkpoint_epoch = save_checkpoint_epoch
        if self.epochs is None:
            raise ValueError("You must specify epochs for RepnoiseTrainer.")
        if self.num_training_steps is None:
            self.num_training_steps = self.epochs * len(self.dataloader) // self.grad_accum

        self.lr_scheduler = get_scheduler(
            "cosine",
            optimizer=self.opt,
            num_warmup_steps=self.num_training_steps // 10,
            num_training_steps=self.num_training_steps,  # You might want to set this dynamically
        )

        self.global_step = 0

        if wandb is not None and wandb.run is not None:
            bs = getattr(self.dataloader, "batch_size", None)
            wandb.config.update(
                {
                    "lr": self.lr,
                    "grad_accum": self.grad_accum,
                    "max_grad_norm": self.max_grad_norm,
                    "log_steps": self.log_steps,
                    "save_steps": self.save_steps,
                    "epochs": self.epochs,
                    "num_training_steps": self.num_training_steps,  
                    "batch_size": bs,
                }
            )

    def save(self, name):
        save_path = os.path.join(self.out_dir, name)
        save_model_and_tokenizer(self.model, self.raw_model, self.tokenizer, save_path)
    
    def train(self):
        os.makedirs(self.out_dir, exist_ok=True)

        for epoch in range(self.epochs):
            set_dataloader_epoch(self.dataloader, epoch)
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}", disable=not is_main_process())
            for step, batch in enumerate(pbar):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                loss_raw = reduce_loss(self.model(**batch).loss)
                trainable_named_params = [
                    (name, p) for name, p in self.stateless_model.named_parameters() if p.requires_grad
                ]
                params = [p for _, p in trainable_named_params]
                grads = torch.autograd.grad(
                    loss_raw,
                    params,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )
                with torch.no_grad():
                    global_norm_sq = None
                    for g in grads:
                        if g is None:
                            continue
                        g2 = (g.detach().float() ** 2).sum()
                        global_norm_sq = g2 if global_norm_sq is None else (global_norm_sq + g2)
                    if global_norm_sq is None:
                        scale = None
                    else:
                        global_norm = torch.sqrt(global_norm_sq)
                        scale = self.rho / (global_norm + 1e-12) if global_norm.item() != 0.0 else None
                
                param_and_buffer_dict = {name: p for name, p in self.stateless_model.named_parameters()}
                param_and_buffer_dict.update({name: b for name, b in self.stateless_model.named_buffers()})
                if scale is not None:
                    for (name, p), g in zip(trainable_named_params, grads):
                        if g is None:
                            continue
                        perturb = g.detach().to(dtype=p.dtype) * scale
                        param_and_buffer_dict[name] = p + perturb
                loss_perturbed = reduce_loss(_functional_call(self.stateless_model, param_and_buffer_dict, (), batch).loss)

                loss = self.alpha * (loss_perturbed - loss_raw) + loss_raw
                loss = loss / self.grad_accum
                loss.backward()

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    raw_loss=f"{loss_raw.item():.4f}",
                    perturbed_loss=f"{loss_perturbed.item():.4f}",
                )
                
                if (step + 1) % self.grad_accum == 0:
                    clip_grad_norm(self.model, self.max_grad_norm)
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()
                    self.global_step += 1

                    # if self.global_step % self.log_steps == 0:
                    if is_main_process():
                        tqdm.write(
                            f"Epoch {epoch+1}, Step {self.global_step}, Loss: {loss_raw.item():.4f}, Perturbed: {loss_perturbed.item():.4f}"
                        )
                    if wandb is not None and wandb.run is not None:
                        wandb.log(
                            {
                                "sam_loss/total": loss.item(),
                                "sam_loss/raw": loss_raw.item(),
                                "sam_loss/perturbed": loss_perturbed.item(),
                                "sam_loss/grad_norm": global_norm.item() if global_norm_sq is not None else 0.0,
                            },
                            step=self.global_step,
                        )

                    if self.save_steps is not None and self.global_step % self.save_steps == 0:
                        self.save(f"checkpoint-step-{self.global_step}")
                    
                    if self.global_step >= self.num_training_steps:
                        # Save final model
                        self.save("final-model")
                        return
            
            # End of epoch
            if self.save_checkpoint_epoch and (epoch + 1) % self.save_checkpoint_epoch == 0:
                self.save(f"checkpoint-epoch-{epoch+1}")
        return


class GOTrainer:
    def __init__(
        self,
        model,
        model_name,
        tokenizer,
        dataloader,
        lr=1e-5,
        # weight_decay=1,
        num_training_steps=None,
        epochs=None,
        grad_accum=1,
        max_grad_norm=1.0,
        device=None,
        log_steps=20,
        out_dir=None,
        save_steps=None,
        std=0.0001
    ):
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.lr = lr
        # self.weight_decay = weight_decay
        self.grad_accum = grad_accum
        self.max_grad_norm = max_grad_norm
        self.log_steps = log_steps
        if out_dir is None:
            out_dir = "./repnoise_checkpoints"
        self.out_dir = out_dir
        self.save_steps = save_steps    
        self.std = std
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.train()

        self.opt = AdamW(self.model.parameters(), lr=self.lr)

        self.num_training_steps = num_training_steps
        self.epochs = epochs
        if self.epochs is None:
            raise ValueError("You must specify epochs for RepnoiseTrainer.")
        if self.num_training_steps is None:
            self.num_training_steps = self.epochs * len(self.dataloader) // self.grad_accum

        self.lr_scheduler = get_scheduler(
            "cosine",
            optimizer=self.opt,
            num_warmup_steps=self.num_training_steps // 10,
            num_training_steps=self.num_training_steps,  # You might want to set this dynamically
        )

        self.global_step = 0

        if wandb is not None and wandb.run is not None:
            bs = getattr(self.dataloader, "batch_size", None)
            wandb.config.update(
                {
                    "lr": self.lr,
                    "std": self.std,
                    "grad_accum": self.grad_accum,
                    "max_grad_norm": self.max_grad_norm,
                    "log_steps": self.log_steps,
                    "save_steps": self.save_steps,
                    "epochs": self.epochs,
                    "num_training_steps": self.num_training_steps,  
                    "batch_size": bs,
                }
            )

    def save(self, name):
        save_path = os.path.join(self.out_dir, name)
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    
    def train(self):
        os.makedirs(self.out_dir, exist_ok=True)

        for epoch in range(self.epochs):
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}")
            for step, batch in enumerate(pbar):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    raw_loss = self.model(**batch).loss

                # get the gaussian perturbation layerwise with std scaled by layer weight norm
                params = [p for p in self.model.parameters() if p.requires_grad]
                perturbations = []
                for p in params:
                    scale = self.std * torch.sqrt((p.detach().float() ** 2).sum()) + 1e-12
                    perturbations.append(torch.randn_like(p) * scale)
                
                # add perturbation and get the loss
                with torch.no_grad():
                    for p, perturb in zip(params, perturbations):
                        p.add_(perturb)
                perturbed_loss = self.model(**batch).loss
                loss = perturbed_loss / self.grad_accum
                loss.backward()

                pbar.set_postfix(
                    loss=f"{raw_loss.item():.4f}",
                    ploss=f"{perturbed_loss.item():.4f}",
                )

                # now restore the original weights
                with torch.no_grad():
                    for p, perturb in zip(params, perturbations):
                        p.sub_(perturb)
                
                if (step + 1) % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()
                    self.global_step += 1

                    # if self.global_step % self.log_steps == 0:
                    tqdm.write(
                        f"Epoch {epoch+1}, Step {self.global_step}, Loss: {raw_loss.item():.4f}, Perturbed: {perturbed_loss.item():.4f}"
                    )
                    if wandb is not None and wandb.run is not None:
                        wandb.log(
                            {
                                "loss/total": raw_loss.item(),
                                "loss/perturbed": perturbed_loss.item(),
                            },
                            step=self.global_step,
                        )

                    if self.save_steps is not None and self.global_step % self.save_steps == 0:
                        self.save(f"checkpoint-step-{self.global_step}")
                    
                    if self.global_step >= self.num_training_steps:
                        # Save final model
                        self.save("final-model")
                        return
            
            # End of epoch
            self.save(f"checkpoint-epoch-{epoch+1}")
        return

class BoosterTrainer(SAMTrainer):
    def __init__(
        self,
        model,
        model_name,
        tokenizer,
        harmful_dataloader=None,
        harmless_dataloader=None,
        lr=1e-5,
        num_training_steps=None,
        epochs=None,
        grad_accum=1,
        max_grad_norm=1.0,
        device=None,
        log_steps=20,
        out_dir=None,
        save_steps=None,
        alpha=0.8,
        rho=0.05,
    ):
        super().__init__(
            model=model,
            model_name=model_name,
            tokenizer=tokenizer,
            dataloader=harmless_dataloader,  # we will override the dataloader with paired one
            lr=lr,
            num_training_steps=num_training_steps,
            epochs=epochs,
            grad_accum=grad_accum,
            max_grad_norm=max_grad_norm,
            device=device,
            log_steps=log_steps,
            out_dir=out_dir,
            save_steps=save_steps,
            rho=rho,
        )
        self.alpha = alpha
        self.safe_dataloader = harmless_dataloader
        self.unsafe_dataloader = harmful_dataloader
        if self.safe_dataloader is None or self.unsafe_dataloader is None:
            raise ValueError(
                "BoosterTrainer requires paired dataloaders: harmful/harmless (or unsafe/safe)."
            )
        # Don't create a persistent zip iterator here; it would be exhausted after one epoch.
        self.lr_scheduler = get_scheduler(
            "constant",
            optimizer=self.opt,
        )
        if wandb is not None and wandb.run is not None:
            safe_bs = getattr(self.safe_dataloader, "batch_size", None)
            unsafe_bs = getattr(self.unsafe_dataloader, "batch_size", None)
            wandb.config.update(
                {
                    "lr": self.lr,
                    "epochs": self.epochs,
                    "num_training_steps": self.num_training_steps,  
                    "alpha": self.alpha,
                    "rho": self.rho,
                    "safe_batch_size": safe_bs,
                    "unsafe_batch_size": unsafe_bs,
                }
            )
    
    def train(self):
        os.makedirs(self.out_dir, exist_ok=True)

        for epoch in range(self.epochs):
            set_dataloader_epoch(self.unsafe_dataloader, epoch)
            set_dataloader_epoch(self.safe_dataloader, epoch)
            total_steps = min(len(self.unsafe_dataloader), len(self.safe_dataloader))
            epoch_iter = zip(self.unsafe_dataloader, self.safe_dataloader)
            pbar = tqdm(epoch_iter, total=total_steps, desc=f"Epoch {epoch+1}", disable=not is_main_process())
            for step, (unsafe_batch, safe_batch) in enumerate(pbar):
                unsafe_batch = {k: v.to(self.device) for k, v in unsafe_batch.items()}
                safe_batch = {k: v.to(self.device) for k, v in safe_batch.items()}

                # --- SAM-style perturbation direction from unsafe loss (no inplace param edits) ---
                unsafe_loss_for_grad = reduce_loss(self.model(**unsafe_batch).loss)
                trainable_named_params = [
                    (name, p) for name, p in self.stateless_model.named_parameters() if p.requires_grad
                ]
                params = [p for _, p in trainable_named_params]
                grads = torch.autograd.grad(
                    unsafe_loss_for_grad,
                    params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )

                with torch.no_grad():
                    global_norm_sq = None
                    for g in grads:
                        if g is None:
                            continue
                        g2 = (g.detach().float() ** 2).sum()
                        global_norm_sq = g2 if global_norm_sq is None else (global_norm_sq + g2)
                    if global_norm_sq is None:
                        scale = None
                    else:
                        global_norm = torch.sqrt(global_norm_sq)
                        scale = self.rho / (global_norm + 1e-12) if global_norm.item() != 0.0 else None
                        # scale = self.rho

                # Compute perturbed unsafe loss using functional_call to avoid inplace modifications
                param_and_buffer_dict = {name: p for name, p in self.stateless_model.named_parameters()}
                param_and_buffer_dict.update({name: b for name, b in self.stateless_model.named_buffers()})
                if scale is not None:
                    for (name, p), g in zip(trainable_named_params, grads):
                        if g is None:
                            continue
                        perturb = g.detach().to(dtype=p.dtype) * scale
                        param_and_buffer_dict[name] = p - perturb
                # unsafe_loss_perturbed = _functional_call(self.model, param_and_buffer_dict, (), unsafe_batch).loss
                unsafe_loss_perturbed = reduce_loss(
                    _functional_call(self.stateless_model, param_and_buffer_dict, (), safe_batch).loss
                )
                # test: use weighted loss for each token
                # unsafe_logits_perturbed = _functional_call(self.model, param_and_buffer_dict, (), safe_batch).logits
                # unsafe_loss_perturbed = weighted_ce_loss(unsafe_logits_perturbed, safe_batch["labels"])

                # Losses at the original parameters (safe for backward)
                safe_loss_raw = reduce_loss(self.model(**safe_batch).loss)
                # safe_loss_raw = weighted_ce_loss(self.model(**safe_batch).logits, safe_batch["labels"])
                unsafe_loss_raw = reduce_loss(self.model(**unsafe_batch).loss)

                # IMPORTANT: only backprop through graphs built with current (unmodified) parameters
                # loss = safe_loss_raw - torch.log(
                #     (1 - self.alpha) * unsafe_loss_raw + self.alpha * unsafe_loss_perturbed
                # )
                loss = (1 - self.alpha) * safe_loss_raw + self.alpha * unsafe_loss_perturbed - torch.log(
                    unsafe_loss_raw
                )
                # loss = (1 - self.alpha) * safe_loss_raw + self.alpha * unsafe_loss_perturbed
                loss = loss / self.grad_accum
                loss.backward()
                
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    sloss=f"{safe_loss_raw.item():.4f}",
                    hloss=f"{unsafe_loss_raw.item():.4f}",
                    ploss=f"{unsafe_loss_perturbed.item():.4f}",
                )
                
                if (step + 1) % self.grad_accum == 0:
                    clip_grad_norm(self.model, self.max_grad_norm)
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()
                    self.global_step += 1

                    # if self.global_step % self.log_steps == 0:
                    if is_main_process():
                        tqdm.write(
                            f"Epoch {epoch+1}, Step {self.global_step}, Loss: {loss.item() * self.grad_accum:.4f}, Safe: {safe_loss_raw.item():.4f}, Unsafe: {unsafe_loss_raw.item():.4f}, PerturbedUnsafe: {unsafe_loss_perturbed.item():.4f}"
                        )
                    if wandb is not None and wandb.run is not None:
                        wandb.log(
                            {
                                "loss/total": loss.item() * self.grad_accum,
                                "loss/safe": safe_loss_raw.item(),
                                "loss/unsafe": unsafe_loss_raw.item(),
                                "loss/perturbed_unsafe": unsafe_loss_perturbed.item(),
                            },
                            step=self.global_step,
                        )

                    if self.save_steps is not None and self.global_step % self.save_steps == 0:
                        self.save(f"checkpoint-step-{self.global_step}")
                    
                    if self.global_step >= self.num_training_steps:
                        # Save final model
                        self.save("final-model")
                        return
            
            # End of epoch
            self.save(f"checkpoint-epoch-{epoch+1}")
        return

class SEAMTrainer(BoosterTrainer):
    def __init__(
        self,
        model,
        model_name,
        tokenizer,
        benign_dataloader=None,
        harmful_dataloader=None,
        harmless_dataloader=None,
        lr=1e-5,
        num_training_steps=None,
        epochs=None,
        grad_accum=1,
        max_grad_norm=1.0,
        device=None,
        log_steps=20,
        out_dir=None,
        save_steps=None,
        alpha=1.0,
        beta=0.01,
        epsilon=0.001,
    ):
        super().__init__(
            model=model,
            model_name=model_name,
            tokenizer=tokenizer,
            harmful_dataloader=harmful_dataloader,
            harmless_dataloader=harmless_dataloader,
            lr=lr,
            num_training_steps=num_training_steps,
            epochs=epochs,
            grad_accum=grad_accum,
            max_grad_norm=max_grad_norm,
            device=device,
            log_steps=log_steps,
            out_dir=out_dir,
            save_steps=save_steps,
            alpha=alpha,
            rho=None,
        )
        self.benign_dataloader = benign_dataloader
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        if wandb is not None and wandb.run is not None:
            benign_bs = getattr(self.benign_dataloader, "batch_size", None)
            wandb.config.update(
                {
                    "benign_batch_size": benign_bs,
                    "beta": self.beta,
                    "epsilon": self.epsilon,
                }
            )

    def train(self):
        os.makedirs(self.out_dir, exist_ok=True)

        for epoch in range(self.epochs):
            total_steps = min(len(self.unsafe_dataloader), len(self.safe_dataloader), len(self.benign_dataloader))
            epoch_iter = zip(self.unsafe_dataloader, self.safe_dataloader, self.benign_dataloader)
            pbar = tqdm(epoch_iter, total=total_steps, desc=f"Epoch {epoch+1}")
            for step, (unsafe_batch, safe_batch, benign_batch) in enumerate(pbar):
                unsafe_batch = {k: v.to(self.device) for k, v in unsafe_batch.items()}
                safe_batch = {k: v.to(self.device) for k, v in safe_batch.items()}
                benign_batch = {k: v.to(self.device) for k, v in benign_batch.items()}

                # --- Similarity-based SEAM gradient (no inplace param edits) ---
                # Trainable params (keep names so we can functional_call with perturbed weights)
                trainable_named_params = [
                    (name, p) for name, p in self.model.named_parameters() if p.requires_grad
                ]
                params = [p for _, p in trainable_named_params]

                if len(params) == 0:
                    sim = torch.tensor(0.0, device=self.device)
                else:
                    with torch.autocast(dtype=torch.bfloat16, device_type=self.device.type):
                        benign_loss = self.model(**benign_batch).loss
                        unsafe_loss = self.model(**unsafe_batch).loss
                    
                    benign_grad = torch.autograd.grad(
                        benign_loss,
                        params,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )
                    unsafe_grad = torch.autograd.grad(
                        unsafe_loss,
                        params,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )

                    # global L2 norms (on-device; avoid .item() sync)
                    benign_norm_sq = None
                    for g in benign_grad:
                        if g is None:
                            continue
                        g2 = (g.detach().float() ** 2).sum()
                        benign_norm_sq = g2 if benign_norm_sq is None else (benign_norm_sq + g2)
                    unsafe_norm_sq = None
                    for g in unsafe_grad:
                        if g is None:
                            continue
                        g2 = (g.detach().float() ** 2).sum()
                        unsafe_norm_sq = g2 if unsafe_norm_sq is None else (unsafe_norm_sq + g2)

                    benign_grad_norm = torch.sqrt(benign_norm_sq) if benign_norm_sq is not None else torch.tensor(0.0, device=self.device)
                    unsafe_grad_norm = torch.sqrt(unsafe_norm_sq) if unsafe_norm_sq is not None else torch.tensor(0.0, device=self.device)
                    benign_denom = benign_grad_norm + 1e-12
                    unsafe_denom = unsafe_grad_norm + 1e-12

                    norm_benign_grad = [g / benign_denom if g is not None else None for g in benign_grad]
                    norm_unsafe_grad = [g / unsafe_denom if g is not None else None for g in unsafe_grad]

                    # calculate cosine similarity on the normalized grads
                    sim = 0.0
                    for bg, ug in zip(norm_benign_grad, norm_unsafe_grad):
                        if bg is None or ug is None:
                            continue
                        sim += (bg.detach() * ug.detach()).sum()

                    # Build param+buffer dict for functional_call
                    param_and_buffer_dict = {name: p for name, p in self.model.named_parameters()}
                    param_and_buffer_dict.update({name: b for name, b in self.model.named_buffers()})

                    # Perturbation #1 on benign batch: eps * (ug - sim * bg)
                    eps = float(self.epsilon)
                    perturbed_1 = dict(param_and_buffer_dict)
                    for (name, p), bg, ug in zip(trainable_named_params, norm_benign_grad, norm_unsafe_grad):
                        if bg is None or ug is None:
                            continue
                        perturb = eps * (
                            ug.detach().to(dtype=p.dtype) - sim.to(dtype=p.dtype) * bg.detach().to(dtype=p.dtype)
                        )
                        perturbed_1[name] = p + perturb
                    
                    with torch.autocast(dtype=torch.bfloat16, device_type=self.device.type):
                        sim_loss1 = _functional_call(self.model, perturbed_1, (), benign_batch).loss

                    sim_loss1_grad = torch.autograd.grad(
                        sim_loss1,
                        params,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )
                    sim_loss1_grad = [
                        (g1 - g0) / benign_denom if (g1 is not None and g0 is not None) else None
                        for g1, g0 in zip(sim_loss1_grad, benign_grad)
                    ]

                    # Perturbation #2 on unsafe batch: eps * (1 + sim) * (bg - ug)
                    perturbed_2 = dict(param_and_buffer_dict)
                    for (name, p), bg, ug in zip(trainable_named_params, norm_benign_grad, norm_unsafe_grad):
                        if bg is None or ug is None:
                            continue
                        perturb2 = eps * (
                            bg.detach().to(dtype=p.dtype) - sim.to(dtype=p.dtype) * ug.detach().to(dtype=p.dtype)
                        )
                        perturbed_2[name] = p + perturb2

                    with torch.autocast(dtype=torch.bfloat16, device_type=self.device.type):
                        sim_loss2 = _functional_call(self.model, perturbed_2, (), unsafe_batch).loss

                    sim_loss2_grad = torch.autograd.grad(
                        sim_loss2,
                        params,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )
                    sim_loss2_grad = [
                        (g2 - g0) / unsafe_denom if (g2 is not None and g0 is not None) else None
                        for g2, g0 in zip(sim_loss2_grad, unsafe_grad)
                    ]

                    sim_loss_grad = []
                    for g1, g2 in zip(sim_loss1_grad, sim_loss2_grad):
                        if g1 is None and g2 is None:
                            sim_loss_grad.append(None)
                        else:
                            g_sum = (g1 if g1 is not None else 0.0) + (g2 if g2 is not None else 0.0)
                            sim_loss_grad.append((self.beta * g_sum) / (eps + 1e-12))

                    # scale with grad accumulation, then add into .grad
                    if self.grad_accum != 1:
                        sim_loss_grad = [g / self.grad_accum if g is not None else None for g in sim_loss_grad]
                    # get grad norm
                    sim_norm_sq = None
                    for g in sim_loss_grad:
                        if g is None:
                            continue
                        g2 = (g.detach().float() ** 2).sum()
                        sim_norm_sq = g2 if sim_norm_sq is None else (sim_norm_sq + g2)
                    sim_grad_norm = torch.sqrt(sim_norm_sq) if sim_norm_sq is not None else torch.tensor(0.0, device=self.device)

                    for p, g in zip(params, sim_loss_grad):
                        if g is None:
                            continue
                        if p.grad is None:
                            p.grad = g.detach().clone()
                        else:
                            p.grad.add_(g.detach())

                # Losses at the original parameters (safe for backward)
                with torch.autocast(dtype=torch.bfloat16, device_type=self.device.type):
                    safe_loss_raw = self.model(**safe_batch).loss
                    unsafe_loss_raw = self.model(**unsafe_batch).loss

                # IMPORTANT: only backprop through graphs built with current (unmodified) parameters
                loss = self.alpha * safe_loss_raw - torch.log(unsafe_loss_raw)
                loss = loss / self.grad_accum
                loss.backward()
                
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    sloss=f"{safe_loss_raw.item():.4f}",
                    hloss=f"{unsafe_loss_raw.item():.4f}",
                    simloss=f"{sim.item() * self.beta:.4f}",
                )
                
                if (step + 1) % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()
                    self.global_step += 1

                    # if self.global_step % self.log_steps == 0:
                    tqdm.write(
                        f"Epoch {epoch+1}, Step {self.global_step}, Loss: {loss.item() * self.grad_accum:.4f}, Safe: {safe_loss_raw.item():.4f}, Unsafe: {unsafe_loss_raw.item():.4f}, Trap: {sim.item() * self.beta:.4f}, Benign Grad Norm: {benign_grad_norm.item():.4f}, Unsafe Grad Norm: {unsafe_grad_norm.item():.4f}, Sim Grad Norm: {sim_grad_norm.item():.4f}"
                    )
                    if wandb is not None and wandb.run is not None:
                        wandb.log(
                            {
                                "loss/total": loss.item() * self.grad_accum,
                                "loss/safe": safe_loss_raw.item(),
                                "loss/unsafe": unsafe_loss_raw.item(),
                                "loss/trap": sim.item() * self.beta,
                                "grad_norm/sim_loss": sim_grad_norm.item(),
                                "grad_norm/benign": benign_grad_norm.item(),
                                "grad_norm/unsafe": unsafe_grad_norm.item(),
                            },
                            step=self.global_step,
                        )

                    if self.save_steps is not None and self.global_step % self.save_steps == 0:
                        self.save(f"checkpoint-step-{self.global_step}")
                    
                    if self.global_step >= self.num_training_steps:
                        # Save final model
                        self.save("final-model")
                        return
            
            # End of epoch
            self.save(f"checkpoint-epoch-{epoch+1}")
        return

class OrthogonalTrainer:
    def __init__(
        self,
        model,
        model_name,
        tokenizer,
        tar_dataloader,
        aux_dataloader,
        lr=1e-5,
        num_training_steps=None,
        epochs=None,
        grad_accum=1,
        max_grad_norm=1.0,
        device=None,
        log_steps=20,
        out_dir=None,
        save_steps=None,
        save_per_outer_step=1,
        epsilon=0.01,
    ):
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.tar_dataloader = tar_dataloader
        self.aux_dataloader = aux_dataloader
        self.lr = lr
        self.grad_accum = grad_accum
        self.max_grad_norm = max_grad_norm
        self.log_steps = log_steps
        if out_dir is None:
            out_dir = "./repnoise_checkpoints"
        self.out_dir = out_dir
        self.save_steps = save_steps 
        self.save_per_outer_step = save_per_outer_step   
        self.epsilon = epsilon
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.train()

        self.opt = AdamW(self.model.parameters(), lr=self.lr)

        self.num_training_steps = num_training_steps
        self.epochs = epochs
        if self.epochs is None:
            raise ValueError("You must specify epochs for OrthogonalTrainer.")

        # we will specify this in training loop
        if self.num_training_steps is None:
            self.num_training_steps = self.epochs * len(self.tar_dataloader)
        self.lr_scheduler = get_scheduler(
            "cosine",
            optimizer=self.opt,
            num_warmup_steps=self.num_training_steps // 10,
            num_training_steps=self.num_training_steps,  # You might want to set this dynamically
        )

        self.global_step = 0
        if wandb is not None and wandb.run is not None:
            wandb.config.update(
                {
                    "lr": self.lr,
                    "grad_accum": self.grad_accum,
                    "max_grad_norm": self.max_grad_norm,
                    "log_steps": self.log_steps,
                    "save_steps": self.save_steps,
                    "epochs": self.epochs,
                    "num_training_steps": self.num_training_steps,  
                    "epsilon": self.epsilon,
                }
            )
    
    def save(self, name):
        save_path = os.path.join(self.out_dir, name)
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    
    def train(self):
        os.makedirs(self.out_dir, exist_ok=True)
        # if self.num_training_steps is None:
        #     per_pair_steps = [min(len(tar_dl), len(aux_dl)) for tar_dl, aux_dl in zip(self.tar_dataloader_lst, self.aux_dataloader_lst)]
        #     inner_steps_per_epoch = min(per_pair_steps) if len(per_pair_steps) > 0 else 0
        #     micro_steps_per_epoch = inner_steps_per_epoch * len(self.tar_dataloader_lst)
            # self.num_training_steps = max(1, (self.epochs * micro_steps_per_epoch) // max(1, self.grad_accum))

        # self.lr_scheduler = get_scheduler(
        #     "cosine",
        #     optimizer=self.opt,
        #     num_warmup_steps=self.num_training_steps // 10,
        #     num_training_steps=self.num_training_steps,
        # )

        for epoch in range(self.epochs):
            # Recreate iterators every epoch; zipped iterators are exhausted after one pass.
            # epoch_iters = [
            #     zip(tar_dl, aux_dl)
            #     for tar_dl, aux_dl in zip(self.tar_dataloader_lst, self.aux_dataloader_lst)
            # ]
            # total_outer_steps = len(self.tar_dataloader_lst)  # the number of tasks (dataloaders)

            epoch_iter = zip(self.tar_dataloader, self.aux_dataloader)
            # outer_pbar = tqdm(epoch_iter, total=total_outer_steps, desc=f"Epoch {epoch+1}")
            # for outer_step, batch_tuple in enumerate(outer_pbar):
            #     inner_pbar = tqdm(batch_tuple, total=len(batch_tuple), desc="Processing batches", leave=False)
                # batch_tuple is a tuple of tuples: ((tar_batch1, aux_batch1), (tar_batch2, aux_batch2), ...)
            inner_pbar = tqdm(epoch_iter, total=len(self.tar_dataloader), desc=f"Epoch {epoch+1}")
            for inner_step, (tar_batch, aux_batch) in enumerate(inner_pbar):
                # Move to device
                tar_batch = {k: v.to(self.device) for k, v in tar_batch.items()} 
                aux_batch = {k: v.to(self.device) for k, v in aux_batch.items()}

                params = [p for p in self.model.parameters() if p.requires_grad]
                if len(params) == 0:
                    continue

                # Aux gradients: compute without touching p.grad (avoid being zeroed later)
                aux_loss = self.model(**aux_batch).loss
                aux_loss_scaled = aux_loss / max(1, self.grad_accum)
                aux_grads = torch.autograd.grad(
                    aux_loss_scaled,
                    params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )

                aux_norm_sq = None
                for g in aux_grads:
                    if g is None:
                        continue
                    g2 = (g.detach().float() ** 2).sum()
                    aux_norm_sq = g2 if aux_norm_sq is None else (aux_norm_sq + g2)
                aux_grads_norm = torch.sqrt(aux_norm_sq) if aux_norm_sq is not None else torch.tensor(0.0, device=self.device)

                # Target gradients: also compute via autograd.grad so we can orthogonalize per-microstep
                tar_loss = self.model(**tar_batch).loss
                tar_loss_scaled = tar_loss / max(1, self.grad_accum)
                tar_grads = torch.autograd.grad(
                    tar_loss_scaled,
                    params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )

                # Orthogonalize tar grads w.r.t aux grads, then accumulate into p.grad
                tar_ortho_norm_sq = None
                proj_g_norm_sq = None
                with torch.no_grad():
                    denom = aux_grads_norm + 1e-12
                    for p, tg, ag in zip(params, tar_grads, aux_grads):
                        if tg is None:
                            continue

                        tar_g = tg.detach()
                        if aux_grads_norm.item() == 0.0 or ag is None:
                            ortho_g = tar_g
                        else:
                            unit_aux = (ag.detach() / denom).to(dtype=tar_g.dtype)
                            proj = (tar_g * unit_aux).sum() * unit_aux
                            ortho_g = tar_g - proj
                            

                        # Rescale to preserve original tar grad norm (layerwise)
                        tar_norm = tar_g.float().norm()
                        ortho_norm = ortho_g.float().norm() + 1e-12
                        ortho_g = ortho_g * (tar_norm / ortho_norm)

                        if p.grad is None:
                            p.grad = ortho_g.clone()
                        else:
                            p.grad.add_(ortho_g)
                        
                        proj_g = (proj.float() ** 2).sum()
                        g2 = (ortho_g.float() ** 2).sum()
                        tar_ortho_norm_sq = g2 if tar_ortho_norm_sq is None else (tar_ortho_norm_sq + g2)
                        proj_g_norm_sq = proj_g if proj_g_norm_sq is None else (proj_g_norm_sq + proj_g)

                tar_grad_norm = torch.sqrt(tar_ortho_norm_sq) if tar_ortho_norm_sq is not None else torch.tensor(0.0, device=self.device)
                proj_g_norm = torch.sqrt(proj_g_norm_sq) if proj_g_norm_sq is not None else torch.tensor(0.0, device=self.device)

                inner_pbar.set_postfix(
                    tar_loss=f"{tar_loss.item():.4f}",
                    aux_loss=f"{aux_loss.item():.4f}",
                    aux_grad_norm=f"{aux_grads_norm.item():.4f}",
                    tar_grad_norm=f"{tar_grad_norm.item():.4f}",
                    proj_grad_norm=f"{proj_g_norm.item():.4f}",
                )

                if (inner_step + 1) % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad(set_to_none=True)
                    self.global_step += 1

                    tqdm.write(
                        f"Epoch {epoch+1}, Step {self.global_step}, Tar Loss: {tar_loss.item():.4f}, Aux Loss: {aux_loss.item():.4f}, Aux Grad Norm: {aux_grads_norm.item():.4f}, Tar Grad Norm: {tar_grad_norm.item():.4f}, Proj Grad Norm: {proj_g_norm.item():.4f}"
                    )
                    if wandb is not None and wandb.run is not None:
                        wandb.log(
                            {
                                "loss/tar": tar_loss.item(),
                                "loss/aux": aux_loss.item(),
                                "grad_norm/aux": aux_grads_norm.item(),
                                "grad_norm/tar": tar_grad_norm.item(),
                                "grad_norm/proj": proj_g_norm.item(),
                                "inner_step": inner_step,
                                # "outer_step": outer_step,
                                "epoch": epoch + 1,
                            },
                            step=self.global_step,
                        )
                    
                    if self.global_step >= self.num_training_steps:
                        # Save final model
                        self.save("final-model")
                        return
                    # if self.save_per_outer_step is not None and (outer_step + 1) % self.save_per_outer_step == 0:
                    #     self.save(f"checkpoint-epoch-{epoch+1}-outerstep-{outer_step+1}")

            # End of epoch
            self.save(f"checkpoint-epoch-{epoch+1}")
        return

class DoubleTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        dataloaders,
        eval_dataloader=None,
        lr=1e-5,
        num_training_steps=None,
        epochs=None,
        grad_accum=1,
        max_grad_norm=1.0,
        device=None,
        log_steps=1,
        out_dir=None,
        save_steps=None,
        eval_steps=100,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataloaders = dataloaders
        self.eval_dataloader = eval_dataloader
        self.lr = lr
        self.grad_accum = grad_accum
        self.max_grad_norm = max_grad_norm
        self.log_steps = log_steps
        if out_dir is None:
            out_dir = "./sft_checkpoints"
        self.out_dir = out_dir
        self.save_steps = save_steps    

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.train()

        self.opt = AdamW(self.model.parameters(), lr=self.lr)

        self.num_training_steps = num_training_steps
        self.epochs = epochs
        if self.epochs is None:
            raise ValueError("You must specify epochs for SFTtrainer.")
        if self.num_training_steps is None:
            self.num_training_steps = self.epochs * len(self.dataloaders[0]) // self.grad_accum

        self.lr_scheduler = get_scheduler(
            "cosine",
            optimizer=self.opt,
            num_warmup_steps=self.num_training_steps // 10,
            num_training_steps=self.num_training_steps,  # You might want to set this dynamically
        )

        self.global_step = 0
        if wandb is not None and wandb.run is not None:
            wandb.config.update(
                {
                    "lr": self.lr,
                    "grad_accum": self.grad_accum,
                    "max_grad_norm": self.max_grad_norm,
                    "log_steps": self.log_steps,
                    "save_steps": self.save_steps,
                    "epochs": self.epochs,
                    "num_training_steps": self.num_training_steps,
                }
            )
        
    def save(self, name):
        save_path = os.path.join(self.out_dir, name)
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

    def train(self):
        os.makedirs(self.out_dir, exist_ok=True)

        for epoch in range(self.epochs):
            dataloaders_iter = zip(*self.dataloaders)
            pbar = tqdm(dataloaders_iter, desc=f"Epoch {epoch+1}")
            for step, batches in enumerate(pbar):
                loss = 0.0
                loss_lst = []
                for batch in batches:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    loss += outputs.loss
                    loss_lst.append(outputs.loss.item())

                loss = loss / (self.grad_accum * len(batches))  # average over grad_accum and number of dataloaders
                loss.backward()

                pbar.set_postfix(loss=f"{loss.item() * self.grad_accum:.4f}")

                if (step + 1) % self.grad_accum == 0:
                    # get the grad norm
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()
                    self.global_step += 1

                    if self.global_step % self.log_steps == 0:
                        print(f"Epoch {epoch+1}, Step {self.global_step}, Loss: {loss.item() * self.grad_accum:.4f}, Loss_individual: {loss_lst}")
                        if wandb is not None and wandb.run is not None:
                            wandb.log(
                                {
                                    "loss/total": loss.item() * self.grad_accum,
                                    "loss/individual": loss_lst,
                                    "grad_norm": grad_norm.item(),
                                },
                                step=self.global_step,
                            )
                    
                    if self.eval_dataloader is not None and self.global_step % self.eval_steps == 0:
                        eval_pbar = tqdm(self.eval_dataloader, desc="Evaluating", leave=False)
                        eval_avg_loss = 0.0
                        for eval_step, eval_batch in enumerate(eval_pbar):
                            eval_batch = {k: v.to(self.device) for k, v in eval_batch.items()}
                            with torch.no_grad():
                                eval_outputs = self.model(**eval_batch)
                                eval_loss = eval_outputs.loss
                                eval_avg_loss += eval_loss.item()
                            eval_pbar.set_postfix(eval_loss=f"{eval_loss.item():.4f}")
                        # calculate average loss
                        eval_avg_loss /= (eval_step + 1)
                        print(f"Epoch {epoch+1}, Step {self.global_step}, Eval Loss: {eval_avg_loss:.4f}")
                        if wandb is not None and wandb.run is not None:
                            wandb.log(
                                {
                                    "eval/loss": eval_avg_loss,
                                },
                                step=self.global_step,
                            )

                    if self.save_steps is not None and self.global_step % self.save_steps == 0:
                        self.save(f"checkpoint-step-{self.global_step}")
                    
                    if self.global_step >= self.num_training_steps:
                        # Save final model
                        self.save("final-model")
                        return
            
            # End of epoch
            self.save(f"checkpoint-epoch-{epoch+1}")
        return      

class GATrainer(SAMTrainer):
    def __init__(
        self,
        model,
        model_name,
        tokenizer,
        harmful_dataloader=None,
        harmless_dataloader=None,
        util_dataloader=None,
        lr=1e-5,
        num_training_steps=None,
        epochs=None,
        grad_accum=1,
        max_grad_norm=1.0,
        device=None,
        log_steps=20,
        out_dir=None,
        save_steps=None,
        alpha=0.8,
    ):
        super().__init__(
            model=model,
            model_name=model_name,
            tokenizer=tokenizer,
            dataloader=harmless_dataloader,  # we will override the dataloader with paired one
            lr=lr,
            num_training_steps=num_training_steps,
            epochs=epochs,
            grad_accum=grad_accum,
            max_grad_norm=max_grad_norm,
            device=device,
            log_steps=log_steps,
            out_dir=out_dir,
            save_steps=save_steps,
            rho=0,
        )
        self.alpha = alpha
        self.safe_dataloader = harmless_dataloader
        self.unsafe_dataloader = harmful_dataloader
        self.util_dataloader = util_dataloader
        if self.safe_dataloader is None or self.unsafe_dataloader is None:
            raise ValueError(
                "BoosterTrainer requires paired dataloaders: harmful/harmless (or unsafe/safe)."
            )
        # Don't create a persistent zip iterator here; it would be exhausted after one epoch.
        self.lr_scheduler = get_scheduler(
            "constant",
            optimizer=self.opt,
        )
    
    def train(self):
        os.makedirs(self.out_dir, exist_ok=True)

        for epoch in range(self.epochs):
            total_steps = min(len(self.unsafe_dataloader), len(self.safe_dataloader), len(self.util_dataloader) if self.util_dataloader is not None else float('inf'))
            epoch_iter = zip(self.unsafe_dataloader, self.safe_dataloader, self.util_dataloader) if self.util_dataloader is not None else zip(self.unsafe_dataloader, self.safe_dataloader)
            pbar = tqdm(epoch_iter, total=total_steps, desc=f"Epoch {epoch+1}")
            for step, (unsafe_batch, safe_batch, util_batch) in enumerate(pbar) if self.util_dataloader is not None else enumerate(pbar):
                unsafe_batch = {k: v.to(self.device) for k, v in unsafe_batch.items()}
                safe_batch = {k: v.to(self.device) for k, v in safe_batch.items()}
                util_batch = {k: v.to(self.device) for k, v in util_batch.items()} if self.util_dataloader is not None else None
                # Losses at the original parameters (safe for backward)
                safe_loss_raw = self.model(**safe_batch).loss
                unsafe_loss_raw = self.model(**unsafe_batch).loss
                util_loss_raw = self.model(**util_batch).loss if self.util_dataloader is not None else None

                # IMPORTANT: only backprop through graphs built with current (unmodified) parameters
                loss = util_loss_raw + safe_loss_raw - self.alpha * torch.log(unsafe_loss_raw)
                loss = loss / self.grad_accum
                loss.backward()
                
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    sloss=f"{safe_loss_raw.item():.4f}",
                    hloss=f"{unsafe_loss_raw.item():.4f}",
                    uloss=f"{util_loss_raw.item():.4f}" if util_loss_raw is not None else None,
                )
                
                if (step + 1) % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()
                    self.global_step += 1

                    # if self.global_step % self.log_steps == 0:
                    tqdm.write(
                        f"Epoch {epoch+1}, Step {self.global_step}, Loss: {loss.item() * self.grad_accum:.4f}, Safe: {safe_loss_raw.item():.4f}, Unsafe: {unsafe_loss_raw.item():.4f}"
                    )
                    if wandb is not None and wandb.run is not None:
                        wandb.log(
                            {
                                "loss/total": loss.item() * self.grad_accum,
                                "loss/safe": safe_loss_raw.item(),
                                "loss/unsafe": unsafe_loss_raw.item(),
                                "loss/util": util_loss_raw.item() if util_loss_raw is not None else None,
                            },
                            step=self.global_step,
                        )

                    if self.save_steps is not None and self.global_step % self.save_steps == 0:
                        self.save(f"checkpoint-step-{self.global_step}")
                    
                    if self.global_step >= self.num_training_steps:
                        # Save final model
                        self.save("final-model")
                        return
            
            # End of epoch
            self.save(f"checkpoint-epoch-{epoch+1}")
        return

class NPOTrainer(GATrainer):
    def __init__(
        self,
        model,
        ref_model,
        model_name,
        tokenizer,
        harmful_dataloader=None, # for harmful loss 
        harmless_dataloader=None, # for retain loss
        lr=1e-5,
        num_training_steps=None,
        epochs=None,
        grad_accum=1,
        max_grad_norm=1.0,
        device=None,
        log_steps=20,
        out_dir=None,
        save_steps=None,
        alpha=0.8, # ratio between harmful loss and harmless loss
        beta=0.1,  # temperature of NPO
    ):
        super().__init__(
            model=model,
            model_name=model_name,
            tokenizer=tokenizer,
            harmful_dataloader=harmful_dataloader,
            harmless_dataloader=harmless_dataloader,
            lr=lr,
            num_training_steps=num_training_steps,
            epochs=epochs,
            grad_accum=grad_accum,
            max_grad_norm=max_grad_norm,
            device=device,
            log_steps=log_steps,
            out_dir=out_dir,
            save_steps=save_steps,
            alpha=alpha,  # weight for safe loss
        )
        self.beta = beta  # weight for similarity loss
        self.ref_model = ref_model
        self.ref_model.eval()  # reference model is fixed; no gradients needed
    
    def train(self):
        os.makedirs(self.out_dir, exist_ok=True)

        for epoch in range(self.epochs):
            total_steps = min(len(self.unsafe_dataloader), len(self.safe_dataloader))
            epoch_iter = zip(self.unsafe_dataloader, self.safe_dataloader)
            pbar = tqdm(epoch_iter, total=total_steps, desc=f"Epoch {epoch+1}")
            for step, (unsafe_batch, safe_batch) in enumerate(pbar):
                unsafe_batch = {k: v.to(self.device) for k, v in unsafe_batch.items()}
                safe_batch = {k: v.to(self.device) for k, v in safe_batch.items()}

                # NPO loss for unsafe batch
                ref_outputs = self.ref_model(**unsafe_batch)
                ref_loss = ref_outputs.loss
                cur_outputs = self.model(**unsafe_batch)
                cur_loss = cur_outputs.loss
                npo_loss = (-2.0 / self.beta) * torch.log(torch.sigmoid(-self.beta * (cur_loss - ref_loss)))

                # retain loss
                retain_loss = self.model(**safe_batch).loss

                # overall loss
                loss = npo_loss + self.alpha * retain_loss
                loss = loss / self.grad_accum
                loss.backward()

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    npo_loss=f"{npo_loss.item():.4f}",
                    retain_loss=f"{retain_loss.item():.4f}",
                )

                if (step + 1) % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()
                    self.global_step += 1

                    # if self.global_step % self.log_steps == 0:
                    tqdm.write(
                        f"Epoch {epoch+1}, Step {self.global_step}, Loss: {loss.item() * self.grad_accum:.4f}, NPO Loss: {npo_loss.item():.4f}, Retain Loss: {retain_loss.item():.4f}"
                    )
                    if wandb is not None and wandb.run is not None:
                        wandb.log(
                            {
                                "loss/total": loss.item() * self.grad_accum,
                                "loss/npo": npo_loss.item(),
                                "loss/retain": retain_loss.item(),
                            },
                            step=self.global_step,
                        )

                    if self.save_steps is not None and self.global_step % self.save_steps == 0:
                        self.save(f"checkpoint-step-{self.global_step}")
                    
                    if self.global_step >= self.num_training_steps:
                        # Save final model
                        self.save("final-model")
                        return
            
            # End of epoch
            self.save(f"checkpoint-epoch-{epoch+1}")
        return

class BoosterDualTrainer(SAMTrainer):
    def __init__(
        self,
        model,
        model_name,
        tokenizer,
        harmful_dataloader=None,
        harmless_dataloader=None,
        lr=1e-5,
        num_training_steps=None,
        epochs=None,
        grad_accum=1,
        max_grad_norm=1.0,
        device=None,
        log_steps=20,
        out_dir=None,
        save_steps=None,
        alpha=0.8,
        rho=0.05,
        lam=0.01,
        target=0.5,
    ):
        super().__init__(
            model=model,
            model_name=model_name,
            tokenizer=tokenizer,
            dataloader=harmless_dataloader,  # we will override the dataloader with paired one
            lr=lr,
            num_training_steps=num_training_steps,
            epochs=epochs,
            grad_accum=grad_accum,
            max_grad_norm=max_grad_norm,
            device=device,
            log_steps=log_steps,
            out_dir=out_dir,
            save_steps=save_steps,
            rho=rho,
        )
        self.alpha = alpha
        self.lam = lam
        self.safe_dataloader = harmless_dataloader
        self.unsafe_dataloader = harmful_dataloader
        self.target = target
        if self.safe_dataloader is None or self.unsafe_dataloader is None:
            raise ValueError(
                "BoosterTrainer requires paired dataloaders: harmful/harmless (or unsafe/safe)."
            )
        # Don't create a persistent zip iterator here; it would be exhausted after one epoch.
    
    def train(self):
        os.makedirs(self.out_dir, exist_ok=True)

        for epoch in range(self.epochs):
            set_dataloader_epoch(self.unsafe_dataloader, epoch)
            set_dataloader_epoch(self.safe_dataloader, epoch)
            total_steps = min(len(self.unsafe_dataloader), len(self.safe_dataloader))
            epoch_iter = zip(self.unsafe_dataloader, self.safe_dataloader)
            pbar = tqdm(epoch_iter, total=total_steps, desc=f"Epoch {epoch+1}", disable=not is_main_process())
            for step, (unsafe_batch, safe_batch) in enumerate(pbar):
                unsafe_batch = {k: v.to(self.device) for k, v in unsafe_batch.items()}
                safe_batch = {k: v.to(self.device) for k, v in safe_batch.items()}

                # --- SAM-style perturbation direction from unsafe loss (no inplace param edits) ---
                unsafe_loss_for_grad = reduce_loss(self.model(**unsafe_batch).loss)
                trainable_named_params = [
                    (name, p) for name, p in self.stateless_model.named_parameters() if p.requires_grad
                ]
                params = [p for _, p in trainable_named_params]
                grads = torch.autograd.grad(
                    unsafe_loss_for_grad,
                    params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )

                with torch.no_grad():
                    global_norm_sq = None
                    for g in grads:
                        if g is None:
                            continue
                        g2 = (g.detach().float() ** 2).sum()
                        global_norm_sq = g2 if global_norm_sq is None else (global_norm_sq + g2)
                    if global_norm_sq is None:
                        scale = None
                    else:
                        global_norm = torch.sqrt(global_norm_sq)
                        scale = self.rho / (global_norm + 1e-12) if global_norm.item() != 0.0 else None

                # Compute perturbed unsafe loss using functional_call to avoid inplace modifications
                param_and_buffer_dict = {name: p for name, p in self.stateless_model.named_parameters()}
                param_and_buffer_dict.update({name: b for name, b in self.stateless_model.named_buffers()})
                perturbed_params = []
                if scale is not None:
                    for (name, p), g in zip(trainable_named_params, grads):
                        if g is None:
                            perturbed_params.append(p)
                            continue
                        perturb = g.detach().to(dtype=p.dtype) * scale.to(dtype=p.dtype)
                        param_and_buffer_dict[name] = p - perturb
                        perturbed_params.append(param_and_buffer_dict[name])
                else:
                    perturbed_params = params
                # unsafe_loss_perturbed = _functional_call(self.model, param_and_buffer_dict, (), unsafe_batch).loss
                unsafe_loss_perturbed = reduce_loss(
                    _functional_call(self.stateless_model, param_and_buffer_dict, (), safe_batch).loss
                )

                # optimize the perturbation coefficent rho
                perturbed_grads = torch.autograd.grad(
                    unsafe_loss_perturbed,
                    perturbed_params,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )

                safe_loss_raw = reduce_loss(self.model(**safe_batch).loss)
                # get the unsafe loss for optimization
                with torch.no_grad():
                    rho_grad = None
                    for pg, g in zip(perturbed_grads, grads):
                        if pg is None or g is None:
                            continue
                        grad_dot = (pg.detach() * g.detach()).sum()
                        rho_grad = grad_dot if rho_grad is None else (rho_grad + grad_dot)
                    loss_diff = unsafe_loss_perturbed - safe_loss_raw - self.target
                    valid = unsafe_loss_perturbed.item() > safe_loss_raw.item()  # only update rho when perturbed loss is worse than original loss
                    if valid and rho_grad is not None:
                        rho_grad = -rho_grad / (global_norm + 1e-12) if global_norm.item() != 0.0 else None
                        self.rho -= self.lam * loss_diff.sign().item() 
                        self.rho = max(0.0, self.rho)  # ensure rho is non-negative

                # Losses at the original parameters (safe for backward)
                unsafe_loss_raw = reduce_loss(self.model(**unsafe_batch).loss)

                # IMPORTANT: only backprop through graphs built with current (unmodified) parameters
                # loss = safe_loss_raw - torch.log(
                #     (1 - self.alpha) * unsafe_loss_raw + self.alpha * unsafe_loss_perturbed
                # )
                loss = safe_loss_raw + self.alpha * torch.clamp_min(unsafe_loss_perturbed - safe_loss_raw, 0.0) - torch.log(
                    unsafe_loss_raw
                )
                # loss = (1 - self.alpha) * safe_loss_raw + self.alpha * unsafe_loss_perturbed
                loss = loss / self.grad_accum
                loss.backward()
                
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    sloss=f"{safe_loss_raw.item():.4f}",
                    hloss=f"{unsafe_loss_raw.item():.4f}",
                    ploss=f"{unsafe_loss_perturbed.item():.4f}",
                    rho=f"{self.rho:.4f}",
                )
                
                if (step + 1) % self.grad_accum == 0:
                    clip_grad_norm(self.model, self.max_grad_norm)
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()
                    self.global_step += 1

                    # if self.global_step % self.log_steps == 0:
                    if is_main_process():
                        tqdm.write(
                            f"Epoch {epoch+1}, Step {self.global_step}, Loss: {loss.item() * self.grad_accum:.4f}, Safe: {safe_loss_raw.item():.4f}, Unsafe: {unsafe_loss_raw.item():.4f}, PerturbedUnsafe: {unsafe_loss_perturbed.item():.4f}"
                        )
                    if wandb is not None and wandb.run is not None:
                        wandb.log(
                            {
                                "loss/total": loss.item() * self.grad_accum,
                                "loss/safe": safe_loss_raw.item(),
                                "loss/unsafe": unsafe_loss_raw.item(),
                                "loss/perturbed_unsafe": unsafe_loss_perturbed.item(),
                                "rho": self.rho,
                            },
                            step=self.global_step,
                        )

                    if self.save_steps is not None and self.global_step % self.save_steps == 0:
                        self.save(f"checkpoint-step-{self.global_step}")
                    
                    if self.global_step >= self.num_training_steps:
                        # Save final model
                        self.save("final-model")
                        return
            
            # End of epoch
            self.save(f"checkpoint-epoch-{epoch+1}")
        return

class HarmfulBoosterTrainer(SAMTrainer):
    def __init__(
        self,
        model,
        model_name,
        tokenizer,
        harmful_dataloader=None,
        harmless_dataloader=None,
        lr=1e-5,
        num_training_steps=None,
        epochs=None,
        grad_accum=1,
        max_grad_norm=1.0,
        device=None,
        log_steps=20,
        out_dir=None,
        save_steps=None,
        save_epochs=None,
        alpha=0.8,
        rho=0.05,
    ):
        super().__init__(
            model=model,
            model_name=model_name,
            tokenizer=tokenizer,
            dataloader=harmless_dataloader,  # we will override the dataloader with paired one
            lr=lr,
            num_training_steps=num_training_steps,
            epochs=epochs,
            grad_accum=grad_accum,
            max_grad_norm=max_grad_norm,
            device=device,
            log_steps=log_steps,
            out_dir=out_dir,
            save_steps=save_steps,
            rho=rho,
        )
        self.alpha = alpha
        self.safe_dataloader = harmless_dataloader
        self.unsafe_dataloader = harmful_dataloader
        if self.safe_dataloader is None or self.unsafe_dataloader is None:
            raise ValueError(
                "BoosterTrainer requires paired dataloaders: harmful/harmless (or unsafe/safe)."
            )
        # Don't create a persistent zip iterator here; it would be exhausted after one epoch.
        self.lr_scheduler = get_scheduler(
            "constant",
            optimizer=self.opt,
        )
        self.save_epochs = save_epochs
    
    def train(self):
        os.makedirs(self.out_dir, exist_ok=True)

        for epoch in range(self.epochs):
            set_dataloader_epoch(self.unsafe_dataloader, epoch)
            set_dataloader_epoch(self.safe_dataloader, epoch)
            total_steps = min(len(self.unsafe_dataloader), len(self.safe_dataloader))
            epoch_iter = zip(self.unsafe_dataloader, self.safe_dataloader)
            pbar = tqdm(epoch_iter, total=total_steps, desc=f"Epoch {epoch+1}", disable=not is_main_process())
            for step, (unsafe_batch, safe_batch) in enumerate(pbar):
                unsafe_batch = {k: v.to(self.device) for k, v in unsafe_batch.items()}
                safe_batch = {k: v.to(self.device) for k, v in safe_batch.items()}

                # --- SAM-style perturbation direction from unsafe loss (no inplace param edits) ---
                unsafe_loss_for_grad = reduce_loss(self.model(**unsafe_batch).loss)
                trainable_named_params = [
                    (name, p) for name, p in self.stateless_model.named_parameters() if p.requires_grad
                ]
                params = [p for _, p in trainable_named_params]
                grads = torch.autograd.grad(
                    unsafe_loss_for_grad,
                    params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )

                with torch.no_grad():
                    global_norm_sq = None
                    for g in grads:
                        if g is None:
                            continue
                        g2 = (g.detach().float() ** 2).sum()
                        global_norm_sq = g2 if global_norm_sq is None else (global_norm_sq + g2)
                    if global_norm_sq is None:
                        scale = None
                    else:
                        global_norm = torch.sqrt(global_norm_sq)
                        scale = self.rho / (global_norm + 1e-12) if global_norm.item() != 0.0 else None
                        # scale = self.rho

                # Compute perturbed unsafe loss using functional_call to avoid inplace modifications
                param_and_buffer_dict = {name: p for name, p in self.stateless_model.named_parameters()}
                param_and_buffer_dict.update({name: b for name, b in self.stateless_model.named_buffers()})
                if scale is not None:
                    for (name, p), g in zip(trainable_named_params, grads):
                        if g is None:
                            continue
                        perturb = g.detach().to(dtype=p.dtype) * scale
                        param_and_buffer_dict[name] = p - perturb
                # unsafe_loss_perturbed = _functional_call(self.model, param_and_buffer_dict, (), unsafe_batch).loss
                unsafe_loss_perturbed = reduce_loss(
                    _functional_call(self.stateless_model, param_and_buffer_dict, (), unsafe_batch).loss
                )
                # test: use weighted loss for each token
                # unsafe_logits_perturbed = _functional_call(self.model, param_and_buffer_dict, (), safe_batch).logits
                # unsafe_loss_perturbed = weighted_ce_loss(unsafe_logits_perturbed, safe_batch["labels"])

                # Losses at the original parameters (safe for backward)
                safe_loss_raw = reduce_loss(self.model(**safe_batch).loss)
                # safe_loss_raw = weighted_ce_loss(self.model(**safe_batch).logits, safe_batch["labels"])
                unsafe_loss_raw = reduce_loss(self.model(**unsafe_batch).loss)

                # IMPORTANT: only backprop through graphs built with current (unmodified) parameters
                # loss = safe_loss_raw - torch.log(
                #     (1 - self.alpha) * unsafe_loss_raw + self.alpha * unsafe_loss_perturbed
                # )
                loss = safe_loss_raw + self.alpha * (unsafe_loss_raw - unsafe_loss_perturbed)
                # loss = (1 - self.alpha) * safe_loss_raw + self.alpha * unsafe_loss_perturbed
                loss = loss / self.grad_accum
                loss.backward()
                
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    sloss=f"{safe_loss_raw.item():.4f}",
                    hloss=f"{unsafe_loss_raw.item():.4f}",
                    ploss=f"{unsafe_loss_perturbed.item():.4f}",
                )
                
                if (step + 1) % self.grad_accum == 0:
                    clip_grad_norm(self.model, self.max_grad_norm)
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()
                    self.global_step += 1

                    # if self.global_step % self.log_steps == 0:
                    if is_main_process():
                        tqdm.write(
                            f"Epoch {epoch+1}, Step {self.global_step}, Loss: {loss.item() * self.grad_accum:.4f}, Safe: {safe_loss_raw.item():.4f}, Unsafe: {unsafe_loss_raw.item():.4f}, PerturbedUnsafe: {unsafe_loss_perturbed.item():.4f}"
                        )
                    if wandb is not None and wandb.run is not None:
                        wandb.log(
                            {
                                "loss/total": loss.item() * self.grad_accum,
                                "loss/safe": safe_loss_raw.item(),
                                "loss/unsafe": unsafe_loss_raw.item(),
                                "loss/perturbed_unsafe": unsafe_loss_perturbed.item(),
                            },
                            step=self.global_step,
                        )

                    if self.save_steps is not None and self.global_step % self.save_steps == 0:
                        self.save(f"checkpoint-step-{self.global_step}")
                    
                    if self.global_step >= self.num_training_steps:
                        # Save final model
                        self.save("final-model")
                        return
            
            # End of epoch
            if self.save_epochs is not None and (epoch + 1) % self.save_epochs == 0:
                self.save(f"checkpoint-epoch-{epoch+1}")
        return


class LISATrainer(SAMTrainer):
    def __init__(
        self,
        model,
        model_name,
        tokenizer,
        harmful_dataloader=None,
        harmless_dataloader=None,
        lr=1e-5,
        num_training_steps=None,
        epochs=None,
        grad_accum=1,
        max_grad_norm=1.0,
        device=None,
        log_steps=20,
        out_dir=None,
        save_steps=None,
        save_epochs=None,
        rho=0.0,
        alignment_steps=1,
        finetune_steps=1,
    ):
        base_dataloader = harmless_dataloader if harmless_dataloader is not None else harmful_dataloader
        super().__init__(
            model=model,
            model_name=model_name,
            tokenizer=tokenizer,
            dataloader=base_dataloader,
            lr=lr,
            num_training_steps=num_training_steps,
            epochs=epochs,
            grad_accum=grad_accum,
            max_grad_norm=max_grad_norm,
            device=device,
            log_steps=log_steps,
            out_dir=out_dir,
            save_steps=save_steps,
            rho=rho,
        )
        self.safe_dataloader = harmless_dataloader
        self.unsafe_dataloader = harmful_dataloader
        self.alignment_steps = alignment_steps
        self.finetune_steps = finetune_steps
        self.save_epochs = save_epochs

        if self.alignment_steps < 0 or self.finetune_steps < 0:
            raise ValueError("alignment_steps and finetune_steps must be non-negative.")
        if self.alignment_steps == 0 and self.finetune_steps == 0:
            raise ValueError("At least one of alignment_steps or finetune_steps must be positive.")
        if self.alignment_steps > 0 and self.safe_dataloader is None:
            raise ValueError("LISATrainer requires harmless_dataloader when alignment_steps > 0.")
        if self.finetune_steps > 0 and self.unsafe_dataloader is None:
            raise ValueError("LISATrainer requires harmful_dataloader when finetune_steps > 0.")

        self.cycles_per_epoch = self._compute_cycles_per_epoch()
        self.steps_per_epoch = self.cycles_per_epoch * (self.alignment_steps + self.finetune_steps)
        if num_training_steps is None:
            self.num_training_steps = self.epochs * self.steps_per_epoch
        self.lr_scheduler = get_scheduler(
            "constant",
            optimizer=self.opt,
        )

        if wandb is not None and wandb.run is not None:
            safe_bs = getattr(self.safe_dataloader, "batch_size", None)
            unsafe_bs = getattr(self.unsafe_dataloader, "batch_size", None)
            wandb.config.update(
                {
                    "lr": self.lr,
                    "epochs": self.epochs,
                    "num_training_steps": self.num_training_steps,
                    "grad_accum": self.grad_accum,
                    "rho": self.rho,
                    "alignment_steps": self.alignment_steps,
                    "finetune_steps": self.finetune_steps,
                    "cycles_per_epoch": self.cycles_per_epoch,
                    "safe_batch_size": safe_bs,
                    "unsafe_batch_size": unsafe_bs,
                }
            )

    def _compute_cycles_per_epoch(self):
        cycle_limits = []
        if self.alignment_steps > 0:
            safe_batches_per_cycle = max(1, self.alignment_steps * self.grad_accum)
            safe_cycles = len(self.safe_dataloader) // safe_batches_per_cycle
            cycle_limits.append(max(1, safe_cycles))
        if self.finetune_steps > 0:
            unsafe_batches_per_cycle = max(1, self.finetune_steps * self.grad_accum)
            unsafe_cycles = len(self.unsafe_dataloader) // unsafe_batches_per_cycle
            cycle_limits.append(max(1, unsafe_cycles))
        return max(1, min(cycle_limits)) if cycle_limits else 1

    def _clone_trainable_params(self):
        return {
            name: param.detach().clone()
            for name, param in self.stateless_model.named_parameters()
            if param.requires_grad
        }

    def _compute_prox_loss(self, reference_weights):
        if self.rho is None or self.rho <= 0 or not reference_weights:
            return torch.tensor(0.0, device=self.device)

        prox_loss = None
        for name, param in self.stateless_model.named_parameters():
            if not param.requires_grad:
                continue
            reference = reference_weights.get(name)
            if reference is None:
                continue
            diff_sq = (param.float() - reference.float()).pow(2).sum()
            prox_loss = diff_sq if prox_loss is None else (prox_loss + diff_sq)

        if prox_loss is None:
            return torch.tensor(0.0, device=self.device)
        return 0.5 * float(self.rho) * prox_loss

    def _next_batch(self, iterator, dataloader):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        return batch, iterator

    def _run_state_step(self, iterator, dataloader, state_name, reference_weights):
        ce_loss_sum = 0.0
        prox_loss_sum = 0.0
        total_loss_sum = 0.0

        for _ in range(self.grad_accum):
            batch, iterator = self._next_batch(iterator, dataloader)
            batch = {k: v.to(self.device) for k, v in batch.items()}

            ce_loss = reduce_loss(self.model(**batch).loss)
            prox_loss = self._compute_prox_loss(reference_weights)
            total_loss = ce_loss + prox_loss.to(dtype=ce_loss.dtype)

            (total_loss / self.grad_accum).backward()

            ce_loss_sum += ce_loss.detach().item()
            prox_loss_sum += prox_loss.detach().item()
            total_loss_sum += total_loss.detach().item()

        grad_norm = clip_grad_norm(self.model, self.max_grad_norm)
        self.opt.step()
        self.lr_scheduler.step()
        self.opt.zero_grad()
        self.global_step += 1

        ce_loss_avg = ce_loss_sum / self.grad_accum
        prox_loss_avg = prox_loss_sum / self.grad_accum
        total_loss_avg = total_loss_sum / self.grad_accum

        if self.global_step % self.log_steps == 0 and is_main_process():
            tqdm.write(
                f"Step {self.global_step} [{state_name}] Loss: {total_loss_avg:.4f}, "
                f"CE: {ce_loss_avg:.4f}, Prox: {prox_loss_avg:.4f}"
            )

        if wandb is not None and wandb.run is not None:
            wandb.log(
                {
                    "lisa/loss_total": total_loss_avg,
                    "lisa/loss_ce": ce_loss_avg,
                    "lisa/loss_prox": prox_loss_avg,
                    "lisa/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                    "lisa/is_alignment": 1 if state_name == "alignment" else 0,
                },
                step=self.global_step,
            )

        if self.save_steps is not None and self.global_step % self.save_steps == 0:
            self.save(f"checkpoint-step-{self.global_step}")

        return iterator, total_loss_avg, ce_loss_avg, prox_loss_avg

    def train(self):
        os.makedirs(self.out_dir, exist_ok=True)

        for epoch in range(self.epochs):
            if self.safe_dataloader is not None:
                set_dataloader_epoch(self.safe_dataloader, epoch)
                safe_iterator = iter(self.safe_dataloader)
            else:
                safe_iterator = None
            if self.unsafe_dataloader is not None:
                set_dataloader_epoch(self.unsafe_dataloader, epoch)
                unsafe_iterator = iter(self.unsafe_dataloader)
            else:
                unsafe_iterator = None

            pbar = tqdm(
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch+1}",
                disable=not is_main_process(),
            )

            for _ in range(self.cycles_per_epoch):
                if self.alignment_steps > 0:
                    alignment_reference = self._clone_trainable_params()
                    for _ in range(self.alignment_steps):
                        safe_iterator, total_loss, ce_loss, prox_loss = self._run_state_step(
                            safe_iterator,
                            self.safe_dataloader,
                            "alignment",
                            alignment_reference,
                        )
                        pbar.update(1)
                        pbar.set_postfix(
                            state="align",
                            loss=f"{total_loss:.4f}",
                            ce=f"{ce_loss:.4f}",
                            prox=f"{prox_loss:.4f}",
                        )
                        if self.global_step >= self.num_training_steps:
                            self.save("final-model")
                            pbar.close()
                            return

                if self.finetune_steps > 0:
                    finetune_reference = self._clone_trainable_params()
                    for _ in range(self.finetune_steps):
                        unsafe_iterator, total_loss, ce_loss, prox_loss = self._run_state_step(
                            unsafe_iterator,
                            self.unsafe_dataloader,
                            "finetune",
                            finetune_reference,
                        )
                        pbar.update(1)
                        pbar.set_postfix(
                            state="ft",
                            loss=f"{total_loss:.4f}",
                            ce=f"{ce_loss:.4f}",
                            prox=f"{prox_loss:.4f}",
                        )
                        if self.global_step >= self.num_training_steps:
                            self.save("final-model")
                            pbar.close()
                            return

            pbar.close()

            if self.save_epochs is not None and (epoch + 1) % self.save_epochs == 0:
                self.save(f"checkpoint-epoch-{epoch+1}")
        self.save("final-model")
        return


class TARTrainer:
    def __init__(
        self,
        model,
        model_name,
        tokenizer,
        dataloaders,
        ref_model=None,
        retain_model=None,
        lr=2e-5,
        num_training_steps=None,
        epochs=None,
        grad_accum=1,
        max_grad_norm=1.0,
        device=None,
        log_steps=1,
        out_dir=None,
        save_steps=None,
        save_epochs=None,
        tar_inner_loop_steps=4,
        tar_tamper_resistance_loss_lower_bound=-11.76,
        tar_tamper_resistance_grad_scale=4.0,
        tar_tamper_resistance_loss_type="max_entropy",
        schedule_lambda=0.5,
        inner_optimizer_warmup_steps=20,
        unbounded=False,
        use_weighting_schedule=False,
        adversary_dist_types="forget_train:1.0",
        adversary_lr_schedulers="constant:1.0",
        tar_num_tasks_sampled=1,
        adversary_lr_samples="2e-5,4e-5,1e-4",
        tar_inner_loop_subsample=1,
        tar_retain_scale=1.0,
        retain_representations=False,
        switching_point_coeffs="alpha:6.0,beta:3.0",
        dpo_beta=0.1,
    ):
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.dataloaders = dataloaders
        self.retain_dataloader = dataloaders["retain"]
        self.meta_dataloader = dataloaders["meta"]
        self.adversary_dataloaders = {k: v for k, v in dataloaders.items() if k not in ["retain", "meta"]}
        self.lr = lr
        self.grad_accum = grad_accum
        self.max_grad_norm = max_grad_norm
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_steps = log_steps
        self.out_dir = out_dir or "./tar_checkpoints"
        self.save_steps = save_steps
        self.save_epochs = save_epochs
        self.tar_inner_loop_steps = tar_inner_loop_steps
        self.tar_tamper_resistance_loss_lower_bound = tar_tamper_resistance_loss_lower_bound
        self.tar_tamper_resistance_grad_scale = tar_tamper_resistance_grad_scale
        self.tar_tamper_resistance_loss_type = tar_tamper_resistance_loss_type
        self.schedule_lambda = schedule_lambda
        self.inner_optimizer_warmup_steps = inner_optimizer_warmup_steps
        self.unbounded = unbounded
        self.use_weighting_schedule = use_weighting_schedule
        self.adversary_dist_types = adversary_dist_types
        self.adversary_lr_schedulers = adversary_lr_schedulers
        self.tar_num_tasks_sampled = tar_num_tasks_sampled
        self.adversary_lr_samples = adversary_lr_samples
        self.tar_inner_loop_subsample = tar_inner_loop_subsample
        self.tar_retain_scale = tar_retain_scale
        self.retain_representations = retain_representations
        self.switching_point_coeffs = switching_point_coeffs
        self.dpo_beta = dpo_beta

        self.raw_model = unwrap_model(self.model)
        if not is_fsdp_model(self.model):
            self.raw_model.to(self.device)
        self.model.train()
        self.opt_model = get_optimizer_model(self.model, self.raw_model)
        self.opt = AdamW(self.opt_model.parameters(), lr=self.lr)
        self.global_step = 0

        self.epochs = epochs
        if num_training_steps is None:
            if self.epochs is None:
                raise ValueError("Specify either num_training_steps or epochs for TARTrainer.")
            steps_per_epoch = max(1, len(self.retain_dataloader) // max(1, self.grad_accum))
            self.num_training_steps = self.epochs * steps_per_epoch
        else:
            self.num_training_steps = num_training_steps

        self.ref_model = ref_model
        if self.tar_tamper_resistance_loss_type == "dpo":
            if self.ref_model is None:
                self.ref_model = copy.deepcopy(self.raw_model)
            self.ref_model.to(self.device)
            self.ref_model.eval()

        self.retain_model = retain_model
        if self.retain_representations:
            if self.retain_model is None:
                self.retain_model = copy.deepcopy(self.raw_model)
            self.retain_model.to(self.device)
            self.retain_model.eval()

        if wandb is not None and wandb.run is not None:
            wandb.config.update(
                {
                    "lr": self.lr,
                    "num_training_steps": self.num_training_steps,
                    "tar_inner_loop_steps": self.tar_inner_loop_steps,
                    "tar_tamper_resistance_grad_scale": self.tar_tamper_resistance_grad_scale,
                    "tar_tamper_resistance_loss_type": self.tar_tamper_resistance_loss_type,
                    "tar_retain_scale": self.tar_retain_scale,
                    "adversary_dist_types": self.adversary_dist_types,
                    "adversary_lr_schedulers": self.adversary_lr_schedulers,
                    "adversary_lr_samples": self.adversary_lr_samples,
                    "tar_num_tasks_sampled": self.tar_num_tasks_sampled,
                    "tar_inner_loop_subsample": self.tar_inner_loop_subsample,
                    "dpo_beta": self.dpo_beta,
                }
            )

    def save(self, name):
        save_path = os.path.join(self.out_dir, name)
        save_model_and_tokenizer(self.model, self.raw_model, self.tokenizer, save_path)

    def tamper_resistance_obj(self, batches, scale: float):
        total_loss = 0.0
        total_diag = 0.0
        diag_name = "next_token"
        for i in range(self.grad_accum):
            batch = move_batch_to_device(batches[i], self.device)
            if self.tar_tamper_resistance_loss_type == "max_entropy":
                outputs = self.model(**_forward_inputs(batch), output_hidden_states=False)
                loss = max_entropy_loss_from_logits(outputs.logits) * scale
                diag = log_p_loss_from_logits(outputs.logits, batch["labels"]).item() / self.grad_accum
                (loss / self.grad_accum).backward()
                total_loss += (loss / self.grad_accum).item()
                total_diag += diag
            else:
                diag_name = "reward_accs"
                chosen_batch = _filter_dpo_inputs(batch, chosen=True)
                rejected_batch = _filter_dpo_inputs(batch, chosen=False)
                policy_chosen_logits = self.model(**_forward_inputs(chosen_batch), output_hidden_states=False).logits
                policy_rejected_logits = self.model(**_forward_inputs(rejected_batch), output_hidden_states=False).logits
                reference_chosen_logps = batch.get("reference_chosen_logps")
                reference_rejected_logps = batch.get("reference_rejected_logps")
                if reference_chosen_logps is not None:
                    reference_chosen_logps = reference_chosen_logps.to(self.device)
                    reference_rejected_logps = reference_rejected_logps.to(self.device)
                else:
                    with torch.no_grad():
                        ref_chosen_logits = self.ref_model(**_forward_inputs(chosen_batch), output_hidden_states=False).logits
                        ref_rejected_logits = self.ref_model(**_forward_inputs(rejected_batch), output_hidden_states=False).logits
                        reference_chosen_logps = sequence_logps_from_logits(ref_chosen_logits, chosen_batch["labels"])
                        reference_rejected_logps = sequence_logps_from_logits(ref_rejected_logits, rejected_batch["labels"])
                loss, reward_acc = dpo_loss_from_logits(
                    policy_chosen_logits,
                    chosen_batch["labels"],
                    policy_rejected_logits,
                    rejected_batch["labels"],
                    reference_chosen_logps=reference_chosen_logps,
                    reference_rejected_logps=reference_rejected_logps,
                    beta=self.dpo_beta,
                )
                scaled_loss = loss * scale / self.grad_accum
                scaled_loss.backward()
                total_loss += scaled_loss.item()
                total_diag += reward_acc.item() / self.grad_accum
        if wandb is not None and wandb.run is not None and is_main_process():
            wandb.log(
                {
                    f"tr_{self.tar_tamper_resistance_loss_type}_loss": total_loss / max(scale, 1e-12),
                    f"tr_{diag_name}_loss": total_diag,
                },
                step=self.global_step,
            )
        return total_loss

    def adversary_next_token_obj_step(self, adversary_batches, sub_pbar=None):
        total_loss = 0.0
        for i in range(self.grad_accum):
            batch = move_batch_to_device(adversary_batches[i], self.device)
            loss = obj_standard_max_next_token(self.model, batch) / self.grad_accum
            loss.backward()
            total_loss += loss.item()
        if sub_pbar is not None:
            sub_pbar.update(1)
            sub_pbar.set_postfix({"inner loss": total_loss})
        if wandb is not None and wandb.run is not None and is_main_process():
            wandb.log({"inner_next_token_loss": total_loss}, step=self.global_step)
        return total_loss

    def inner_loop_step(
        self,
        adversary_batches,
        meta_forget_batches,
        inner_optimizer,
        inner_scheduler,
        model_storage,
        sub_pbar,
        meta_grad_scale,
        compute_tamper_resistance_grad,
    ):
        self.adversary_next_token_obj_step(adversary_batches, sub_pbar=sub_pbar)
        inner_optimizer.step()
        if inner_scheduler is not None:
            inner_scheduler.step()
        self.model.zero_grad(set_to_none=True)

        tamper_resistance_loss = 0.0
        if compute_tamper_resistance_grad:
            tamper_resistance_loss = self.tamper_resistance_obj(
                meta_forget_batches,
                scale=meta_grad_scale,
            )
            model_storage.collect_param_or_grad(self.model, to_cpu=False, mode="grads")
            self.model.zero_grad(set_to_none=False)
        return tamper_resistance_loss

    def train(self):
        os.makedirs(self.out_dir, exist_ok=True)
        retain_iterator = iter(self.retain_dataloader)
        meta_iterator = iter(self.meta_dataloader)
        adversary_iterators = {
            key: {"iter": iter(value), "dataloader": value}
            for key, value in self.adversary_dataloaders.items()
        }

        storage = ModelStorage()

        if is_fsdp_model(self.model):
            init_batch, retain_iterator = get_next_batch(retain_iterator, self.retain_dataloader)
            init_batch = move_batch_to_device(init_batch, self.device)
            obj_standard_max_next_token(self.model, init_batch).backward()
            self.model.zero_grad(set_to_none=False)

        pbar = tqdm(
            colour="blue",
            desc="Outer Training Loop",
            total=self.num_training_steps,
            dynamic_ncols=True,
            disable=not is_main_process(),
        )

        for _ in range(self.num_training_steps):
            tamper_resistance_loss = 0.0
            storage.collect_param_or_grad(self.model, to_cpu=True, mode="params")

            for _task_idx in range(self.tar_num_tasks_sampled):
                adversary_type = distributed_sample_task(self.adversary_dist_types)
                sub_pbar = None
                if is_main_process():
                    sub_pbar = tqdm(
                        colour="blue",
                        desc=f"Inner Training Loop ({adversary_type})",
                        total=self.tar_inner_loop_steps,
                        dynamic_ncols=True,
                        leave=False,
                    )

                meta_forget_batches, meta_iterator = next_n_batches(
                    meta_iterator,
                    self.meta_dataloader,
                    self.grad_accum,
                )

                adversary_lr = distributed_sample_adversary_lr(self.adversary_lr_samples)
                inner_optimizer = torch.optim.AdamW(self.model.parameters(), lr=adversary_lr)
                inner_scheduler = None
                adversary_lr_scheduler = distributed_sample_task(self.adversary_lr_schedulers)
                if adversary_lr_scheduler == "linear_warmup":
                    inner_scheduler = torch.optim.lr_scheduler.LambdaLR(
                        inner_optimizer,
                        lambda step: self.tar_inner_loop_steps / self.inner_optimizer_warmup_steps,
                    )

                switching_point = None
                if adversary_type == "retain_forget_switch":
                    switching_point = sample_switching_point(
                        self.switching_point_coeffs,
                        self.tar_inner_loop_steps,
                    )

                for inner_step in range(self.tar_inner_loop_steps):
                    current_adversary_type = adversary_type
                    if adversary_type == "retain_forget_switch":
                        current_adversary_type = "adv_retain" if inner_step < switching_point else "forget_train"

                    adversary_batches, adversary_iterators[current_adversary_type]["iter"] = next_n_batches(
                        adversary_iterators[current_adversary_type]["iter"],
                        adversary_iterators[current_adversary_type]["dataloader"],
                        self.grad_accum,
                    )

                    scheduled_weighting = (
                        schedule(inner_step, self.tar_inner_loop_steps, self.schedule_lambda)
                        if self.use_weighting_schedule
                        else 1 / self.tar_inner_loop_steps
                    )
                    compute_tr_grad = (inner_step + 1) % self.tar_inner_loop_subsample == 0
                    tamper_resistance_loss += self.inner_loop_step(
                        adversary_batches=adversary_batches,
                        meta_forget_batches=meta_forget_batches,
                        inner_optimizer=inner_optimizer,
                        inner_scheduler=inner_scheduler,
                        model_storage=storage,
                        sub_pbar=sub_pbar,
                        meta_grad_scale=(
                            self.tar_tamper_resistance_grad_scale
                            * scheduled_weighting
                            / self.tar_num_tasks_sampled
                        ),
                        compute_tamper_resistance_grad=compute_tr_grad,
                    )

                storage.add_from_storage_to_model(self.model, skip_check=True, mode="params")

            outer_retain_batches, retain_iterator = next_n_batches(
                retain_iterator,
                self.retain_dataloader,
                self.grad_accum,
            )
            total_retain_loss = 0.0
            for i in range(self.grad_accum):
                retain_batch = move_batch_to_device(outer_retain_batches[i], self.device)
                if self.retain_representations:
                    retain_loss = obj_model_mse_representations(
                        self.model,
                        retain_batch,
                        self.retain_model,
                    )
                else:
                    retain_loss = obj_standard_max_next_token(self.model, retain_batch)
                retain_loss = retain_loss / self.grad_accum * self.tar_retain_scale
                retain_loss.backward()
                total_retain_loss += retain_loss.item()

            if tamper_resistance_loss >= self.tar_tamper_resistance_loss_lower_bound or self.unbounded:
                storage.add_from_storage_to_model(self.model, mode="grads")

            storage.clear_grads()
            storage.clear_params()

            clip_grad_norm(self.model, self.max_grad_norm)
            self.opt.step()
            self.model.zero_grad(set_to_none=True)
            self.global_step += 1

            if is_main_process():
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "retain / tr": f"{total_retain_loss:.4f} / {tamper_resistance_loss:.4f}",
                    }
                )
            if wandb is not None and wandb.run is not None:
                wandb.log(
                    {
                        "retain_loss": total_retain_loss,
                        "tamper_resistance_loss": tamper_resistance_loss,
                        "learning_rate": self.opt.param_groups[0]["lr"],
                    },
                    step=self.global_step,
                )

            if self.save_steps is not None and self.global_step % self.save_steps == 0:
                self.save(f"checkpoint-step-{self.global_step}")

        pbar.close()
        self.save("final-model")
        return

class AttackTrainer(SFTTrainer):
    def __init__(
        self,
        model,
        tokenizer,
        train_dataloader,
        eval_dataloader=None,
        lr=1e-5,
        num_training_steps=None,
        epochs=None,
        grad_accum=1,
        max_grad_norm=1.0,
        device=None,
        out_dir=None,
        save_steps=None,
        eval_steps=10,
        save_checkpoint_epoch=5,
        rho=0.05,
        alpha=0.8,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            lr=lr,
            num_training_steps=num_training_steps,
            epochs=epochs,
            grad_accum=grad_accum,
            max_grad_norm=max_grad_norm,
            device=device,
            log_steps=eval_steps,  # reuse eval_steps for logging
            out_dir=out_dir,
            save_steps=save_steps,
            eval_steps=eval_steps,
            save_checkpoint_epoch=save_checkpoint_epoch,
        )
        self.rho = rho
        self.alpha = alpha
        
    def train(self):
        os.makedirs(self.out_dir, exist_ok=True)

        for epoch in range(self.epochs):
            set_dataloader_epoch(self.train_dataloader, epoch)
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}", disable=not is_main_process())
            for step, batch in enumerate(pbar):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                loss_raw = reduce_loss(self.model(**batch).loss)
                trainable_named_params = [
                    (name, p) for name, p in self.stateless_model.named_parameters() if p.requires_grad
                ]
                params = [p for _, p in trainable_named_params]
                grads = torch.autograd.grad(
                    loss_raw,
                    params,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )
                with torch.no_grad():
                    global_norm_sq = None
                    for g in grads:
                        if g is None:
                            continue
                        g2 = (g.detach().float() ** 2).sum()
                        global_norm_sq = g2 if global_norm_sq is None else (global_norm_sq + g2)
                    if global_norm_sq is None:
                        scale = None
                    else:
                        global_norm = torch.sqrt(global_norm_sq)
                        scale = self.rho / (global_norm + 1e-12) if global_norm.item() != 0.0 else None
                
                param_and_buffer_dict = {name: p for name, p in self.stateless_model.named_parameters()}
                param_and_buffer_dict.update({name: b for name, b in self.stateless_model.named_buffers()})
                if scale is not None:
                    for (name, p), g in zip(trainable_named_params, grads):
                        if g is None:
                            continue
                        perturb = g.detach().to(dtype=p.dtype) * scale.to(dtype=p.dtype)
                        param_and_buffer_dict[name] = p + perturb
                
                loss_perturbed = reduce_loss(_functional_call(self.stateless_model, param_and_buffer_dict, (), batch).loss)
                
                loss = self.alpha * (loss_raw - loss_perturbed) + loss_raw
                loss = loss / self.grad_accum
                loss.backward()

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    raw_loss=f"{loss_raw.item():.4f}",
                    perturbed_loss=f"{loss_perturbed.item():.4f}",
                )

                if (step + 1) % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()
                    self.global_step += 1

                    # if self.global_step % self.log_steps == 0:
                    tqdm.write(
                        f"Epoch {epoch+1}, Step {self.global_step}, Loss: {loss.item() * self.grad_accum:.4f}, Raw Loss: {loss_raw.item():.4f}, Perturbed Loss: {loss_perturbed.item():.4f}"
                    )
                    if wandb is not None and wandb.run is not None:
                        wandb.log(
                            {
                                "attack_loss/total": loss.item(),
                                "attack_loss/raw": loss_raw.item(),
                                "attack_loss/perturbed": loss_perturbed.item(),
                                "attack_loss/grad_norm": global_norm.item() if global_norm_sq is not None else 0.0,
                            },
                            step=self.global_step,
                        )

                    if self.save_steps is not None and self.global_step % self.save_steps == 0:
                        self.save(f"checkpoint-step-{self.global_step}")
                    
                    if self.global_step >= self.num_training_steps:
                        # Save final model
                        self.save("final-model")
                        return

            if self.save_checkpoint_epoch and (epoch + 1) % self.save_checkpoint_epoch == 0:
                self.save(f"checkpoint-epoch-{epoch+1}")
        return

class NPOGTrainer(BoosterTrainer):
    def __init__(
        self,
        model,
        model_name,
        tokenizer,
        harmful_dataloader=None,
        harmless_dataloader=None,
        lr=1e-5,
        num_training_steps=None,
        epochs=None,
        grad_accum=1,
        max_grad_norm=1.0,
        device=None,
        log_steps=20,
        out_dir=None,
        save_steps=None,
        beta=0.8,
        rho=0.05,
        save_epochs=None,
    ):
        super().__init__(
            model=model,
            model_name=model_name,
            tokenizer=tokenizer,
            harmful_dataloader=harmful_dataloader,
            harmless_dataloader=harmless_dataloader,
            lr=lr,
            num_training_steps=num_training_steps,
            epochs=epochs,
            grad_accum=grad_accum,
            max_grad_norm=max_grad_norm,
            device=device,
            log_steps=log_steps,
            out_dir=out_dir,
            save_steps=save_steps,
            rho=rho,
        )
        self.beta = beta
        self.save_epochs = save_epochs

        if wandb is not None and wandb.run is not None:
            wandb.config.update({
                "beta": beta,
                "rho": rho,
            })

    def normalize_grads(self, grads):
        global_norm_sq = None
        for g in grads:
            if g is None:
                continue
            g2 = (g.detach().float() ** 2).sum()
            global_norm_sq = g2 if global_norm_sq is None else (global_norm_sq + g2)
        if global_norm_sq is None:
            return grads, None
        global_norm = torch.sqrt(global_norm_sq)
        normalized_grads = [g / (global_norm + 1e-12) if g is not None else None for g in grads]
        return normalized_grads, global_norm
    
    def train(self):
        os.makedirs(self.out_dir, exist_ok=True)
        for epoch in range(self.epochs):
            total_steps = min(len(self.unsafe_dataloader), len(self.safe_dataloader))
            epoch_iter = zip(self.unsafe_dataloader, self.safe_dataloader)
            pbar = tqdm(epoch_iter, total=total_steps, desc=f"Epoch {epoch+1}")
            for step, (unsafe_batch, safe_batch) in enumerate(pbar):
                unsafe_batch = {k: v.to(self.device) for k, v in unsafe_batch.items()}
                safe_batch = {k: v.to(self.device) for k, v in safe_batch.items()}

                trainable_named_params = [
                    (name, p) for name, p in self.model.named_parameters() if p.requires_grad
                ]
                params = [p for _, p in trainable_named_params]
                named_params = {name: p for name, p in self.model.named_parameters()}
                named_buffers = {name: b for name, b in self.model.named_buffers()}

                safe_loss_for_grad = self.model(**safe_batch).loss
                safe_grads = torch.autograd.grad(
                    safe_loss_for_grad,
                    params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )

                with torch.no_grad():
                    # Normalize the safe gradients to get the perturbation direction
                    normalized_safe_grads, _ = self.normalize_grads(safe_grads)

                param_and_buffer_dict = dict(named_params)
                param_and_buffer_dict.update(named_buffers)

                for (name, p), g in zip(trainable_named_params, normalized_safe_grads):
                    if g is None:
                        continue
                    perturb = g.detach().to(dtype=p.dtype) * self.rho
                    param_and_buffer_dict[name] = p + perturb
            
                safe_loss_perturbed = _functional_call(self.model, param_and_buffer_dict, (), safe_batch).loss
                del safe_loss_for_grad, safe_grads, normalized_safe_grads, param_and_buffer_dict
                
                # --- SAM-style perturbation direction from unsafe loss (no inplace param edits) ---
                unsafe_loss_for_grad = self.model(**unsafe_batch).loss
                unsafe_grads = torch.autograd.grad(
                    unsafe_loss_for_grad,
                    params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )

                with torch.no_grad():
                    # Normalize the unsafe gradients to get the perturbation direction
                    normalized_unsafe_grads, _ = self.normalize_grads(unsafe_grads)
                    

                # Compute perturbed unsafe loss using functional_call to avoid inplace modifications
                param_and_buffer_dict = dict(named_params)
                param_and_buffer_dict.update(named_buffers)
                
                for (name, p), g in zip(trainable_named_params, normalized_unsafe_grads):
                    if g is None:
                        continue
                    perturb = g.detach().to(dtype=p.dtype) * self.rho
                    param_and_buffer_dict[name] = p + perturb
                        
                # unsafe_loss_perturbed = _functional_call(self.model, param_and_buffer_dict, (), unsafe_batch).loss
                unsafe_loss_perturbed = _functional_call(self.model, param_and_buffer_dict, (), unsafe_batch).loss
                del unsafe_loss_for_grad, unsafe_grads, normalized_unsafe_grads, param_and_buffer_dict, named_params, named_buffers
 
                # Losses at the original parameters (safe for backward)
                safe_loss_raw = self.model(**safe_batch).loss

                unsafe_loss_raw = self.model(**unsafe_batch).loss

                # IMPORTANT: only backprop through graphs built with current (unmodified) parameters
                # loss = safe_loss_raw - torch.log(
                #     (1 - self.alpha) * unsafe_loss_raw + self.alpha * unsafe_loss_perturbed
                # )
                # loss = safe_loss_raw + self.alpha * (unsafe_loss_raw - unsafe_loss_perturbed)
                # loss = (1 - self.alpha) * safe_loss_raw + self.alpha * unsafe_loss_perturbed
                loss = safe_loss_raw - torch.log(torch.sigmoid(self.beta * (torch.log(safe_loss_raw / safe_loss_perturbed) - torch.log(unsafe_loss_raw / unsafe_loss_perturbed)))) / self.beta
                
                loss = loss / self.grad_accum
                loss.backward()
                
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    sloss=f"{safe_loss_raw.item():.4f}",
                    sploss=f"{safe_loss_perturbed.item():.4f}",
                    hloss=f"{unsafe_loss_raw.item():.4f}",
                    hploss=f"{unsafe_loss_perturbed.item():.4f}",
                )
                
                if (step + 1) % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad(set_to_none=True)
                    self.global_step += 1

                    # if self.global_step % self.log_steps == 0:
                    tqdm.write(
                        f"Epoch {epoch+1}, Step {self.global_step}, Loss: {loss.item() * self.grad_accum:.4f}, Safe: {safe_loss_raw.item():.4f}, Perturbed Safe: {safe_loss_perturbed.item():.4f}, Unsafe: {unsafe_loss_raw.item():.4f}, Perturbed Unsafe: {unsafe_loss_perturbed.item():.4f}"
                    )
                    if wandb is not None and wandb.run is not None:
                        wandb.log(
                            {
                                "loss/total": loss.item() * self.grad_accum,
                                "loss/safe": safe_loss_raw.item(),
                                "loss/safe_perturbed": safe_loss_perturbed.item(),
                                "loss/unsafe": unsafe_loss_raw.item(),
                                "loss/unsafe_perturbed": unsafe_loss_perturbed.item(),
                            },
                            step=self.global_step,
                        )

                    if self.save_steps is not None and self.global_step % self.save_steps == 0:
                        self.save(f"checkpoint-step-{self.global_step}")
                    
                    if self.global_step >= self.num_training_steps:
                        # Save final model
                        self.save("final-model")
                        return
            
            # End of epoch
            if self.save_epochs is not None and (epoch + 1) % self.save_epochs == 0:
                self.save(f"checkpoint-epoch-{epoch+1}")
        return

class PatchTrainer(SFTTrainer):
    def __init__(
        self,
        model,
        attacked_model,
        tokenizer,
        train_dataloader,
        eval_dataloader=None,
        lr=1e-5,
        num_training_steps=None,
        epochs=None,
        grad_accum=1,
        max_grad_norm=1.0,
        device=None,
        out_dir=None,
        save_steps=None,
        eval_steps=10,
        save_checkpoint_epoch=5,
        alpha=0.5,
        # lambda_reg=0.01,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            lr=lr,
            num_training_steps=num_training_steps,
            epochs=epochs,
            grad_accum=grad_accum,
            max_grad_norm=max_grad_norm,
            device=device,
            log_steps=eval_steps,  # reuse eval_steps for logging
            out_dir=out_dir,
            save_steps=save_steps,
            eval_steps=eval_steps,
            save_checkpoint_epoch=save_checkpoint_epoch,
        )
        self.attacked_model = attacked_model
        self.attacked_raw_model = unwrap_model(self.attacked_model)
        self.attacked_raw_model.to(self.device)
        self.attacked_model.eval()

        self.alpha = alpha
        # self.lambda_reg = lambda_reg
        # initialize patch
        base_params = {
            name: p.detach() for name, p in self.stateless_model.named_parameters()
        }
        attacked_base_params = {
            name: p.detach()
            for name, p in self.attacked_raw_model.named_parameters()
        }
        if set(base_params.keys()) != set(attacked_base_params.keys()):
            raise ValueError("model and attacked_model must share the same parameter names for PatchTrainer.")

        # calculate the attack vector 
        self.attack_vector = {
            name: attacked_base_params[name] - base_params[name]
            for name in base_params.keys()
        }

    def _compose_params(self, params, stateless_model):
        params_and_buffers = {
            name: tensor for name, tensor in stateless_model.named_buffers()
        }
        for name, param in params.items():
            params_and_buffers[name] = param + self.attack_vector[name]
        return params_and_buffers

    # def save(self, name):
    #     save_path = os.path.join(self.out_dir, name)
    #     os.makedirs(save_path, exist_ok=True)

    #     merged_state_dict = {
    #         key: value.detach().cpu().clone()
    #         for key, value in self.raw_model.state_dict().items()
    #     }
    #     for name, base_param in self.base_params.items():
    #         merged_state_dict[name] = (base_param + self.patch[name]).detach().cpu().clone()

    #     if is_main_process():
    #         self.raw_model.save_pretrained(save_path, state_dict=merged_state_dict)
    #         self.tokenizer.save_pretrained(save_path)
    #         print(f"Model saved to {save_path}")
    #     if is_distributed():
    #         dist.barrier()
        
    
    def train(self):
        os.makedirs(self.out_dir, exist_ok=True)

        for epoch in range(self.epochs):
            set_dataloader_epoch(self.train_dataloader, epoch)
            
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}", disable=not is_main_process())
            for step, batch in enumerate(pbar):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # attacked_params = self._compose_params(self.attacked_base_params, self.attacked_raw_model)
                # safe_params = self._compose_params(self.base_params, self.stateless_model)

                # loss_attack = reduce_loss(_functional_call(self.attacked_raw_model, attacked_params, (), batch).loss)

                # loss_safe = reduce_loss(_functional_call(self.stateless_model, safe_params, (), batch).loss)

                # loss_reg = 0.5 * self.lambda_reg * sum((p ** 2).sum() for p in self.patch.values())

                # loss = self.alpha * loss_attack + (1 - self.alpha) * loss_safe + loss_reg

                loss_safe = self.model(**batch).loss

                params = {name: p for name, p in self.stateless_model.named_parameters()}
                attack_params = self._compose_params(params, self.stateless_model)
                loss_attack = _functional_call(self.stateless_model, attack_params, (), batch).loss

                loss = self.alpha * loss_attack + (1 - self.alpha) * loss_safe

                loss = loss / self.grad_accum
                loss.backward()
                pbar.set_postfix(
                    loss=f"{loss.item() * self.grad_accum:.4f}",
                    attack_loss=f"{loss_attack.item():.4f}",
                    safe_loss=f"{loss_safe.item():.4f}",
                )
                if (step + 1) % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.stateless_model.parameters(), self.max_grad_norm)
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()
                    self.global_step += 1

                    tqdm.write(
                        f"Epoch {epoch+1}, Step {self.global_step}, Loss: {loss.item() * self.grad_accum:.4f}, Attack Loss: {loss_attack.item():.4f}, Safe Loss: {loss_safe.item():.4f}, Attack-Safe Gap: {(loss_attack - loss_safe).item():.4f}"
                    )
                    if wandb is not None and wandb.run is not None:
                        wandb.log(
                            {
                                "patch_loss/total": loss.item() * self.grad_accum,
                                "patch_loss/attack": loss_attack.item(),
                                "patch_loss/safe": loss_safe.item(),
                                "patch_loss/attack_safe_gap": (loss_attack - loss_safe).item(),
                                # "patch_loss/reg": loss_reg.item(),
                            },
                            step=self.global_step,
                        )
                    if self.save_steps is not None and self.global_step % self.save_steps == 0:
                        self.save(f"checkpoint-step-{self.global_step}")
                    if self.global_step >= self.num_training_steps:
                        # Save final model
                        self.save("final-model")
                        return

            if self.save_checkpoint_epoch and (epoch + 1) % self.save_checkpoint_epoch == 0:
                self.save(f"checkpoint-epoch-{epoch+1}")
        return
