# a naive trainer using repnoise loss and beavertails dataset
import json
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, get_scheduler
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


def named_parameters_dict(model):
    return dict(model.named_parameters())


def read_latest_manifest_if_exists(manifest_path):
    if not manifest_path or not os.path.exists(manifest_path):
        return None

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    return manifest if isinstance(manifest, dict) else None


def broadcast_python_object(obj):
    if not is_distributed():
        return obj

    payload = [obj if is_main_process() else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def write_manifest_atomic(manifest_path, payload):
    if not manifest_path:
        return

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    tmp_path = f"{manifest_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, manifest_path)


@contextmanager
def apply_attack_vector(model, attack_vector):
    params = named_parameters_dict(model)
    try:
        with torch.no_grad():
            for name, delta in attack_vector.items():
                if name not in params:
                    continue
                params[name].add_(delta)
        yield
    finally:
        with torch.no_grad():
            for name, delta in attack_vector.items():
                if name not in params:
                    continue
                params[name].sub_(delta)


def reduce_loss(loss):
    if isinstance(loss, torch.Tensor) and loss.ndim > 0:
        return loss.mean()
    return loss

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
        self.sft_manifest_path = os.path.join(self.out_dir, "latest_sft.json")

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
        if is_main_process():
            payload = {
                "kind": "sft_checkpoint",
                "name": name,
                "version": self.global_step,
                "global_step": self.global_step,
                "path": save_path,
                "out_dir": self.out_dir,
            }
            write_manifest_atomic(self.sft_manifest_path, payload)
        if is_distributed():
            dist.barrier()

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
                    grad_norm = clip_grad_norm(self.model, self.max_grad_norm)
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
        attack_model_path=None,
        attack_manifest_path=None,
        check_attack_model=0,
        ema_decay=0.9,
        attack_model_builder=None,
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
        self.alpha = alpha
        self.attack_model_path = attack_model_path
        self.attack_checkpoint_dir = (
            os.path.dirname(os.path.abspath(attack_model_path)) if attack_model_path is not None else self.out_dir
        )
        self.patch_manifest_path = os.path.join(self.out_dir, "latest_patch.json")
        self.attack_manifest_path = attack_manifest_path
        self.check_attack_model = check_attack_model
        self.ema_decay = ema_decay
        self.attack_model_builder = attack_model_builder
        self.last_attack_version = -1
        self.last_attack_path = None

        self.base_params = {
            name: p.detach().clone() for name, p in self.raw_model.named_parameters()
        }
        self.attack_vector = None
        self._update_attack_vector_from_model(attacked_model, use_ema=False)
        self.last_attack_path = attack_model_path

        del attacked_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    def _build_attack_vector_from_model(self, attacked_model):
        attacked_raw_model = unwrap_model(attacked_model)
        attacked_params = {
            name: p.detach().clone() for name, p in attacked_raw_model.named_parameters()
        }
        if set(self.base_params.keys()) != set(attacked_params.keys()):
            raise ValueError("model and attacked_model must share the same parameter names for PatchTrainer.")
        return {
            name: attacked_params[name] - self.base_params[name]
            for name in self.base_params.keys()
        }

    def _merge_attack_vector(self, new_attack_vector, use_ema=True):
        if self.attack_vector is None or not use_ema:
            self.attack_vector = {
                name: tensor.detach().clone()
                for name, tensor in new_attack_vector.items()
            }
            return

        for name in self.attack_vector.keys():
            self.attack_vector[name].mul_(self.ema_decay).add_(
                new_attack_vector[name].to(self.attack_vector[name].device),
                alpha=1 - self.ema_decay,
            )

    def _update_attack_vector_from_model(self, attacked_model, use_ema=True):
        new_attack_vector = self._build_attack_vector_from_model(attacked_model)
        self._merge_attack_vector(new_attack_vector, use_ema=use_ema)

    def save(self, name):
        save_path = os.path.join(self.out_dir, name)
        save_model_and_tokenizer(self.model, self.raw_model, self.tokenizer, save_path)
        if is_main_process():
            payload = {
                "kind": "patch_checkpoint",
                "name": name,
                "version": self.global_step,
                "global_step": self.global_step,
                "path": save_path,
                "out_dir": self.out_dir,
            }
            write_manifest_atomic(self.patch_manifest_path, payload)
        if is_distributed():
            dist.barrier()

    def load_attack_model(self, attack_model_path):
        if not os.path.exists(attack_model_path):
            print(f"Attack model checkpoint {attack_model_path} does not exist.")
            return None
        attack_model = AutoModelForCausalLM.from_pretrained(attack_model_path, low_cpu_mem_usage=True)
        if self.attack_model_builder is not None:
            attack_model = self.attack_model_builder(attack_model, self.device)
        else:
            attack_model = attack_model.to(self.device)
        attack_model.requires_grad_(False)
        attack_model.eval()
        return attack_model

    def _resolve_manifest_update(self, manifest):
        if manifest is None:
            return None

        version = manifest.get("version", manifest.get("latest_attack_step", -1))
        if version is None or version <= self.last_attack_version:
            return None

        attack_model_path = manifest.get("path") or manifest.get("attack_model_path")
        if attack_model_path is None:
            latest_attack_step = manifest.get("latest_attack_step")
            if latest_attack_step is None:
                return None
            attack_model_path = os.path.join(
                self.attack_checkpoint_dir,
                f"checkpoint-step-{latest_attack_step}",
            )

        return {
            "version": version,
            "path": attack_model_path,
        }

    def maybe_refresh_attack_vector(self):
        if not self.attack_manifest_path or self.check_attack_model <= 0:
            return

        update_info = None
        if is_main_process():
            manifest = read_latest_manifest_if_exists(self.attack_manifest_path)
            update_info = self._resolve_manifest_update(manifest)

        update_info = broadcast_python_object(update_info)
        if update_info is None:
            return

        attack_model = self.load_attack_model(update_info["path"])
        if attack_model is None:
            return

        self._update_attack_vector_from_model(attack_model, use_ema=self.attack_vector is not None)
        self.last_attack_version = update_info["version"]
        self.last_attack_path = update_info["path"]

        del attack_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if is_distributed():
            dist.barrier()
        
    
    def train(self):
        os.makedirs(self.out_dir, exist_ok=True)

        for epoch in range(self.epochs):
            set_dataloader_epoch(self.train_dataloader, epoch)
            if self.eval_dataloader is not None:
                set_dataloader_epoch(self.eval_dataloader, epoch)
            
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}", disable=not is_main_process())
            for step, batch in enumerate(pbar):
                if self.check_attack_model > 0 and (step + 1) % self.check_attack_model == 0:
                    self.maybe_refresh_attack_vector()

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
                    eval_avg_loss /= (eval_step + 1)
                    tqdm.write(f"Epoch {epoch+1}, Step {self.global_step}, Eval Loss: {eval_avg_loss:.4f}")
                    if wandb is not None and wandb.run is not None:
                        wandb.log(
                            {
                                "eval/loss": eval_avg_loss,
                            },
                            step=self.global_step,
                        )

                batch = {k: v.to(self.device) for k, v in batch.items()}

                loss_safe = self.model(**batch).loss

                with apply_attack_vector(self.raw_model, self.attack_vector):
                    loss_attack = self.model(**batch).loss

                loss = self.alpha * loss_attack + (1 - self.alpha) * loss_safe

                loss = loss / self.grad_accum
                loss.backward()
                pbar.set_postfix(
                    loss=f"{loss.item() * self.grad_accum:.4f}",
                    attack_loss=f"{loss_attack.item():.4f}",
                    safe_loss=f"{loss_safe.item():.4f}",
                )
                if (step + 1) % self.grad_accum == 0:
                    grad_norm = clip_grad_norm(self.model, self.max_grad_norm)
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
                                "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
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
