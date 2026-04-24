import random
from typing import Dict

from torch.utils.data import DataLoader

from reproduce.datasets.get_data import get_repnoise
from reproduce.datasets.utils import (
    ConversationDataset,
    DPODataCollatorWithPadding,
    DPOPairDataset,
    make_collate_fn,
)
from reproduce.train.utils import build_distributed_sampler


def _split_indices(length: int, meta_ratio: float, seed: int):
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    meta_size = max(1, int(length * meta_ratio))
    meta_indices = set(indices[:meta_size])
    train_indices = [idx for idx in indices if idx not in meta_indices]
    meta_indices = [idx for idx in indices if idx in meta_indices]
    return train_indices, meta_indices


def _subset(data, indices):
    return [data[idx] for idx in indices]


def build_tar_dataloaders(
    tokenizer,
    batch_size: int,
    tar_loss_type: str,
    model_name: str = "qwen",
    meta_ratio: float = 0.2,
    seed: int = 42,
    adversary_batch_size: int = None,
    distributed: bool = False,
) -> Dict[str, DataLoader]:
    safe_data, unsafe_data = get_repnoise(split="train")
    adversary_batch_size = adversary_batch_size or batch_size

    retain_dataset = ConversationDataset(safe_data)
    retain_sampler = build_distributed_sampler(retain_dataset, shuffle=True) if distributed else None
    retain_dataloader = DataLoader(
        retain_dataset,
        batch_size=batch_size,
        shuffle=retain_sampler is None,
        sampler=retain_sampler,
        collate_fn=make_collate_fn(tokenizer, mask_prompts=True, model_name=model_name),
    )

    if tar_loss_type == "dpo":
        pref_dataset = DPOPairDataset(safe_data, unsafe_data, tokenizer, model_name=model_name)
        pref_sampler = build_distributed_sampler(pref_dataset, shuffle=True) if distributed else None
        pref_dataloader = DataLoader(
            pref_dataset,
            batch_size=batch_size,
            shuffle=pref_sampler is None,
            sampler=pref_sampler,
            collate_fn=DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                label_pad_token_id=-100,
            ),
        )
        unsafe_dataset = ConversationDataset(unsafe_data)
        unsafe_sampler = build_distributed_sampler(unsafe_dataset, shuffle=True) if distributed else None
        forget_train_dataloader = DataLoader(
            unsafe_dataset,
            batch_size=adversary_batch_size,
            shuffle=unsafe_sampler is None,
            sampler=unsafe_sampler,
            collate_fn=make_collate_fn(tokenizer, mask_prompts=True, model_name=model_name),
        )
        return {
            "retain": retain_dataloader,
            "adv_retain": retain_dataloader,
            "forget_train": forget_train_dataloader,
            "harmful_completions": pref_dataloader,
            "meta": pref_dataloader,
        }

    train_indices, meta_indices = _split_indices(len(unsafe_data), meta_ratio=meta_ratio, seed=seed)
    unsafe_train_dataset = ConversationDataset(_subset(unsafe_data, train_indices))
    unsafe_meta_dataset = ConversationDataset(_subset(unsafe_data, meta_indices))

    train_sampler = build_distributed_sampler(unsafe_train_dataset, shuffle=True) if distributed else None
    meta_sampler = build_distributed_sampler(unsafe_meta_dataset, shuffle=True) if distributed else None

    forget_train_dataloader = DataLoader(
        unsafe_train_dataset,
        batch_size=adversary_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=make_collate_fn(tokenizer, mask_prompts=True, model_name=model_name),
    )
    meta_dataloader = DataLoader(
        unsafe_meta_dataset,
        batch_size=batch_size,
        shuffle=meta_sampler is None,
        sampler=meta_sampler,
        collate_fn=make_collate_fn(tokenizer, mask_prompts=True, model_name=model_name),
    )
    return {
        "retain": retain_dataloader,
        "adv_retain": retain_dataloader,
        "forget_train": forget_train_dataloader,
        "meta": meta_dataloader,
    }
