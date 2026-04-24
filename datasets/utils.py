from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from random import randint

def _extract_messages(sample: Union[Dict[str, Any], List[Dict[str, Any]], str]):
    if isinstance(sample, dict) and "messages" in sample:
        return sample["messages"]
    if isinstance(sample, list):
        return sample
    if isinstance(sample, str):
        return sample
    raise ValueError("Unsupported sample type for collate.")


def _format_messages_with_template(tokenizer, messages: List[Dict[str, Any]], add_generation_prompt: bool = False) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer does not support apply_chat_template.")
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def _format_messages_fallback(messages: List[Dict[str, Any]]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"<<SYS>>\n{content}\n<</SYS>>\n")
        elif role == "user":
            parts.append(f"User: {content}\n")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(str(content))
    return "\n".join(parts)


def _build_text(tokenizer, messages: List[Dict[str, Any]], use_template: bool) -> str:
    if use_template and hasattr(tokenizer, "apply_chat_template"):
        return _format_messages_with_template(tokenizer, messages, add_generation_prompt=False)
    return _format_messages_fallback(messages)


def _build_prompt_text_qwen(tokenizer, messages: List[Dict[str, Any]], use_template: bool) -> str:
    prompt_messages = [dict(m) for m in messages]
    if prompt_messages and prompt_messages[-1].get("role") == "assistant":
        prompt_messages[-1]["content"] = ""
    if use_template and hasattr(tokenizer, "apply_chat_template"):
        formatted_message = _format_messages_with_template(tokenizer, prompt_messages, add_generation_prompt=False)
        return formatted_message.rsplit("<|im_end|>\n", 1)[0] # remove "<|im_end|>\n"
    return _format_messages_fallback(prompt_messages)

def _build_prompt_text_llama(tokenizer, messages: List[Dict[str, Any]], use_template: bool) -> str:
    prompt_messages = [dict(m) for m in messages]
    if prompt_messages and prompt_messages[-1].get("role") == "assistant":
        prompt_messages[-1]["content"] = ""
    if use_template and hasattr(tokenizer, "apply_chat_template"):
        formatted_message = _format_messages_with_template(tokenizer, prompt_messages, add_generation_prompt=False)
        return formatted_message.rsplit("<|eot_id|>", 1)[0] # remove "<|eot_id|>"
    return _format_messages_fallback(prompt_messages)


def make_collate_fn(
    tokenizer,
    use_template: bool = True,
    mask_prompts: bool = False,
    max_length: Optional[int] = None,
    model_name: Optional[str] = None,
) -> Callable[[List[Any]], Dict[str, torch.Tensor]]:
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def collate_fn(batch: List[Any]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for sample in batch:
            messages = _extract_messages(sample)
            if isinstance(messages, str):
                full_text = messages
                prompt_len = 0
            else:
                full_text = _build_text(tokenizer, messages, use_template=use_template)
                prompt_len = 0
                if mask_prompts:
                    if model_name == "qwen":
                        prompt_text = _build_prompt_text_qwen(tokenizer, messages, use_template=use_template)
                    elif model_name == "llama":
                        prompt_text = _build_prompt_text_llama(tokenizer, messages, use_template=use_template)
                    else:
                        raise Warning("currently only qwen and llama model is supported for prompt length calculation when mask_prompts is True.")
                        prompt_text = _build_text(tokenizer, messages, use_template=use_template)
                prompt_len = len(
                    tokenizer(prompt_text, add_special_tokens=False).input_ids
                )

            encoded = tokenizer(full_text, add_special_tokens=False)
            input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()

            if mask_prompts and prompt_len > 0:
                prompt_len = min(prompt_len, labels.shape[0])
                labels[:prompt_len] = -100

            if max_length is not None and input_ids.shape[0] > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                labels = labels[:max_length]

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        max_len = max(t.shape[0] for t in input_ids_list)
        if max_length is not None:
            max_len = min(max_len, max_length)

        def _pad(t: torch.Tensor, value: int) -> torch.Tensor:
            if t.shape[0] == max_len:
                return t
            pad_amount = max_len - t.shape[0]
            return torch.nn.functional.pad(t, (0, pad_amount), value=value)

        input_ids = torch.stack([_pad(t, pad_token_id) for t in input_ids_list])
        attention_mask = torch.stack([_pad(t, 0) for t in attention_mask_list])
        labels = torch.stack([_pad(t, -100) for t in labels_list])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate_fn

def augment_collate_fn(
    tokenizer,  
    use_template: bool = True,
    mask_prompts: bool = False,
    max_length: Optional[int] = None,
    model_name: Optional[str] = None,
) -> Callable[[List[Any]], Dict[str, torch.Tensor]]:
    
    def collate_fn(batch: List[Any]) -> Dict[str, torch.Tensor]:
        dummy_assistant = {
            "role": "assistant",
            "content": ""
        }
        dummy_batch = []
        base_batch = []
        adv_batch = []
        adv_indices = []
        for sample in batch:
            if len(sample) == 4:
                system, user, assistant, assistant_adv = sample
                dummy_sample = [system, user, dummy_assistant]
                base_sample = [system, user, assistant]
                adv_sample = [system, user, assistant_adv]
                adv_indices.append(len(base_batch))
                dummy_batch.append(dummy_sample)
                base_batch.append(base_sample)
                adv_batch.append(adv_sample)
            else:
                system, user, assistant = sample
                dummy_sample = [system, user, dummy_assistant]
                base_sample = [system, user, assistant]
                dummy_batch.append(dummy_sample)
                base_batch.append(base_sample)

        if adv_batch:
            adv_inputs = tokenizer.apply_chat_template(
                adv_batch,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = tokenizer.apply_chat_template(
                base_batch,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            seq_len = max(inputs["input_ids"].shape[1], adv_inputs["input_ids"].shape[1])
            # update
            adv_inputs = tokenizer.apply_chat_template(
                adv_batch,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=seq_len
            )
            inputs = tokenizer.apply_chat_template(
                base_batch,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=seq_len
            )
        else:
            inputs = tokenizer.apply_chat_template(
                base_batch,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            seq_len = inputs["input_ids"].shape[1]

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        seq_len = input_ids.shape[1]
        dummy_inputs = tokenizer.apply_chat_template(
            dummy_batch,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_len
        )
        dummy_input_ids = dummy_inputs["input_ids"]

        labels = input_ids.clone()
        shared = (input_ids == dummy_input_ids) & (attention_mask == 1)
        labels = labels.masked_fill(shared, -100)
        labels = labels.masked_fill(attention_mask == 0, -100)

        if adv_batch:
            aug_input_ids = []
            aug_labels = []
            aug_attention_mask = []
            # the labels should mask out the prompt + adversarial part
            adv_input_ids = adv_inputs["input_ids"]
            adv_attention_mask = adv_inputs["attention_mask"]
            adv_labels = adv_input_ids.clone()
            shared_adv = (adv_input_ids == dummy_input_ids) & (adv_attention_mask == 1)
            adv_labels = adv_labels.masked_fill(shared_adv, -100)
            adv_labels = adv_labels.masked_fill(adv_attention_mask == 0, -100)

            for j, i in enumerate(adv_indices):
                safe_input_id_item = input_ids[i]
                adv_input_id_item = adv_input_ids[j]
                safe_label_item = labels[i]
                adv_label_item = adv_labels[j]
                safe_idx_item = torch.where(safe_label_item != -100)[0]
                adv_idx_item = torch.where(adv_label_item != -100)[0]

                toss = randint(0, 1)
                if toss == 0: # 50% chance to augment harmful tokens

                    cutoff_point = randint(1, min(10, len(adv_idx_item)))
                    cutoff_point = adv_idx_item[cutoff_point - 1]
                    input_ids_item = torch.cat(
                        [ adv_input_id_item[:cutoff_point+1], safe_input_id_item[safe_idx_item[0] : safe_idx_item[-1]+1] ]
                    )
                    labels_item = torch.cat(
                        [ adv_label_item[:cutoff_point+1], safe_label_item[safe_idx_item[0] : safe_idx_item[-1]+1] ]
                    )
                    labels_item[:cutoff_point+1] = -100     # block gradients of harmful tokens
                else:
                    input_ids_item = safe_input_id_item[: safe_idx_item[-1]+1]
                    labels_item = safe_label_item[: safe_idx_item[-1]+1]
                
                aug_input_ids.append(input_ids_item)
                aug_labels.append(labels_item)
                aug_attention_mask.append(torch.ones_like(input_ids_item))
            
            # Pad sequences
            max_length = max([x.size(0) for x in aug_input_ids])
            aug_input_ids = torch.stack([torch.nn.functional.pad(x, (0, max_length - x.size(0)), value=tokenizer.pad_token_id) for x in aug_input_ids])
            aug_labels = torch.stack([torch.nn.functional.pad(x, (0, max_length - x.size(0)), value=-100) for x in aug_labels])
            aug_attention_mask = torch.stack([torch.nn.functional.pad(x, (0, max_length - x.size(0)), value=0) for x in aug_attention_mask])

            inputs["input_ids"] = aug_input_ids
            inputs["labels"] = aug_labels
            inputs["attention_mask"] = aug_attention_mask

        return inputs
    
    return collate_fn 
        

def inf_collate_fn(
    tokenizer,
    use_template: bool = True,
    max_length: Optional[int] = None,
    model_name: Optional[str] = None,
) -> Callable[[List[Any]], Dict[str, torch.Tensor]]:

    def collate_fn(batch: List[Any]) -> Dict[str, torch.Tensor]:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer.apply_chat_template(
            batch,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return inputs

    return collate_fn

class ConversationDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

    def shuffle(self):
        import random
        random.shuffle(self.data_list)
        return self


def _build_prompt_text(
    tokenizer,
    messages: List[Dict[str, Any]],
    use_template: bool = True,
    model_name: Optional[str] = None,
) -> str:
    if model_name == "qwen":
        return _build_prompt_text_qwen(tokenizer, messages, use_template=use_template)
    if model_name == "llama":
        return _build_prompt_text_llama(tokenizer, messages, use_template=use_template)
    return _build_text(tokenizer, messages, use_template=use_template)


def build_sft_feature(
    sample: List[Dict[str, Any]],
    tokenizer,
    use_template: bool = True,
    model_name: Optional[str] = None,
) -> Dict[str, List[int]]:
    full_text = _build_text(tokenizer, sample, use_template=use_template)
    prompt_text = _build_prompt_text(
        tokenizer,
        sample,
        use_template=use_template,
        model_name=model_name,
    )
    full_encoded = tokenizer(full_text, add_special_tokens=False)
    prompt_encoded = tokenizer(prompt_text, add_special_tokens=False)
    input_ids = list(full_encoded["input_ids"])
    attention_mask = [1] * len(input_ids)
    labels = list(input_ids)
    prompt_len = min(len(prompt_encoded["input_ids"]), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class DPOPairDataset(Dataset):
    def __init__(self, chosen_data, rejected_data, tokenizer, model_name: Optional[str] = None, use_template: bool = True):
        if len(chosen_data) != len(rejected_data):
            raise ValueError("chosen_data and rejected_data must have the same length.")
        self.chosen_data = chosen_data
        self.rejected_data = rejected_data
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.use_template = use_template

    def __len__(self):
        return len(self.chosen_data)

    def __getitem__(self, idx):
        chosen_feature = build_sft_feature(
            self.chosen_data[idx],
            self.tokenizer,
            use_template=self.use_template,
            model_name=self.model_name,
        )
        rejected_feature = build_sft_feature(
            self.rejected_data[idx],
            self.tokenizer,
            use_template=self.use_template,
            model_name=self.model_name,
        )
        return {
            "chosen_input_ids": chosen_feature["input_ids"],
            "chosen_attention_mask": chosen_feature["attention_mask"],
            "chosen_labels": chosen_feature["labels"],
            "rejected_input_ids": rejected_feature["input_ids"],
            "rejected_attention_mask": rejected_feature["attention_mask"],
            "rejected_labels": rejected_feature["labels"],
        }


@dataclass
class DPODataCollatorWithPadding:
    pad_token_id: int = 0
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        padded_batch = {}
        for key in features[0].keys():
            if key.endswith("_input_ids") or key.endswith("_attention_mask") or key.endswith("_labels"):
                to_pad = [torch.tensor(feature[key], dtype=torch.long) for feature in features]
                if key.endswith("_input_ids"):
                    padding_value = self.pad_token_id
                elif key.endswith("_attention_mask"):
                    padding_value = 0
                else:
                    padding_value = self.label_pad_token_id
                padded_batch[key] = pad_sequence(
                    to_pad,
                    batch_first=True,
                    padding_value=padding_value,
                )
            elif key.endswith("_logps"):
                padded_batch[key] = torch.tensor([feature[key] for feature in features])
            else:
                padded_batch[key] = [feature[key] for feature in features]
        return padded_batch
