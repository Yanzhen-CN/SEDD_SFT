import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def segment_text(segments, train=None):
    selected = segments
    if train is not None:
        selected = [segment for segment in segments if bool(segment["train"]) is train]
    return "".join(segment["text"] for segment in selected)


class AnswerSegmentDataset(Dataset):
    def __init__(self, path, tokenizer, max_length, min_target_tokens=1):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.min_target_tokens = int(min_target_tokens)
        self.samples = []

        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    if not isinstance(sample, list):
                        raise TypeError(f"Expected one segment list per JSONL line in {self.path}")
                    self.samples.append(sample)

        if not self.samples:
            raise ValueError(f"No samples found in {self.path}")

    def __len__(self):
        return len(self.samples)

    def _encode_segments(self, segments):
        encoded = []
        for segment in segments:
            token_ids = self.tokenizer(segment["text"], add_special_tokens=False).input_ids
            encoded.append({
                "name": segment.get("name", "segment"),
                "ids": token_ids,
                "train": bool(segment["train"]),
            })

        ids, train_mask = self._truncate(encoded)
        if sum(train_mask) == 0:
            ids = ids[: self.max_length - 1] + [self.tokenizer.eos_token_id]
            train_mask = train_mask[: self.max_length - 1] + [1]

        attention_mask = [1] * len(ids)
        pad_len = self.max_length - len(ids)
        if pad_len > 0:
            ids += [self.tokenizer.eos_token_id] * pad_len
            train_mask += [0] * pad_len
            attention_mask += [0] * pad_len

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "train_mask": torch.tensor(train_mask, dtype=torch.bool),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "prompt_len": sum(len(item["ids"]) for item in encoded if not item["train"]),
            "train_len": int(sum(train_mask)),
        }

    def _truncate(self, encoded):
        total_len = sum(len(item["ids"]) for item in encoded)
        if total_len <= self.max_length:
            ids = []
            train_mask = []
            for item in encoded:
                ids.extend(item["ids"])
                train_mask.extend([1 if item["train"] else 0] * len(item["ids"]))
            return ids, train_mask

        fixed_before_assistant = []
        assistant = []
        reasoning_label = []
        reasoning = []
        seen_assistant = False

        for item in encoded:
            if item["name"] == "assistant":
                seen_assistant = True
                assistant = item["ids"]
            elif item["name"] == "reasoning_label":
                reasoning_label = item["ids"]
            elif item["name"] == "reasoning":
                reasoning = item["ids"]
            elif not seen_assistant:
                fixed_before_assistant.extend(item["ids"])

        if not assistant:
            return self._truncate_generic(encoded)

        has_reasoning = len(reasoning) > 0
        reserved_reasoning = self.min_target_tokens if has_reasoning else 0
        prompt_budget = max(1, self.max_length - reserved_reasoning - len(reasoning_label) - self.min_target_tokens)
        prompt_ids = fixed_before_assistant[-prompt_budget:]
        assistant_budget = self.max_length - len(prompt_ids) - len(reasoning_label) - reserved_reasoning
        assistant_ids = assistant[:max(1, assistant_budget)]
        remaining = self.max_length - len(prompt_ids) - len(assistant_ids) - len(reasoning_label)
        reasoning_ids = reasoning[:max(0, remaining)]

        ids = prompt_ids + assistant_ids + reasoning_label + reasoning_ids
        train_mask = (
            [0] * len(prompt_ids)
            + [1] * len(assistant_ids)
            + [0] * len(reasoning_label)
            + [1] * len(reasoning_ids)
        )
        return ids[:self.max_length], train_mask[:self.max_length]

    def _truncate_generic(self, encoded):
        ids = []
        train_mask = []
        remaining = self.max_length
        for item in encoded:
            if remaining <= 0:
                break
            take = min(len(item["ids"]), remaining)
            ids.extend(item["ids"][:take])
            train_mask.extend([1 if item["train"] else 0] * take)
            remaining -= take
        return ids, train_mask

    def __getitem__(self, idx):
        segments = self.samples[idx]
        item = self._encode_segments(segments)
        item["id"] = str(idx)
        item["prompt"] = segment_text(segments, train=False)
        item["target"] = segment_text(segments, train=True)
        item["answer"] = next((segment["text"] for segment in segments if segment.get("name") == "assistant"), "")
        item["segments"] = segments
        return item


def collate_answer_batch(items):
    tensor_keys = ["input_ids", "train_mask", "attention_mask"]
    batch = {key: torch.stack([item[key] for item in items], dim=0) for key in tensor_keys}
    batch["ids"] = [item["id"] for item in items]
    batch["prompts"] = [item["prompt"] for item in items]
    batch["targets"] = [item["target"] for item in items]
    batch["answers"] = [item["answer"] for item in items]
    batch["segments"] = [item["segments"] for item in items]
    batch["prompt_lens"] = torch.tensor([item["prompt_len"] for item in items], dtype=torch.long)
    batch["train_lens"] = torch.tensor([item["train_len"] for item in items], dtype=torch.long)
    return batch


def make_answer_loader(path, tokenizer, max_length, min_target_tokens, batch_size, shuffle, num_workers):
    dataset = AnswerSegmentDataset(path, tokenizer, max_length, min_target_tokens=min_target_tokens)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_answer_batch,
        pin_memory=torch.cuda.is_available(),
    )
