import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


SEGMENT_ORDER = ["user_label", "user", "assistant_label", "assistant", "reasoning_label", "reasoning"]


def ordered_segments(sample):
    segments = sample["segments"]
    return [(name, segments[name]) for name in SEGMENT_ORDER if name in segments]


def sample_text(sample, train=None):
    parts = []
    for _, segment in ordered_segments(sample):
        if train is None or bool(segment["train"]) is train:
            parts.append(segment["text"])
    return "".join(parts)


class AnswerSegmentDataset(Dataset):
    def __init__(self, path, tokenizer, max_length, min_target_tokens=1):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.min_target_tokens = int(min_target_tokens)
        self.samples = []

        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                if not isinstance(sample, dict) or "segments" not in sample:
                    raise TypeError(f"Expected each JSONL line to be a sample object with a segments field: {self.path}")
                self.samples.append(sample)

        if not self.samples:
            raise ValueError(f"No samples found in {self.path}")

    def __len__(self):
        return len(self.samples)

    def _encode_sample(self, sample):
        encoded = []
        for name, segment in ordered_segments(sample):
            token_ids = self.tokenizer(segment["text"], add_special_tokens=False).input_ids
            encoded.append({"name": name, "ids": token_ids, "train": bool(segment["train"])})

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
            "condition_len": sum(len(item["ids"]) for item in encoded if not item["train"]),
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

        by_name = {item["name"]: item for item in encoded}
        prefix_ids = []
        for name in ["user_label", "user", "assistant_label"]:
            if name in by_name:
                prefix_ids.extend(by_name[name]["ids"])

        assistant_ids = by_name.get("assistant", {"ids": []})["ids"]
        reasoning_label_ids = by_name.get("reasoning_label", {"ids": []})["ids"]
        reasoning_ids = by_name.get("reasoning", {"ids": []})["ids"]

        if not assistant_ids:
            return self._truncate_generic(encoded)

        reserved_reasoning = self.min_target_tokens if reasoning_ids else 0
        prefix_budget = max(1, self.max_length - reserved_reasoning - len(reasoning_label_ids) - self.min_target_tokens)
        prefix_ids = prefix_ids[-prefix_budget:]

        assistant_budget = self.max_length - len(prefix_ids) - len(reasoning_label_ids) - reserved_reasoning
        assistant_ids = assistant_ids[:max(1, assistant_budget)]
        remaining = self.max_length - len(prefix_ids) - len(assistant_ids) - len(reasoning_label_ids)
        reasoning_ids = reasoning_ids[:max(0, remaining)]

        ids = prefix_ids + assistant_ids + reasoning_label_ids + reasoning_ids
        train_mask = (
            [0] * len(prefix_ids)
            + [1] * len(assistant_ids)
            + [0] * len(reasoning_label_ids)
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
        sample = self.samples[idx]
        item = self._encode_sample(sample)
        item["id"] = sample.get("id", str(idx))
        item["mode"] = sample.get("mode", "")
        item["prompt"] = "".join(
            sample["segments"][name]["text"]
            for name in ["user_label", "user", "assistant_label"]
            if name in sample["segments"]
        )
        item["target"] = sample_text(sample, train=True)
        item["answer"] = sample["segments"].get("assistant", {}).get("text", "")
        item["segments"] = sample["segments"]
        return item


def collate_answer_batch(items):
    tensor_keys = ["input_ids", "train_mask", "attention_mask"]
    batch = {key: torch.stack([item[key] for item in items], dim=0) for key in tensor_keys}
    batch["ids"] = [item["id"] for item in items]
    batch["modes"] = [item["mode"] for item in items]
    batch["prompts"] = [item["prompt"] for item in items]
    batch["targets"] = [item["target"] for item in items]
    batch["answers"] = [item["answer"] for item in items]
    batch["segments"] = [item["segments"] for item in items]
    batch["condition_lens"] = torch.tensor([item["condition_len"] for item in items], dtype=torch.long)
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
