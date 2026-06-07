import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class AnswerJsonlDataset(Dataset):
    def __init__(self, path, tokenizer, max_length, min_target_tokens=1):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.min_target_tokens = int(min_target_tokens)
        self.rows = []

        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

        if not self.rows:
            raise ValueError(f"No rows found in {self.path}")

    def __len__(self):
        return len(self.rows)

    def _encode(self, row):
        prompt_ids = self.tokenizer(row["prompt"], add_special_tokens=False).input_ids
        target_ids = self.tokenizer(row["target"], add_special_tokens=False).input_ids

        if len(prompt_ids) + self.min_target_tokens > self.max_length:
            keep_prompt = max(1, self.max_length - self.min_target_tokens)
            prompt_ids = prompt_ids[-keep_prompt:]

        target_budget = self.max_length - len(prompt_ids)
        target_ids = target_ids[:target_budget]
        if not target_ids:
            target_ids = [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + target_ids
        answer_mask = [0] * len(prompt_ids) + [1] * len(target_ids)
        attention_mask = [1] * len(input_ids)

        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.eos_token_id] * pad_len
            answer_mask += [0] * pad_len
            attention_mask += [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "answer_mask": torch.tensor(answer_mask, dtype=torch.bool),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "prompt_len": len(prompt_ids),
            "target_len": int(sum(answer_mask)),
        }

    def __getitem__(self, idx):
        row = self.rows[idx]
        item = self._encode(row)
        item["id"] = row.get("id", str(idx))
        item["prompt"] = row["prompt"]
        item["target"] = row["target"]
        item["answer"] = row.get("answer", "")
        return item


def collate_answer_batch(items):
    tensor_keys = ["input_ids", "answer_mask", "attention_mask"]
    batch = {key: torch.stack([item[key] for item in items], dim=0) for key in tensor_keys}
    batch["ids"] = [item["id"] for item in items]
    batch["prompts"] = [item["prompt"] for item in items]
    batch["targets"] = [item["target"] for item in items]
    batch["answers"] = [item["answer"] for item in items]
    batch["prompt_lens"] = torch.tensor([item["prompt_len"] for item in items], dtype=torch.long)
    batch["target_lens"] = torch.tensor([item["target_len"] for item in items], dtype=torch.long)
    return batch


def make_answer_loader(path, tokenizer, max_length, min_target_tokens, batch_size, shuffle, num_workers):
    dataset = AnswerJsonlDataset(path, tokenizer, max_length, min_target_tokens=min_target_tokens)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_answer_batch,
        pin_memory=torch.cuda.is_available(),
    )
