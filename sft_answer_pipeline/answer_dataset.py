import json
from pathlib import Path
from statistics import mean

import torch
from torch.utils.data import Dataset


DEFAULT_SEGMENT_ORDER = [
    "system_label",
    "system",
    "user_label",
    "user",
    "reasoning_label",
    "reasoning",
    "assistant_label",
    "assistant",
    "final_answer_label",
    "final_answer",
]


class DatasetFilteringError(RuntimeError):
    pass


def ordered_segments(sample):
    """Return segments in the explicit per-sample order when present.

    Older files did not store segment_order.  For backward compatibility we fall
    back to DEFAULT_SEGMENT_ORDER and then append any unknown segment names in
    insertion order.
    """
    segments = sample["segments"]
    order = list(sample.get("segment_order") or DEFAULT_SEGMENT_ORDER)
    ordered = []
    used = set()
    for name in order:
        if name in segments:
            ordered.append((name, segments[name]))
            used.add(name)
    for name, segment in segments.items():
        if name not in used:
            ordered.append((name, segment))
    return ordered


def sample_text(sample, train=None):
    parts = []
    for _, segment in ordered_segments(sample):
        if train is None or bool(segment.get("train", False)) is train:
            parts.append(segment.get("text", ""))
    return "".join(parts)


def _percentile(sorted_values, q):
    if not sorted_values:
        return 0
    idx = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * q))))
    return sorted_values[idx]


def _length_summary(values):
    values = list(values)
    if not values:
        return {"count": 0}
    values_sorted = sorted(values)
    return {
        "count": len(values_sorted),
        "min": values_sorted[0],
        "avg": round(mean(values_sorted), 2),
        "p50": _percentile(values_sorted, 0.50),
        "p90": _percentile(values_sorted, 0.90),
        "p95": _percentile(values_sorted, 0.95),
        "max": values_sorted[-1],
    }


class AnswerSegmentDataset(Dataset):
    """JSONL dataset for target-only SEDD SFT.

    Important behavior:
    - prompt/condition tokens have train=False and are kept fixed;
    - target tokens have train=True and are noised/lossed by answer_losses.py;
    - over-length samples are dropped, never silently truncated.
    """

    def __init__(
        self,
        path,
        tokenizer,
        max_length,
        min_target_tokens=1,
        drop_overlength=True,
        write_report=True,
        report_path=None,
        keep_skipped_examples=10,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.min_target_tokens = int(min_target_tokens)
        self.drop_overlength = bool(drop_overlength)
        self.keep_skipped_examples = int(keep_skipped_examples)
        self.samples = []
        self.encoded_samples = []
        self.skipped_examples = []
        self.report = {
            "path": str(self.path),
            "max_length": self.max_length,
            "min_target_tokens": self.min_target_tokens,
            "drop_overlength": self.drop_overlength,
            "total_rows": 0,
            "kept_rows": 0,
            "skipped_rows": 0,
            "skip_reasons": {
                "missing_segments": 0,
                "no_train_tokens": 0,
                "short_target": 0,
                "overlength": 0,
            },
            "kept_length_stats": {},
            "all_length_stats": {},
            "skipped_examples": self.skipped_examples,
        }

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        all_total_lens = []
        all_train_lens = []
        all_condition_lens = []
        kept_total_lens = []
        kept_train_lens = []
        kept_condition_lens = []

        with open(self.path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                self.report["total_rows"] += 1
                sample = json.loads(line)
                if not isinstance(sample, dict) or "segments" not in sample:
                    self._skip(sample, line_idx, "missing_segments", 0, 0, 0)
                    continue

                encoded = self._tokenize_sample(sample)
                total_len = len(encoded["input_ids_unpadded"])
                train_len = int(sum(encoded["train_mask_unpadded"]))
                condition_len = total_len - train_len

                all_total_lens.append(total_len)
                all_train_lens.append(train_len)
                all_condition_lens.append(condition_len)

                if train_len <= 0:
                    self._skip(sample, line_idx, "no_train_tokens", total_len, train_len, condition_len)
                    continue
                if train_len < self.min_target_tokens:
                    self._skip(sample, line_idx, "short_target", total_len, train_len, condition_len)
                    continue
                if total_len > self.max_length:
                    if self.drop_overlength:
                        self._skip(sample, line_idx, "overlength", total_len, train_len, condition_len)
                        continue
                    raise DatasetFilteringError(
                        f"Over-length sample {sample.get('id', line_idx)} has {total_len} tokens, "
                        f"which exceeds max_length={self.max_length}."
                    )

                self.samples.append(sample)
                self.encoded_samples.append(encoded)
                kept_total_lens.append(total_len)
                kept_train_lens.append(train_len)
                kept_condition_lens.append(condition_len)

        self.report["kept_rows"] = len(self.samples)
        self.report["skipped_rows"] = self.report["total_rows"] - self.report["kept_rows"]
        self.report["all_length_stats"] = {
            "total_tokens": _length_summary(all_total_lens),
            "condition_tokens": _length_summary(all_condition_lens),
            "train_tokens": _length_summary(all_train_lens),
        }
        self.report["kept_length_stats"] = {
            "total_tokens": _length_summary(kept_total_lens),
            "condition_tokens": _length_summary(kept_condition_lens),
            "train_tokens": _length_summary(kept_train_lens),
        }

        if write_report:
            if report_path is None:
                report_path = self.path.with_name(f"{self.path.stem}_load_report.json")
            self.write_report(report_path)

        print(
            f"[AnswerSegmentDataset] {self.path}: kept={self.report['kept_rows']} / "
            f"total={self.report['total_rows']}; skipped={self.report['skipped_rows']} "
            f"{self.report['skip_reasons']}"
        )

        if not self.samples:
            raise ValueError(
                f"No usable samples after token-length filtering for {self.path}. "
                f"Report: {json.dumps(self.report, ensure_ascii=False)}"
            )

    def _skip(self, sample, line_idx, reason, total_len, train_len, condition_len):
        self.report["skip_reasons"][reason] += 1
        if len(self.skipped_examples) < self.keep_skipped_examples:
            sample_id = sample.get("id", str(line_idx)) if isinstance(sample, dict) else str(line_idx)
            self.skipped_examples.append(
                {
                    "line_idx": line_idx,
                    "id": sample_id,
                    "reason": reason,
                    "total_tokens": int(total_len),
                    "condition_tokens": int(condition_len),
                    "train_tokens": int(train_len),
                }
            )

    def _tokenize_sample(self, sample):
        ids = []
        train_mask = []
        segment_token_lens = {}
        for name, segment in ordered_segments(sample):
            text = segment.get("text", "")
            token_ids = self.tokenizer(text, add_special_tokens=False).input_ids
            is_train = bool(segment.get("train", False))
            ids.extend(token_ids)
            train_mask.extend([1 if is_train else 0] * len(token_ids))
            segment_token_lens[name] = len(token_ids)
        return {
            "input_ids_unpadded": ids,
            "train_mask_unpadded": train_mask,
            "segment_token_lens": segment_token_lens,
        }

    def write_report(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8")

    def __len__(self):
        return len(self.samples)

    def _encode_item(self, idx):
        sample = self.samples[idx]
        encoded = self.encoded_samples[idx]
        ids = list(encoded["input_ids_unpadded"])
        train_mask = list(encoded["train_mask_unpadded"])
        real_len = len(ids)
        train_len = int(sum(train_mask))
        condition_len = real_len - train_len

        pad_len = self.max_length - real_len
        if pad_len < 0:
            raise DatasetFilteringError(
                f"Internal error: kept over-length sample {sample.get('id', idx)} "
                f"with {real_len}>{self.max_length}."
            )
        if pad_len > 0:
            ids += [self.tokenizer.eos_token_id] * pad_len
            train_mask += [0] * pad_len
        attention_mask = [1] * real_len + [0] * pad_len

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "train_mask": torch.tensor(train_mask, dtype=torch.bool),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "condition_len": condition_len,
            "train_len": train_len,
            "real_len": real_len,
            "segment_token_lens": encoded["segment_token_lens"],
        }

    def __getitem__(self, idx):
        sample = self.samples[idx]
        item = self._encode_item(idx)
        item["id"] = sample.get("id", str(idx))
        item["mode"] = sample.get("mode", "")
        item["prompt"] = sample_text(sample, train=False)
        item["target"] = sample_text(sample, train=True)
        item["answer"] = sample.get("answer", sample["segments"].get("assistant", {}).get("text", ""))
        item["reasoning"] = sample.get("reasoning", sample["segments"].get("reasoning", {}).get("text", ""))
        item["segments"] = sample["segments"]
        item["segment_order"] = sample.get("segment_order", DEFAULT_SEGMENT_ORDER)
        return item


def collate_answer_batch(items):
    tensor_keys = ["input_ids", "train_mask", "attention_mask"]
    batch = {key: torch.stack([item[key] for item in items], dim=0) for key in tensor_keys}
    batch["ids"] = [item["id"] for item in items]
    batch["modes"] = [item["mode"] for item in items]
    batch["prompts"] = [item["prompt"] for item in items]
    batch["targets"] = [item["target"] for item in items]
    batch["answers"] = [item["answer"] for item in items]
    batch["reasonings"] = [item["reasoning"] for item in items]
    batch["segments"] = [item["segments"] for item in items]
    batch["segment_orders"] = [item["segment_order"] for item in items]
    batch["condition_lens"] = torch.tensor([item["condition_len"] for item in items], dtype=torch.long)
    batch["train_lens"] = torch.tensor([item["train_len"] for item in items], dtype=torch.long)
    batch["real_lens"] = torch.tensor([item["real_len"] for item in items], dtype=torch.long)
    return batch


def make_answer_loader(
    path,
    tokenizer,
    max_length,
    min_target_tokens,
    batch_size,
    shuffle,
    num_workers,
    drop_overlength=True,
    write_report=True,
    report_path=None,
):
    dataset = AnswerSegmentDataset(
        path,
        tokenizer,
        max_length,
        min_target_tokens=min_target_tokens,
        drop_overlength=drop_overlength,
        write_report=write_report,
        report_path=report_path,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_answer_batch,
        pin_memory=torch.cuda.is_available(),
    )
