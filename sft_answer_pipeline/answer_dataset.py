import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


SEGMENT_ORDER = [
    "user_label",
    "user",
    "assistant_label",
    "assistant",
    "reasoning_label",
    "reasoning",
]


def ordered_segments(sample: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    segments = sample["segments"]
    return [(name, segments[name]) for name in SEGMENT_ORDER if name in segments]


def sample_text(sample: Dict[str, Any], train: Optional[bool] = None) -> str:
    parts: List[str] = []
    for _, segment in ordered_segments(sample):
        if train is None or bool(segment["train"]) is train:
            parts.append(segment["text"])
    return "".join(parts)


def _percentile(sorted_values: List[int], q: float) -> int:
    if not sorted_values:
        return 0
    idx = int(round((len(sorted_values) - 1) * q))
    idx = max(0, min(idx, len(sorted_values) - 1))
    return int(sorted_values[idx])


def _stats(values: Iterable[int]) -> Dict[str, Any]:
    vals = sorted(int(v) for v in values)
    if not vals:
        return {
            "count": 0,
            "avg": 0,
            "min": 0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
        }
    return {
        "count": len(vals),
        "avg": round(float(mean(vals)), 2),
        "min": vals[0],
        "p50": _percentile(vals, 0.50),
        "p90": _percentile(vals, 0.90),
        "p95": _percentile(vals, 0.95),
        "p99": _percentile(vals, 0.99),
        "max": vals[-1],
    }


class AnswerSegmentDataset(Dataset):
    """Dataset for SEDD answer/reasoning SFT.

    Important behavior:
    - We DO NOT truncate any sample.
    - If condition tokens + train tokens exceed max_length, the whole sample is dropped.
    - Prompt/condition tokens keep train_mask=False.
    - Assistant/reasoning target tokens keep train_mask=True.
    - Padding tokens are EOS tokens with train_mask=False and attention_mask=False.

    This avoids the dangerous failure mode where long QAR examples keep only the
    beginning of reasoning and silently lose the final answer.
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        max_length: int,
        min_target_tokens: int = 1,
        drop_overlength: bool = True,
        report_path: Optional[str] = None,
        verbose: bool = True,
        keep_overlength_examples: int = 20,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.min_target_tokens = int(min_target_tokens)
        self.drop_overlength = bool(drop_overlength)
        self.verbose = bool(verbose)
        self.keep_overlength_examples = int(keep_overlength_examples)

        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        if self.min_target_tokens <= 0:
            raise ValueError(f"min_target_tokens must be positive, got {self.min_target_tokens}")
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {self.path}")
        if getattr(self.tokenizer, "eos_token_id", None) is None:
            raise ValueError("tokenizer.eos_token_id is required for padding")

        self.items: List[Dict[str, Any]] = []
        self._all_total_lens: List[int] = []
        self._all_condition_lens: List[int] = []
        self._all_train_lens: List[int] = []
        self._kept_total_lens: List[int] = []
        self._kept_condition_lens: List[int] = []
        self._kept_train_lens: List[int] = []
        self._overlength_examples: List[Dict[str, Any]] = []

        counters = {
            "total_rows": 0,
            "kept_rows": 0,
            "dropped_bad_format": 0,
            "dropped_empty_target": 0,
            "dropped_short_target": 0,
            "dropped_overlength": 0,
        }

        with open(self.path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                counters["total_rows"] += 1

                try:
                    sample = json.loads(line)
                    encoded, condition_len, train_len, total_len = self._tokenize_sample(sample)
                except Exception as exc:
                    counters["dropped_bad_format"] += 1
                    if len(self._overlength_examples) < self.keep_overlength_examples:
                        self._overlength_examples.append(
                            {
                                "line_no": line_no,
                                "id": None,
                                "mode": None,
                                "reason": "bad_format",
                                "error": repr(exc),
                            }
                        )
                    continue

                self._all_total_lens.append(total_len)
                self._all_condition_lens.append(condition_len)
                self._all_train_lens.append(train_len)

                if train_len == 0:
                    counters["dropped_empty_target"] += 1
                    continue
                if train_len < self.min_target_tokens:
                    counters["dropped_short_target"] += 1
                    continue
                if total_len > self.max_length:
                    counters["dropped_overlength"] += 1
                    if len(self._overlength_examples) < self.keep_overlength_examples:
                        self._overlength_examples.append(
                            {
                                "line_no": line_no,
                                "id": sample.get("id", str(line_no - 1)),
                                "mode": sample.get("mode", ""),
                                "reason": "overlength",
                                "total_len": total_len,
                                "condition_len": condition_len,
                                "train_len": train_len,
                                "max_length": self.max_length,
                            }
                        )
                    if self.drop_overlength:
                        continue
                    raise ValueError(
                        f"Overlength sample at line {line_no}: "
                        f"total_len={total_len} > max_length={self.max_length}. "
                        "Set drop_overlength=True to drop it."
                    )

                counters["kept_rows"] += 1
                self._kept_total_lens.append(total_len)
                self._kept_condition_lens.append(condition_len)
                self._kept_train_lens.append(train_len)
                self.items.append(
                    {
                        "sample": sample,
                        "encoded": encoded,
                        "condition_len": condition_len,
                        "train_len": train_len,
                        "total_len": total_len,
                    }
                )

        self.report = self._build_report(counters)
        if not self.items:
            raise ValueError(
                f"No usable samples after filtering {self.path}. "
                f"Report: {json.dumps(self.report, ensure_ascii=False)}"
            )

        if report_path is None:
            report_path = str(self.path.with_suffix(".load_report.json"))
        self.report_path = Path(report_path)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(
            json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if self.verbose:
            print(
                "[AnswerSegmentDataset] "
                f"path={self.path} max_length={self.max_length} "
                f"kept={self.report['kept_rows']} / total={self.report['total_rows']} "
                f"dropped_overlength={self.report['dropped_overlength']} "
                f"dropped_short_target={self.report['dropped_short_target']} "
                f"report={self.report_path}"
            )

    def _tokenize_sample(
        self, sample: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], int, int, int]:
        if not isinstance(sample, dict) or "segments" not in sample:
            raise TypeError(
                f"Expected each JSONL line to be a sample object with a segments field: {self.path}"
            )

        encoded: List[Dict[str, Any]] = []
        condition_len = 0
        train_len = 0
        for name, segment in ordered_segments(sample):
            if not isinstance(segment, dict) or "text" not in segment or "train" not in segment:
                raise TypeError(f"Bad segment format for segment {name}: {segment}")
            token_ids = self.tokenizer(
                str(segment["text"]), add_special_tokens=False
            ).input_ids
            is_train = bool(segment["train"])
            encoded.append({"name": name, "ids": token_ids, "train": is_train})
            if is_train:
                train_len += len(token_ids)
            else:
                condition_len += len(token_ids)
        total_len = condition_len + train_len
        return encoded, condition_len, train_len, total_len

    def _build_report(self, counters: Dict[str, int]) -> Dict[str, Any]:
        total_rows = counters["total_rows"]
        kept_rows = counters["kept_rows"]
        return {
            "path": str(self.path),
            "max_length": self.max_length,
            "min_target_tokens": self.min_target_tokens,
            "drop_overlength": self.drop_overlength,
            **counters,
            "effective_sample_ratio": round(kept_rows / total_rows, 6) if total_rows else 0.0,
            "all_length_stats": {
                "total_len": _stats(self._all_total_lens),
                "condition_len": _stats(self._all_condition_lens),
                "train_len": _stats(self._all_train_lens),
            },
            "kept_length_stats": {
                "total_len": _stats(self._kept_total_lens),
                "condition_len": _stats(self._kept_condition_lens),
                "train_len": _stats(self._kept_train_lens),
            },
            "examples": self._overlength_examples,
        }

    def get_report(self) -> Dict[str, Any]:
        return self.report

    def __len__(self) -> int:
        return len(self.items)

    def _encode_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        ids: List[int] = []
        train_mask: List[int] = []
        for seg in item["encoded"]:
            seg_ids = seg["ids"]
            ids.extend(seg_ids)
            train_mask.extend([1 if seg["train"] else 0] * len(seg_ids))

        if len(ids) > self.max_length:
            # This should never happen because overlength rows are filtered in __init__.
            raise ValueError(
                f"Internal error: overlength item survived filtering: "
                f"len(ids)={len(ids)} > max_length={self.max_length}"
            )
        if sum(train_mask) == 0:
            raise ValueError("Internal error: item has no train tokens after filtering")

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
            "condition_len": item["condition_len"],
            "train_len": item["train_len"],
            "total_len": item["total_len"],
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        sample = item["sample"]
        result = self._encode_item(item)
        result["id"] = sample.get("id", str(idx))
        result["mode"] = sample.get("mode", "")
        result["prompt"] = "".join(
            sample["segments"][name]["text"]
            for name in ["user_label", "user", "assistant_label"]
            if name in sample["segments"]
        )
        result["target"] = sample_text(sample, train=True)
        result["answer"] = sample["segments"].get("assistant", {}).get("text", "")
        result["segments"] = sample["segments"]
        return result


# Backward-compatible alias in case older training files import this name.
AnswerJsonlDataset = AnswerSegmentDataset


def collate_answer_batch(items: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    batch["total_lens"] = torch.tensor([item["total_len"] for item in items], dtype=torch.long)
    return batch


def make_answer_loader(
    path: str,
    tokenizer,
    max_length: int,
    min_target_tokens: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_overlength: bool = True,
    report_path: Optional[str] = None,
    verbose: bool = True,
):
    dataset = AnswerSegmentDataset(
        path,
        tokenizer,
        max_length,
        min_target_tokens=min_target_tokens,
        drop_overlength=drop_overlength,
        report_path=report_path,
        verbose=verbose,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_answer_batch,
        pin_memory=torch.cuda.is_available(),
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Load and audit SEDD answer SFT JSONL data.")
    parser.add_argument("--path", required=True, help="Path to train/validation/test JSONL.")
    parser.add_argument("--tokenizer_name", required=True, help="HF tokenizer name or local tokenizer path.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--min_target_tokens", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--report_path", default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.eos_token_id is None and tokenizer.pad_token_id is not None:
        tokenizer.eos_token = tokenizer.pad_token

    loader = make_answer_loader(
        path=args.path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        min_target_tokens=args.min_target_tokens,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_overlength=True,
        report_path=args.report_path,
        verbose=True,
    )
    first_batch = next(iter(loader))
    print(
        json.dumps(
            {
                "first_batch_shape": list(first_batch["input_ids"].shape),
                "first_batch_ids": first_batch["ids"],
                "first_batch_train_lens": first_batch["train_lens"].tolist(),
                "first_batch_condition_lens": first_batch["condition_lens"].tolist(),
                "first_batch_total_lens": first_batch["total_lens"].tolist(),
                "report": loader.dataset.get_report(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    _main()
