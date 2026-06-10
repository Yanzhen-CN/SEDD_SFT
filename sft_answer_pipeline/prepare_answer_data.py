"""
Pipeline adapter for anchored QA / QAR SEDD SFT data.

This file no longer reads raw S1K directly. It reads root-level S1K_light rows from
prepare_s1k_split.py and only converts them into segment masks:
  QA:  train answer only
  QAR: train reasoning + answer
"""

import argparse
import json
import sys
import shutil
from pathlib import Path

import yaml
from transformers import GPT2TokenizerFast

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_DIR))

from prepare_s1k_split import clean, summary, token_count, write_json, write_jsonl  # noqa: E402

DEFAULT_CONFIG = SCRIPT_DIR / "answer_config.yaml"
DEFAULT_BASE_DIR = REPO_DIR / "data" / "S1K_light"

QA_SEGMENT_ORDER = ["user_label", "user", "assistant_label", "answer_label", "answer"]
QAR_SEGMENT_ORDER = [
    "user_label",
    "user",
    "assistant_label",
    "reasoning_label",
    "reasoning",
    "answer_label",
    "answer",
]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def segment(text, train):
    return {"text": str(text), "train": bool(train)}


def make_qa_sample(row):
    sample = {
        "id": row["id"],
        "mode": "QA",
        "source_index": row.get("source_index"),
        "split": row.get("split"),
        "segment_order": QA_SEGMENT_ORDER,
        "segments": {
            "user_label": segment("User: ", False),
            "user": segment(clean(row.get("question")), False),
            "assistant_label": segment("\nAssistant:\n", False),
            "answer_label": segment("Answer:\n", False),
            "answer": segment(clean(row.get("answer")), True),
        },
        "answer": clean(row.get("answer")),
        "reasoning": "",
        "base_meta": _base_meta(row),
    }
    return sample


def make_qar_sample(row):
    sample = {
        "id": row["id"],
        "mode": "QAR",
        "source_index": row.get("source_index"),
        "split": row.get("split"),
        "segment_order": QAR_SEGMENT_ORDER,
        "segments": {
            "user_label": segment("User: ", False),
            "user": segment(clean(row.get("question")), False),
            "assistant_label": segment("\nAssistant:\n", False),
            "reasoning_label": segment("Reasoning:\n", False),
            "reasoning": segment(clean(row.get("reasoning")), True),
            "answer_label": segment("\n\nAnswer:\n", False),
            "answer": segment(clean(row.get("answer")), True),
        },
        "answer": clean(row.get("answer")),
        "reasoning": clean(row.get("reasoning")),
        "base_meta": _base_meta(row),
    }
    return sample


def _base_meta(row):
    return {
        "reasoning_source": row.get("reasoning_source"),
        "reasoning_field": row.get("reasoning_field"),
        "answer_field": row.get("answer_field"),
        "answer_extract_method": row.get("answer_extract_method"),
        "reasoning_processing": row.get("reasoning_processing"),
        "token_stats": row.get("token_stats"),
    }


def sample_text(sample, train=None):
    parts = []
    for name in sample.get("segment_order", []):
        seg = sample["segments"].get(name)
        if seg is None:
            continue
        if train is None or bool(seg.get("train", False)) is train:
            parts.append(seg.get("text", ""))
    return "".join(parts)


def token_lengths(sample, tokenizer):
    total = train = 0
    segment_tokens = {}
    for name in sample.get("segment_order", []):
        seg = sample["segments"].get(name)
        if seg is None:
            continue
        n = token_count(tokenizer, seg.get("text", ""))
        segment_tokens[name] = n
        total += n
        if bool(seg.get("train", False)):
            train += n
    return total, train, total - train, segment_tokens


def length_stats(samples):
    lengths = sorted(len(sample_text(s)) for s in samples)
    train_lengths = sorted(len(sample_text(s, train=True)) for s in samples)
    if not lengths:
        return {"count": 0}
    return {
        "count": len(samples),
        "avg_chars": round(sum(lengths) / len(lengths), 1),
        "p50_chars": lengths[(len(lengths) - 1) // 2],
        "p90_chars": lengths[round((len(lengths) - 1) * 0.90)],
        "p95_chars": lengths[round((len(lengths) - 1) * 0.95)],
        "max_chars": lengths[-1],
        "avg_train_chars": round(sum(train_lengths) / len(train_lengths), 1),
        "p90_train_chars": train_lengths[round((len(train_lengths) - 1) * 0.90)],
        "p95_train_chars": train_lengths[round((len(train_lengths) - 1) * 0.95)],
        "max_train_chars": train_lengths[-1],
    }


def min_target_for(name, config):
    model_cfg = config.get("model", {})
    per_mode = model_cfg.get("min_target_tokens_by_mode", {}) or {}
    return int(per_mode.get(name, model_cfg.get("min_target_tokens", 1)))


def filter_by_tokens(name, split_name, samples, tokenizer, max_length, min_target_tokens):
    kept = []
    all_total, all_train, kept_total, kept_train = [], [], [], []
    reasons = {"overlength": 0, "short_target": 0, "no_train_tokens": 0}
    crop_methods = {}
    for sample in samples:
        total, train, condition, segment_tokens = token_lengths(sample, tokenizer)
        all_total.append(total)
        all_train.append(train)
        method = (sample.get("base_meta") or {}).get("reasoning_processing", {}).get("method")
        if method:
            crop_methods[method] = crop_methods.get(method, 0) + 1
        reason = None
        if train <= 0:
            reason = "no_train_tokens"
        elif train < min_target_tokens:
            reason = "short_target"
        elif total > max_length:
            reason = "overlength"
        if reason:
            reasons[reason] += 1
            continue
        kept.append(sample)
        kept_total.append(total)
        kept_train.append(train)

    # Keep all diagnostics inside the pipeline-level manifest only.
    # Do not create per-split report JSON files; each dataset folder should
    # contain only train.jsonl / validation.jsonl / test.jsonl.
    report = {
        "name": name,
        "split": split_name,
        "max_length": max_length,
        "min_target_tokens": min_target_tokens,
        "input_rows": len(samples),
        "kept_rows": len(kept),
        "skipped_rows": len(samples) - len(kept),
        "skip_reasons": reasons,
        "crop_method_counts": crop_methods,
        "all_total_tokens": summary(all_total),
        "all_train_tokens": summary(all_train),
        "kept_total_tokens": summary(kept_total),
        "kept_train_tokens": summary(kept_train),
    }
    return kept, report


def save_dataset(name, split_samples, output_dir, tokenizer, max_length, min_target_tokens):
    splits, reports = {}, {}
    input_samples = []
    for split, samples in split_samples.items():
        input_samples.extend(samples)
        kept, report = filter_by_tokens(name, split, samples, tokenizer, max_length, min_target_tokens)
        splits[split] = kept
        reports[split] = report

    base = Path(output_dir) / name
    if base.exists():
        shutil.rmtree(base)
    for split, rows in splits.items():
        write_jsonl(base / f"{split}.jsonl", rows)

    return {
        "name": name,
        "train": len(splits["train"]),
        "validation": len(splits["validation"]),
        "test": len(splits["test"]),
        "min_target_tokens": min_target_tokens,
        "input_stats_before_token_filter": length_stats(input_samples),
        "all_stats": length_stats([s for rows in splits.values() for s in rows]),
        "train_stats": length_stats(splits["train"]),
        "validation_stats": length_stats(splits["validation"]),
        "test_stats": length_stats(splits["test"]),
        "token_filter_summary": {
            split: {
                "input_rows": r["input_rows"],
                "kept_rows": r["kept_rows"],
                "skipped_rows": r["skipped_rows"],
                "skip_reasons": r["skip_reasons"],
                "crop_method_counts": r["crop_method_counts"],
                "kept_total_tokens": r["kept_total_tokens"],
                "kept_train_tokens": r["kept_train_tokens"],
            }
            for split, r in reports.items()
        },
    }


def load_base_splits(light_dir):
    light_dir = Path(light_dir)
    if not (light_dir / "manifest.json").exists():
        raise FileNotFoundError(
            f"Missing S1K_light data at {light_dir}. Run: python prepare_s1k_split.py --config sft_answer_pipeline/answer_config.yaml"
        )
    return {split: read_jsonl(light_dir / f"{split}.jsonl") for split in ("train", "validation", "test")}, json.loads((light_dir / "manifest.json").read_text(encoding="utf-8"))


def build(config):
    data_cfg = config.get("data", {})
    light_dir = (
        data_cfg.get("light_dir")
        or data_cfg.get("light_output_dir")
        or data_cfg.get("base_dir")
        or data_cfg.get("base_output_dir")
        or str(DEFAULT_BASE_DIR)
    )
    output_dir = data_cfg.get("output_dir", str(SCRIPT_DIR / "data"))
    max_length = int(config.get("model", {}).get("max_length", 1024))

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e9)

    base_splits, base_manifest = load_base_splits(light_dir)

    builders = {
        "QA": make_qa_sample,
        "QAR": make_qar_sample,
    }
    split_samples = {name: {"train": [], "validation": [], "test": []} for name in builders}
    for split, rows in base_splits.items():
        for row in rows:
            for name, fn in builders.items():
                split_samples[name][split].append(fn(row))

    manifest = {
        "format": "Pipeline adapter from S1K_light content rows to anchored QA/QAR segment masks.",
        "light_dir": str(light_dir),
        "base_manifest_summary": {
            "light_rows": base_manifest.get("light_rows") or base_manifest.get("base_rows"),
            "crop_method_counts": base_manifest.get("crop_method_counts"),
            "reasoning_source_counts": base_manifest.get("reasoning_source_counts"),
            "reasoning_field_counts": base_manifest.get("reasoning_field_counts"),
            "answer_extract_counts": base_manifest.get("answer_extract_counts"),
        },
        "max_length": max_length,
        "datasets": [
            save_dataset(name, samples, output_dir, tokenizer, max_length, min_target_for(name, config))
            for name, samples in split_samples.items()
        ],
    }
    write_json(Path(output_dir) / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Prepare anchored QA/QAR data from root-level split/base S1K rows.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    build(load_config(args.config))


if __name__ == "__main__":
    main()
