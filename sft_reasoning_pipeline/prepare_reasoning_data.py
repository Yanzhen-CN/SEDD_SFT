"""
Pipeline adapter for anchored QRA data.

This file reads root-level base rows from prepare_s1k_base.py and only changes
segment masks:
  QRA: fixed question + fixed teacher reasoning -> train answer only
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from transformers import GPT2TokenizerFast

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_DIR))

from prepare_s1k_base import clean, summary, token_count, write_json, write_jsonl  # noqa: E402

DEFAULT_CONFIG = SCRIPT_DIR / "reasoning_config.yaml"
DEFAULT_BASE_DIR = REPO_DIR / "data" / "s1k_base"

QRA_SEGMENT_ORDER = [
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


def _base_meta(row):
    return {
        "reasoning_source": row.get("reasoning_source"),
        "reasoning_field": row.get("reasoning_field"),
        "answer_field": row.get("answer_field"),
        "answer_extract_method": row.get("answer_extract_method"),
        "reasoning_processing": row.get("reasoning_processing"),
        "token_stats": row.get("token_stats"),
    }


def make_qra_sample(row):
    return {
        "id": row["id"],
        "mode": "QRA",
        "source_index": row.get("source_index"),
        "split": row.get("split"),
        "segment_order": QRA_SEGMENT_ORDER,
        "segments": {
            "user_label": segment("User: ", False),
            "user": segment(clean(row.get("question")), False),
            "assistant_label": segment("\nAssistant:\n", False),
            "reasoning_label": segment("Reasoning:\n", False),
            "reasoning": segment(clean(row.get("reasoning")), False),
            "answer_label": segment("\n\nAnswer:\n", False),
            "answer": segment(clean(row.get("answer")), True),
        },
        "answer": clean(row.get("answer")),
        "reasoning": clean(row.get("reasoning")),
        "base_meta": _base_meta(row),
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


def min_target_for(name, config):
    model_cfg = config.get("model", {})
    per_mode = model_cfg.get("min_target_tokens_by_mode", {}) or {}
    return int(per_mode.get(name, model_cfg.get("min_target_tokens", 1)))


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


def filter_by_tokens(split_name, samples, tokenizer, max_length, min_target_tokens, output_dir):
    kept, skipped = [], []
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
            if len(skipped) < 20:
                skipped.append({
                    "id": sample.get("id"),
                    "source_index": sample.get("source_index"),
                    "reason": reason,
                    "total_tokens": total,
                    "train_tokens": train,
                    "condition_tokens": condition,
                    "segment_tokens": segment_tokens,
                    "base_meta": sample.get("base_meta"),
                })
            continue
        kept.append(sample)
        kept_total.append(total)
        kept_train.append(train)

    report = {
        "name": "QRA",
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
        "skipped_examples": skipped,
        "note": "Reasoning is fixed condition in QRA. Cropping was done once in root prepare_s1k_base.py.",
    }
    write_json(Path(output_dir) / "QRA" / f"{split_name}_prepare_filter_report.json", report)
    return kept, report


def load_base_splits(base_dir):
    base_dir = Path(base_dir)
    if not (base_dir / "manifest.json").exists():
        raise FileNotFoundError(
            f"Missing base data at {base_dir}. Run: python prepare_s1k_base.py --config sft_answer_pipeline/answer_config.yaml"
        )
    return {split: read_jsonl(base_dir / f"{split}.jsonl") for split in ("train", "validation", "test")}, json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))


def build(config):
    data_cfg = config.get("data", {})
    base_dir = data_cfg.get("base_dir") or data_cfg.get("base_output_dir") or str(DEFAULT_BASE_DIR)
    output_dir = data_cfg.get("output_dir", str(SCRIPT_DIR / "data"))
    max_length = int(config.get("model", {}).get("max_length", 1024))
    min_target_tokens = min_target_for("QRA", config)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e9)

    base_splits, base_manifest = load_base_splits(base_dir)
    qra_by_split = {split: [make_qra_sample(row) for row in rows] for split, rows in base_splits.items()}

    splits, reports = {}, {}
    for split, rows in qra_by_split.items():
        kept, report = filter_by_tokens(split, rows, tokenizer, max_length, min_target_tokens, output_dir)
        splits[split] = kept
        reports[split] = report

    for split, rows in splits.items():
        write_jsonl(Path(output_dir) / "QRA" / f"{split}.jsonl", rows)

    dataset_summary = {
        "name": "QRA",
        "train": len(splits["train"]),
        "validation": len(splits["validation"]),
        "test": len(splits["test"]),
        "min_target_tokens": min_target_tokens,
        "input_stats_before_token_filter": length_stats([s for rows in qra_by_split.values() for s in rows]),
        "all_stats": length_stats([s for rows in splits.values() for s in rows]),
        "train_stats": length_stats(splits["train"]),
        "validation_stats": length_stats(splits["validation"]),
        "test_stats": length_stats(splits["test"]),
        "token_filter_reports": {split: str(Path(output_dir) / "QRA" / f"{split}_prepare_filter_report.json") for split in splits},
        "token_filter_summary": {
            split: {
                "input_rows": r["input_rows"],
                "kept_rows": r["kept_rows"],
                "skipped_rows": r["skipped_rows"],
                "skip_reasons": r["skip_reasons"],
                "crop_method_counts": r["crop_method_counts"],
                "kept_total_tokens": r["kept_total_tokens"],
            }
            for split, r in reports.items()
        },
    }

    manifest = {
        "format": "Pipeline adapter from base content rows to anchored QRA segment mask.",
        "base_dir": str(base_dir),
        "base_manifest_summary": {
            "base_rows": base_manifest.get("base_rows"),
            "crop_method_counts": base_manifest.get("crop_method_counts"),
            "reasoning_source_counts": base_manifest.get("reasoning_source_counts"),
            "reasoning_field_counts": base_manifest.get("reasoning_field_counts"),
            "answer_extract_counts": base_manifest.get("answer_extract_counts"),
        },
        "max_length": max_length,
        "datasets": [dataset_summary],
    }
    write_json(Path(output_dir) / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Prepare anchored QRA data from root-level S1K base rows.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    build(load_config(args.config))


if __name__ == "__main__":
    main()
