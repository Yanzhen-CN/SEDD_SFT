import argparse
import json
import random
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "reasoning_config.yaml"

# Same textual layout as QAR:
#   User: <question>\nAssistant:\nReasoning:\n<teacher reasoning>\n\nAnswer:\n<answer>
# Difference from QAR: reasoning is fixed condition (train=False); only answer
# text is target.  The Answer label is also fixed so generation starts exactly
# where the answer should begin.
RA_SEGMENT_ORDER = [
    "user_label",
    "user",
    "assistant_label",
    "reasoning_label",
    "reasoning",
    "answer_label",
    "answer",
]


def clean(text):
    return str(text or "").strip()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_s1k(data_cfg):
    if data_cfg.get("arrow_path"):
        from datasets import Dataset
        return Dataset.from_file(data_cfg["arrow_path"])
    from datasets import load_dataset
    return load_dataset(data_cfg.get("source_dataset", "simplescaling/s1K-1.1"), split="train")


def choose_reasoning(row, priority):
    fields = {
        "deepseek": clean(row.get("deepseek_thinking_trajectory")),
        "gemini": clean(row.get("gemini_thinking_trajectory")),
    }
    for source in priority:
        if fields.get(source):
            return source, fields[source]
    return None, None


def choose_answer(row, source, data_cfg):
    fallback_field = data_cfg.get("answer_field", "solution")
    fallback = clean(row.get(fallback_field))
    if data_cfg.get("answer_source", "solution") == "matched_attempt" and source:
        attempt = clean(row.get(f"{source}_attempt"))
        if attempt:
            return attempt, f"{source}_attempt"
    return fallback, fallback_field


def split_indices(n_rows, valid_ratio, test_ratio, seed):
    indices = list(range(n_rows))
    random.Random(seed).shuffle(indices)
    n_valid = max(1, int(n_rows * valid_ratio)) if n_rows > 1 and valid_ratio > 0 else 0
    n_test = max(1, int(n_rows * test_ratio)) if n_rows > 1 and test_ratio > 0 else 0
    if n_valid + n_test >= n_rows:
        raise ValueError("valid_ratio + test_ratio leaves no training data.")
    return set(indices[:n_valid]), set(indices[n_valid : n_valid + n_test])


def segment(text, train):
    return {"text": str(text), "train": bool(train)}


def make_ra_sample(row_id, question, answer, reasoning, meta=None):
    segments = {
        "user_label": segment("User: ", False),
        "user": segment(clean(question), False),
        "assistant_label": segment("\nAssistant:\n", False),
        "reasoning_label": segment("Reasoning:\n", False),
        "reasoning": segment(clean(reasoning), False),
        "answer_label": segment("\n\nAnswer:\n", False),
        "answer": segment(clean(answer), True),
    }
    return {
        "id": row_id,
        "mode": "RA",
        "segment_order": RA_SEGMENT_ORDER,
        "segments": segments,
        "answer": clean(answer),
        "reasoning": clean(reasoning),
        **(meta or {}),
    }


def sample_text(sample, train=None):
    parts = []
    for name in sample.get("segment_order", []):
        seg = sample["segments"].get(name)
        if seg is None:
            continue
        if train is None or bool(seg["train"]) is train:
            parts.append(seg["text"])
    return "".join(parts)


def _percentile(sorted_values, q):
    if not sorted_values:
        return 0
    idx = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * q))))
    return sorted_values[idx]


def length_stats(samples):
    lengths = sorted(len(sample_text(s)) for s in samples)
    train_lengths = sorted(len(sample_text(s, train=True)) for s in samples)
    if not lengths:
        return {"count": 0}
    return {
        "count": len(samples),
        "avg_chars": round(sum(lengths) / len(lengths), 1),
        "p50_chars": _percentile(lengths, 0.50),
        "p90_chars": _percentile(lengths, 0.90),
        "p95_chars": _percentile(lengths, 0.95),
        "max_chars": lengths[-1],
        "avg_train_chars": round(sum(train_lengths) / len(train_lengths), 1),
        "p90_train_chars": _percentile(train_lengths, 0.90),
        "p95_train_chars": _percentile(train_lengths, 0.95),
        "max_train_chars": train_lengths[-1],
    }


def write_jsonl(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def save_dataset(samples, valid_indices, test_indices, output_dir):
    splits = {"train": [], "validation": [], "test": []}
    for idx, sample in enumerate(samples):
        if idx in valid_indices:
            splits["validation"].append(sample)
        elif idx in test_indices:
            splits["test"].append(sample)
        else:
            splits["train"].append(sample)
    base = Path(output_dir) / "RA"
    for split, split_samples in splits.items():
        write_jsonl(base / f"{split}.jsonl", split_samples)
    return {
        "name": "RA",
        "train": len(splits["train"]),
        "validation": len(splits["validation"]),
        "test": len(splits["test"]),
        "all_stats": length_stats(samples),
        "train_stats": length_stats(splits["train"]),
        "validation_stats": length_stats(splits["validation"]),
        "test_stats": length_stats(splits["test"]),
    }


def build(config):
    data_cfg = config.get("data", {})
    raw = load_s1k(data_cfg)
    priority = data_cfg.get("reasoning_source_priority", ["deepseek", "gemini"])

    samples = []
    source_counts = {}
    answer_field_counts = {}
    skipped_reasons = {"missing_question": 0, "missing_reasoning": 0, "missing_answer": 0}

    for raw_idx, row in enumerate(raw):
        question = clean(row.get("question"))
        source, reasoning = choose_reasoning(row, priority)
        answer, answer_field = choose_answer(row, source, data_cfg)
        if not question:
            skipped_reasons["missing_question"] += 1
            continue
        if not reasoning:
            skipped_reasons["missing_reasoning"] += 1
            continue
        if not answer:
            skipped_reasons["missing_answer"] += 1
            continue
        source_counts[source] = source_counts.get(source, 0) + 1
        answer_field_counts[answer_field] = answer_field_counts.get(answer_field, 0) + 1
        samples.append(
            make_ra_sample(
                f"s1k_{raw_idx}",
                question,
                answer,
                reasoning,
                meta={"source_index": raw_idx, "reasoning_source": source, "answer_field": answer_field},
            )
        )

    valid_indices, test_indices = split_indices(
        len(samples),
        float(data_cfg.get("valid_ratio", 0.1)),
        float(data_cfg.get("test_ratio", 0.1)),
        int(data_cfg.get("seed", 42)),
    )
    output_dir = data_cfg.get("output_dir", str(SCRIPT_DIR / "data"))
    manifest = {
        "format": "Same text as QAR, but reasoning is train=False and only answer is train=True.",
        "source_dataset": data_cfg.get("source_dataset", "simplescaling/s1K-1.1"),
        "raw_rows": len(raw),
        "usable_rows_before_token_filter": len(samples),
        "skipped_reasons_before_token_filter": skipped_reasons,
        "reasoning_source_counts": source_counts,
        "answer_field_counts": answer_field_counts,
        "answer_source": data_cfg.get("answer_source", "solution"),
        "datasets": [save_dataset(samples, valid_indices, test_indices, output_dir)],
    }
    manifest_path = Path(output_dir) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Prepare reasoning-conditioned answer SFT data.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    build(load_config(args.config))


if __name__ == "__main__":
    main()
