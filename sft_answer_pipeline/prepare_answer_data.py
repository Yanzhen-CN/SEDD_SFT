import argparse
import json
import random
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "answer_config.yaml"
SEGMENT_ORDER = ["user_label", "user", "assistant_label", "assistant", "reasoning_label", "reasoning"]


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
    reasoning_fields = {
        "deepseek": clean(row.get("deepseek_thinking_trajectory")),
        "gemini": clean(row.get("gemini_thinking_trajectory")),
    }
    for source in priority:
        if reasoning_fields.get(source):
            return source, reasoning_fields[source]
    return None, None


def split_indices(n_rows, valid_ratio, test_ratio, seed):
    indices = list(range(n_rows))
    random.Random(seed).shuffle(indices)
    n_valid = max(1, int(n_rows * valid_ratio)) if n_rows > 1 else 0
    n_test = max(1, int(n_rows * test_ratio)) if n_rows > 1 and test_ratio > 0 else 0
    if n_valid + n_test >= n_rows:
        raise ValueError("valid_ratio + test_ratio leaves no training data.")
    return set(indices[:n_valid]), set(indices[n_valid:n_valid + n_test])


def segment(text, train):
    return {"text": clean(text) if train else str(text), "train": bool(train)}


def make_sample(row_id, mode, question, answer, reasoning=None, meta=None):
    segments = {
        "user_label": segment("User: ", False),
        "user": segment(question, False),
        "assistant_label": segment("\nAssistant:\n", False),
        "assistant": segment(answer, True),
    }
    if reasoning is not None:
        segments["reasoning_label"] = segment("\nReasoning:\n", False)
        segments["reasoning"] = segment(reasoning, True)

    return {
        "id": row_id,
        "mode": mode,
        "segments": segments,
        **(meta or {}),
    }


def ordered_segments(sample):
    segments = sample["segments"]
    return [(name, segments[name]) for name in SEGMENT_ORDER if name in segments]


def sample_text(sample, train=None):
    parts = []
    for _, seg in ordered_segments(sample):
        if train is None or bool(seg["train"]) is train:
            parts.append(seg["text"])
    return "".join(parts)


def length_stats(samples):
    lengths = sorted(len(sample_text(sample)) for sample in samples)
    train_lengths = sorted(len(sample_text(sample, train=True)) for sample in samples)
    if not lengths:
        return {"count": 0}
    return {
        "count": len(samples),
        "avg_chars": round(sum(lengths) / len(lengths), 1),
        "p50_chars": lengths[len(lengths) // 2],
        "p90_chars": lengths[int(len(lengths) * 0.9)],
        "max_chars": lengths[-1],
        "avg_train_chars": round(sum(train_lengths) / len(train_lengths), 1),
        "p90_train_chars": train_lengths[int(len(train_lengths) * 0.9)],
        "max_train_chars": train_lengths[-1],
    }


def write_jsonl(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def save_dataset(name, samples, valid_indices, test_indices, output_dir):
    splits = {"train": [], "validation": [], "test": []}
    for idx, sample in enumerate(samples):
        if idx in valid_indices:
            splits["validation"].append(sample)
        elif idx in test_indices:
            splits["test"].append(sample)
        else:
            splits["train"].append(sample)

    base = Path(output_dir) / name
    for split, split_samples in splits.items():
        write_jsonl(base / f"{split}.jsonl", split_samples)

    return {
        "name": name,
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
    qa_samples = []
    qar_samples = []
    source_counts = {"deepseek": 0, "gemini": 0}
    skipped = 0

    for raw_idx, row in enumerate(raw):
        question = clean(row.get("question"))
        answer = clean(row.get("solution"))
        source, reasoning = choose_reasoning(row, priority)
        if not question or not answer or not reasoning:
            skipped += 1
            continue

        source_counts[source] += 1
        row_id = f"s1k_{raw_idx}"
        meta = {"source_index": raw_idx, "reasoning_source": source}
        qa_samples.append(make_sample(row_id, "QA", question, answer, meta=meta))
        qar_samples.append(make_sample(row_id, "QAR", question, answer, reasoning=reasoning, meta=meta))

    valid_indices, test_indices = split_indices(
        len(qa_samples),
        float(data_cfg.get("valid_ratio", 0.1)),
        float(data_cfg.get("test_ratio", 0.1)),
        int(data_cfg.get("seed", 42)),
    )
    output_dir = data_cfg.get("output_dir", str(SCRIPT_DIR / "data"))
    manifest = {
        "source_dataset": data_cfg.get("source_dataset", "simplescaling/s1K-1.1"),
        "raw_rows": len(raw),
        "usable_rows": len(qa_samples),
        "skipped_rows": skipped,
        "segment_order": SEGMENT_ORDER,
        "split_note": "QA and QAR use the same order and split. Each JSONL line is one sample object.",
        "train_note": "segments.*.train=true tokens are noised and used for score entropy loss; train=false tokens are fixed conditions.",
        "reasoning_source_counts": source_counts,
        "datasets": [
            save_dataset("QA", qa_samples, valid_indices, test_indices, output_dir),
            save_dataset("QAR", qar_samples, valid_indices, test_indices, output_dir),
        ],
    }

    manifest_path = Path(output_dir) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Prepare answer SFT JSON-object datasets.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    build(load_config(args.config))


if __name__ == "__main__":
    main()
