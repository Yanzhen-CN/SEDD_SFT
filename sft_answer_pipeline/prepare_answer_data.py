import argparse
import json
import random
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "answer_config.yaml"

# Textual layout follows S1K-style completion and SEDD plain-text LM blocks:
#   User: <question>\nAssistant:\nReasoning:\n<reasoning>\n\nAnswer:\n<answer>
# For target-only SFT, structural markers are fixed anchors (train=False);
# only the content after each marker is generated / noised / lossed.
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


def choose_answer(row, source, data_cfg):
    """Choose answer text.

    answer_source='solution' uses the dataset solution field, which is the
    answer target intended for evaluation.

    answer_source='matched_attempt' uses deepseek_attempt/gemini_attempt when
    available. This keeps reasoning and answer from the same teacher model, but
    the attempt may contain extra explanatory text and may not be as clean as
    the dataset solution.
    """
    answer_source = data_cfg.get("answer_source", "solution")
    fallback_field = data_cfg.get("answer_field", "solution")
    fallback = clean(row.get(fallback_field))
    if answer_source == "matched_attempt" and source:
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


def make_qa_sample(row_id, question, answer, meta=None):
    # QA is a short baseline.  The assistant completion is still labeled as an
    # assistant answer rather than stored in a misleading "assistant=answer" field.
    segments = {
        "user_label": segment("User: ", False),
        "user": segment(clean(question), False),
        "assistant_label": segment("\nAssistant:\n", False),
        "answer_label": segment("Answer:\n", False),
        "answer": segment(clean(answer), True),
    }
    return {
        "id": row_id,
        "mode": "QA",
        "segment_order": QA_SEGMENT_ORDER,
        "segments": segments,
        "answer": clean(answer),
        "reasoning": "",
        **(meta or {}),
    }


def make_qar_sample(row_id, question, answer, reasoning, meta=None):
    # Correct QAR layout: the assistant completion contains fixed section markers
    # and generated section contents.  Reasoning:/Answer: are anchors
    # (train=False), while reasoning text and answer text are train=True.
    segments = {
        "user_label": segment("User: ", False),
        "user": segment(clean(question), False),
        "assistant_label": segment("\nAssistant:\n", False),
        "reasoning_label": segment("Reasoning:\n", False),
        "reasoning": segment(clean(reasoning), True),
        "answer_label": segment("\n\nAnswer:\n", False),
        "answer": segment(clean(answer), True),
    }
    return {
        "id": row_id,
        "mode": "QAR",
        "segment_order": QAR_SEGMENT_ORDER,
        "segments": segments,
        "answer": clean(answer),
        "reasoning": clean(reasoning),
        **(meta or {}),
    }


def sample_text(sample, train=None):
    parts = []
    for name in sample.get("segment_order", []):
        segment_obj = sample["segments"].get(name)
        if segment_obj is None:
            continue
        if train is None or bool(segment_obj["train"]) is train:
            parts.append(segment_obj["text"])
    return "".join(parts)


def _percentile(sorted_values, q):
    if not sorted_values:
        return 0
    idx = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * q))))
    return sorted_values[idx]


def length_stats(samples):
    lengths = sorted(len(sample_text(sample)) for sample in samples)
    train_lengths = sorted(len(sample_text(sample, train=True)) for sample in samples)
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
    source_counts = {}
    answer_field_counts = {}
    skipped = 0
    skipped_reasons = {"missing_question": 0, "missing_reasoning": 0, "missing_answer": 0}

    for raw_idx, row in enumerate(raw):
        question = clean(row.get("question"))
        source, reasoning = choose_reasoning(row, priority)
        answer, answer_field = choose_answer(row, source, data_cfg)

        if not question:
            skipped += 1
            skipped_reasons["missing_question"] += 1
            continue
        if not reasoning:
            skipped += 1
            skipped_reasons["missing_reasoning"] += 1
            continue
        if not answer:
            skipped += 1
            skipped_reasons["missing_answer"] += 1
            continue

        source_counts[source] = source_counts.get(source, 0) + 1
        answer_field_counts[answer_field] = answer_field_counts.get(answer_field, 0) + 1
        row_id = f"s1k_{raw_idx}"
        meta = {
            "source_index": raw_idx,
            "reasoning_source": source,
            "answer_field": answer_field,
        }
        qa_samples.append(make_qa_sample(row_id, question, answer, meta=meta))
        qar_samples.append(make_qar_sample(row_id, question, answer, reasoning, meta=meta))

    valid_indices, test_indices = split_indices(
        len(qa_samples),
        float(data_cfg.get("valid_ratio", 0.1)),
        float(data_cfg.get("test_ratio", 0.1)),
        int(data_cfg.get("seed", 42)),
    )
    output_dir = data_cfg.get("output_dir", str(SCRIPT_DIR / "data"))
    manifest = {
        "format": "User + Assistant completion. QAR uses fixed anchors 'Reasoning:' and 'Answer:'; reasoning/answer contents are train=True.",
        "source_dataset": data_cfg.get("source_dataset", "simplescaling/s1K-1.1"),
        "raw_rows": len(raw),
        "usable_rows_before_token_filter": len(qa_samples),
        "skipped_rows_before_token_filter": skipped,
        "skipped_reasons_before_token_filter": skipped_reasons,
        "reasoning_source_counts": source_counts,
        "answer_source": data_cfg.get("answer_source", "solution"),
        "answer_field_counts": answer_field_counts,
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
    parser = argparse.ArgumentParser(description="Prepare QA/QAR target-only SEDD SFT data.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    build(load_config(args.config))


if __name__ == "__main__":
    main()
