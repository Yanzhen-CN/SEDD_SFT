import argparse
import json
import random
from pathlib import Path

import yaml
from transformers import GPT2TokenizerFast


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


def load_s1k_splits(data_cfg):
    split_dir = data_cfg.get("split_dir")
    if not split_dir:
        return None
    base = Path(split_dir)
    splits = {}
    for split in ("train", "validation", "test"):
        path = base / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing shared S1K split file: {path}")
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        splits[split] = rows
    return splits


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


def token_lengths(sample, tokenizer):
    total = 0
    train = 0
    segment_tokens = {}
    for name in sample.get("segment_order", []):
        segment_obj = sample["segments"].get(name)
        if segment_obj is None:
            continue
        n_tokens = len(tokenizer(segment_obj.get("text", ""), add_special_tokens=False).input_ids)
        segment_tokens[name] = n_tokens
        total += n_tokens
        if bool(segment_obj.get("train", False)):
            train += n_tokens
    return total, train, total - train, segment_tokens


def _summary(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": values[0],
        "avg": round(sum(values) / len(values), 2),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "max": values[-1],
    }


def filter_by_tokens(name, split_name, samples, tokenizer, max_length, min_target_tokens, output_dir):
    kept = []
    skipped = []
    all_total, all_train, kept_total, kept_train = [], [], [], []
    reasons = {"overlength": 0, "short_target": 0, "no_train_tokens": 0}
    for sample in samples:
        total, train, condition, segment_tokens = token_lengths(sample, tokenizer)
        all_total.append(total)
        all_train.append(train)
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
                skipped.append(
                    {
                        "id": sample.get("id"),
                        "source_index": sample.get("source_index"),
                        "reason": reason,
                        "total_tokens": total,
                        "train_tokens": train,
                        "condition_tokens": condition,
                        "segment_tokens": segment_tokens,
                    }
                )
            continue

        kept.append(sample)
        kept_total.append(total)
        kept_train.append(train)

    report = {
        "name": name,
        "split": split_name,
        "max_length": max_length,
        "min_target_tokens": min_target_tokens,
        "input_rows": len(samples),
        "kept_rows": len(kept),
        "skipped_rows": len(samples) - len(kept),
        "skip_reasons": reasons,
        "all_total_tokens": _summary(all_total),
        "all_train_tokens": _summary(all_train),
        "kept_total_tokens": _summary(kept_total),
        "kept_train_tokens": _summary(kept_train),
        "skipped_examples": skipped,
        "note": "Each JSONL row is checked as one independent training sample. Rows with total_tokens > model.max_length are dropped and reported.",
    }
    report_path = Path(output_dir) / name / f"{split_name}_prepare_filter_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return kept, report


def write_jsonl(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def save_dataset(name, split_samples, output_dir, tokenizer, max_length, min_target_tokens):
    splits = {}
    filter_reports = {}
    input_samples = []
    for split, samples in split_samples.items():
        input_samples.extend(samples)
        kept, report = filter_by_tokens(name, split, samples, tokenizer, max_length, min_target_tokens, output_dir)
        splits[split] = kept
        filter_reports[split] = report

    base = Path(output_dir) / name
    for split, split_samples in splits.items():
        write_jsonl(base / f"{split}.jsonl", split_samples)

    return {
        "name": name,
        "train": len(splits["train"]),
        "validation": len(splits["validation"]),
        "test": len(splits["test"]),
        "input_stats_before_token_filter": length_stats(input_samples),
        "all_stats": length_stats([sample for split in splits.values() for sample in split]),
        "train_stats": length_stats(splits["train"]),
        "validation_stats": length_stats(splits["validation"]),
        "test_stats": length_stats(splits["test"]),
        "token_filter_reports": {
            split: str(Path(output_dir) / name / f"{split}_prepare_filter_report.json")
            for split in splits
        },
        "token_filter_summary": {
            split: {
                "input_rows": report["input_rows"],
                "kept_rows": report["kept_rows"],
                "skipped_rows": report["skipped_rows"],
                "skip_reasons": report["skip_reasons"],
                "kept_total_tokens": report["kept_total_tokens"],
            }
            for split, report in filter_reports.items()
        },
    }


def build(config):
    data_cfg = config.get("data", {})
    priority = data_cfg.get("reasoning_source_priority", ["deepseek", "gemini"])
    shared_splits = load_s1k_splits(data_cfg)
    raw = load_s1k(data_cfg) if shared_splits is None else [row for rows in shared_splits.values() for row in rows]
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    max_length = int(config.get("model", {}).get("max_length", 512))
    min_target_tokens = int(config.get("model", {}).get("min_target_tokens", 32))

    qa_by_split = {"train": [], "validation": [], "test": []}
    qar_by_split = {"train": [], "validation": [], "test": []}
    source_counts = {}
    answer_field_counts = {}
    skipped = 0
    skipped_reasons = {"missing_question": 0, "missing_reasoning": 0, "missing_answer": 0}

    if shared_splits is None:
        raw_iter = [("all", idx, row) for idx, row in enumerate(raw)]
    else:
        raw_iter = []
        for split, rows in shared_splits.items():
            raw_iter.extend((split, int(row.get("source_index", idx)), row) for idx, row in enumerate(rows))

    all_qa_samples = []
    for split, raw_idx, row in raw_iter:
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
            "split": split,
        }
        qa_sample = make_qa_sample(row_id, question, answer, meta=meta)
        qar_sample = make_qar_sample(row_id, question, answer, reasoning, meta=meta)
        if shared_splits is None:
            all_qa_samples.append((qa_sample, qar_sample))
        else:
            qa_by_split[split].append(qa_sample)
            qar_by_split[split].append(qar_sample)

    if shared_splits is None:
        valid_indices, test_indices = split_indices(
            len(all_qa_samples),
            float(data_cfg.get("valid_ratio", 0.1)),
            float(data_cfg.get("test_ratio", 0.1)),
            int(data_cfg.get("seed", 42)),
        )
        for idx, (qa_sample, qar_sample) in enumerate(all_qa_samples):
            split = "validation" if idx in valid_indices else "test" if idx in test_indices else "train"
            qa_sample["split"] = split
            qar_sample["split"] = split
            qa_by_split[split].append(qa_sample)
            qar_by_split[split].append(qar_sample)

    output_dir = data_cfg.get("output_dir", str(SCRIPT_DIR / "data"))
    manifest = {
        "format": "User + Assistant completion. QAR uses fixed anchors 'Reasoning:' and 'Answer:'; reasoning/answer contents are train=True.",
        "source_dataset": data_cfg.get("source_dataset", "simplescaling/s1K-1.1"),
        "shared_split_dir": data_cfg.get("split_dir"),
        "raw_rows": len(raw),
        "usable_rows_before_token_filter": sum(len(rows) for rows in qa_by_split.values()),
        "skipped_rows_before_token_filter": skipped,
        "skipped_reasons_before_token_filter": skipped_reasons,
        "reasoning_source_counts": source_counts,
        "answer_source": data_cfg.get("answer_source", "solution"),
        "answer_field_counts": answer_field_counts,
        "max_length": max_length,
        "min_target_tokens": min_target_tokens,
        "split_note": "Rows are assigned to train/validation/test before pipeline-specific token filtering. This prevents leakage across pipelines.",
        "datasets": [
            save_dataset("QA", qa_by_split, output_dir, tokenizer, max_length, min_target_tokens),
            save_dataset("QAR", qar_by_split, output_dir, tokenizer, max_length, min_target_tokens),
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
