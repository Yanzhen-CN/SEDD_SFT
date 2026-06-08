import argparse
import json
import random
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "answer_config.yaml"


def clean(text):
    return str(text or "").strip()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_s1k(data_cfg):
    arrow_path = data_cfg.get("arrow_path")
    if arrow_path:
        from datasets import Dataset

        return Dataset.from_file(arrow_path)

    from datasets import load_dataset

    return load_dataset(data_cfg.get("source_dataset", "simplescaling/s1K-1.1"), split="train")


def choose_reasoning(row, priority):
    mapping = {
        "deepseek": clean(row.get("deepseek_thinking_trajectory")),
        "gemini": clean(row.get("gemini_thinking_trajectory")),
    }
    for source in priority:
        if mapping.get(source):
            return source, mapping[source]
    return None, None


def split_indices(n_rows, valid_ratio, test_ratio, seed):
    indices = list(range(n_rows))
    random.Random(seed).shuffle(indices)
    n_valid = max(1, int(n_rows * valid_ratio)) if n_rows > 1 else 0
    n_test = max(1, int(n_rows * test_ratio)) if n_rows > 1 and test_ratio > 0 else 0
    if n_valid + n_test >= n_rows:
        raise ValueError("valid_ratio + test_ratio leaves no training data.")
    return set(indices[:n_valid]), set(indices[n_valid:n_valid + n_test])


def make_qa(row_id, question, answer, meta):
    prompt = f"User: {question}\nAssistant:\nAnswer:\n"
    target = clean(answer)
    return {
        "id": row_id,
        "mode": "QA",
        "prompt": prompt,
        "target": target,
        "text": prompt + target,
        "segments": [
            {"text": prompt, "loss": False, "name": "prompt"},
            {"text": target, "loss": True, "name": "answer"},
        ],
        "question": question,
        "answer": answer,
        **meta,
    }


def make_qar(row_id, question, reasoning, answer, meta):
    prompt = f"User: {question}\nAssistant:\nAnswer:\n"
    answer_text = clean(answer)
    reason_cue = "\n\nReason:\n"
    reasoning_text = clean(reasoning)
    target = answer_text + reason_cue + reasoning_text
    return {
        "id": row_id,
        "mode": "QAR",
        "prompt": prompt,
        "target": target,
        "text": prompt + target,
        "segments": [
            {"text": prompt, "loss": False, "name": "prompt"},
            {"text": answer_text, "loss": True, "name": "answer"},
            {"text": reason_cue, "loss": False, "name": "reason_cue"},
            {"text": reasoning_text, "loss": True, "name": "reasoning"},
        ],
        "question": question,
        "reasoning": reasoning,
        "answer": answer,
        **meta,
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def char_stats(rows):
    lengths = sorted(len(row["text"]) for row in rows)
    target_lengths = sorted(len(row["target"]) for row in rows)
    if not lengths:
        return {"count": 0}
    return {
        "count": len(rows),
        "avg_chars": round(sum(lengths) / len(lengths), 1),
        "p50_chars": lengths[len(lengths) // 2],
        "p90_chars": lengths[int(len(lengths) * 0.9)],
        "max_chars": lengths[-1],
        "avg_target_chars": round(sum(target_lengths) / len(target_lengths), 1),
        "p90_target_chars": target_lengths[int(len(target_lengths) * 0.9)],
        "max_target_chars": target_lengths[-1],
    }


def save_dataset(rows, valid_indices, test_indices, output_dir, name):
    splits = {"train": [], "validation": [], "test": []}
    for idx, row in enumerate(rows):
        if idx in valid_indices:
            splits["validation"].append(row)
        elif idx in test_indices:
            splits["test"].append(row)
        else:
            splits["train"].append(row)

    base = Path(output_dir) / name
    for split, split_rows in splits.items():
        write_jsonl(base / f"{split}.jsonl", split_rows)

    return {
        "name": name,
        "train": len(splits["train"]),
        "validation": len(splits["validation"]),
        "test": len(splits["test"]),
        "all_stats": char_stats(rows),
        "train_stats": char_stats(splits["train"]),
        "validation_stats": char_stats(splits["validation"]),
        "test_stats": char_stats(splits["test"]),
    }


def build(config):
    data_cfg = config.get("data", {})
    raw = load_s1k(data_cfg)
    priority = data_cfg.get("reasoning_source_priority", ["deepseek", "gemini"])
    qa_rows = []
    qar_rows = []
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
        qa_rows.append(make_qa(row_id, question, answer, meta))
        qar_rows.append(make_qar(row_id, question, reasoning, answer, meta))

    valid_indices, test_indices = split_indices(
        len(qa_rows),
        float(data_cfg.get("valid_ratio", 0.1)),
        float(data_cfg.get("test_ratio", 0.1)),
        int(data_cfg.get("seed", 42)),
    )
    output_dir = data_cfg.get("output_dir", str(SCRIPT_DIR / "data"))
    manifest = {
        "source_dataset": data_cfg.get("source_dataset", "simplescaling/s1K-1.1"),
        "raw_rows": len(raw),
        "usable_rows": len(qa_rows),
        "skipped_rows": skipped,
        "split_note": "QA and QAR use the same sample ids and split; samples are never concatenated into blocks.",
        "loss_note": "Training uses prompt as condition and computes score entropy only on target tokens.",
        "reasoning_source_counts": source_counts,
        "datasets": [
            save_dataset(qa_rows, valid_indices, test_indices, output_dir, "QA"),
            save_dataset(qar_rows, valid_indices, test_indices, output_dir, "QAR"),
        ],
    }
    manifest_path = Path(output_dir) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Prepare answer-conditioned QA/QAR datasets.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    build(load_config(args.config))


if __name__ == "__main__":
    main()
