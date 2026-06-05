import argparse
import json
import random
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
DATA_ROOT = PIPELINE_DIR / "data"


def clean(text):
    return str(text or "").strip()


def format_qa(question, answer):
    return f"User: {clean(question)}\nAssistant: {clean(answer)}"


def format_qar(question, reasoning, answer):
    return (
        f"User: {clean(question)}\n"
        f"Assistant:\n{clean(reasoning)}\n\n"
        f"Final Answer: {clean(answer)}"
    )


def choose_reasoning(row):
    deepseek = clean(row.get("deepseek_thinking_trajectory"))
    gemini = clean(row.get("gemini_thinking_trajectory"))
    if deepseek:
        return "deepseek", deepseek
    if gemini:
        return "gemini", gemini
    return None, None


def load_s1k(args):
    if args.arrow:
        from datasets import Dataset

        return Dataset.from_file(args.arrow)

    from datasets import load_dataset

    return load_dataset("simplescaling/s1K-1.1", split="train")


def split_indices(n_rows, valid_ratio, seed):
    indices = list(range(n_rows))
    random.Random(seed).shuffle(indices)
    n_valid = max(1, int(n_rows * valid_ratio)) if n_rows > 1 else 0
    valid = set(indices[:n_valid])
    return valid


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def length_stats(rows):
    lengths = sorted(len(row["text"]) for row in rows)
    if not lengths:
        return {
            "count": 0,
            "avg_chars": 0,
            "min_chars": 0,
            "p50_chars": 0,
            "p90_chars": 0,
            "max_chars": 0,
        }
    return {
        "count": len(lengths),
        "avg_chars": round(sum(lengths) / len(lengths), 1),
        "min_chars": lengths[0],
        "p50_chars": lengths[len(lengths) // 2],
        "p90_chars": lengths[int(len(lengths) * 0.9)],
        "max_chars": lengths[-1],
    }


def save_split(name, rows, valid_indices):
    train_rows = []
    valid_rows = []
    for idx, row in enumerate(rows):
        if idx in valid_indices:
            valid_rows.append({"text": row["text"]})
        else:
            train_rows.append({"text": row["text"]})

    output_dir = DATA_ROOT / name
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "validation.jsonl", valid_rows)

    return {
        "name": name,
        "train": len(train_rows),
        "validation": len(valid_rows),
        "total": len(rows),
        "all": length_stats(rows),
        "train_stats": length_stats(train_rows),
        "validation_stats": length_stats(valid_rows),
    }


def build_datasets(args):
    raw = load_s1k(args)
    qa_rows = []
    qar_rows = []
    source_counts = {"deepseek": 0, "gemini": 0}
    skipped = 0

    for raw_idx, row in enumerate(raw):
        question = clean(row.get("question"))
        answer = clean(row.get("solution"))
        source, reasoning = choose_reasoning(row)

        if not question or not answer or not reasoning:
            skipped += 1
            continue

        source_counts[source] += 1
        meta = {
            "source_index": raw_idx,
            "reasoning_source": source,
        }
        qa_rows.append({
            "text": format_qa(question, answer),
            **meta,
        })
        qar_rows.append({
            "text": format_qar(question, reasoning, answer),
            **meta,
        })

    valid_indices = split_indices(len(qa_rows), args.valid_ratio, args.seed)
    stats = {
        "source_dataset": "simplescaling/s1K-1.1",
        "raw_rows": len(raw),
        "usable_rows": len(qa_rows),
        "skipped_rows": skipped,
        "valid_ratio": args.valid_ratio,
        "seed": args.seed,
        "reasoning_source_counts": source_counts,
        "split_note": "QA and QAR use exactly the same sample order and train/validation split.",
        "datasets": [
            save_split("QA", qa_rows, valid_indices),
            save_split("QAR", qar_rows, valid_indices),
        ],
        "notes": [
            "QA uses question + gold solution.",
            "QAR uses question + reasoning trajectory + gold solution.",
            "Reasoning source priority is DeepSeek first, Gemini second.",
            "Accuracy is not the primary evaluation target; the split supports loss/stability/style comparisons under the same questions.",
            "QAR is much longer and may need larger model.length or truncation analysis.",
        ],
    }

    write_jsonl(PIPELINE_DIR / "manifest.jsonl", [stats])
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Convert s1K-1.1 into matched QA/QAR local JSONL datasets.")
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arrow", default=None, help="Optional local Arrow cache file for offline conversion.")
    args = parser.parse_args()
    build_datasets(args)


if __name__ == "__main__":
    main()
