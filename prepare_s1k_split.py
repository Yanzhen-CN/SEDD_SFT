import argparse
import json
import random
from pathlib import Path

import yaml


DEFAULT_OUTPUT_DIR = Path("data") / "s1k_split"


def clean_row(row, source_index):
    item = dict(row)
    item["source_index"] = int(source_index)
    item["id"] = f"s1k_{source_index}"
    return item


def load_s1k(config):
    if config.get("arrow_path"):
        from datasets import Dataset

        return Dataset.from_file(config["arrow_path"])

    from datasets import load_dataset

    return load_dataset(config.get("source_dataset", "simplescaling/s1K-1.1"), split="train")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create one fixed S1K split shared by all SFT/RL pipelines.")
    parser.add_argument("--config", default=None, help="Optional YAML config with data.source_dataset/arrow_path/ratios/seed.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--source-dataset", default="simplescaling/s1K-1.1")
    parser.add_argument("--arrow-path", default=None)
    parser.add_argument("--valid-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            root = yaml.safe_load(f) or {}
        cfg = root.get("data", root)

    data_cfg = {
        "source_dataset": cfg.get("source_dataset", args.source_dataset),
        "arrow_path": cfg.get("arrow_path", args.arrow_path),
    }
    valid_ratio = float(cfg.get("valid_ratio", args.valid_ratio))
    test_ratio = float(cfg.get("test_ratio", args.test_ratio))
    seed = int(cfg.get("seed", args.seed))

    raw = load_s1k(data_cfg)
    rows = [clean_row(row, idx) for idx, row in enumerate(raw)]
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)

    n_valid = max(1, int(len(rows) * valid_ratio)) if valid_ratio > 0 else 0
    n_test = max(1, int(len(rows) * test_ratio)) if test_ratio > 0 else 0
    if n_valid + n_test >= len(rows):
        raise ValueError("valid_ratio + test_ratio leaves no training rows.")

    split_indices = {
        "validation": set(indices[:n_valid]),
        "test": set(indices[n_valid : n_valid + n_test]),
    }
    splits = {"train": [], "validation": [], "test": []}
    for idx, row in enumerate(rows):
        if idx in split_indices["validation"]:
            splits["validation"].append(row)
        elif idx in split_indices["test"]:
            splits["test"].append(row)
        else:
            splits["train"].append(row)

    out_dir = Path(args.output_dir)
    for split, split_rows in splits.items():
        write_jsonl(out_dir / f"{split}.jsonl", split_rows)

    manifest = {
        "source_dataset": data_cfg["source_dataset"],
        "arrow_path": data_cfg["arrow_path"],
        "seed": seed,
        "valid_ratio": valid_ratio,
        "test_ratio": test_ratio,
        "raw_rows": len(rows),
        "splits": {name: len(split_rows) for name, split_rows in splits.items()},
        "note": "This raw S1K split is shared by answer, reasoning, and RL pipelines. Pipeline-specific token filters run after this split.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
