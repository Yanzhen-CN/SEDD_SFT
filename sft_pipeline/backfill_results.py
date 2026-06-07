import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_MODEL_ROOT = SCRIPT_DIR / "modelparameter"

LOSS_RE = re.compile(r"step:\s*(\d+),\s*(training|evaluation)_loss:\s*([0-9.eE+-]+)")
PRETRAIN_RE = re.compile(r"pretrain_evaluation_loss:\s*([0-9.eE+-]+)(?:\s+over\s+(\d+)\s+batch)?")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_csv(rows, path):
    fieldnames = ["step", "kind", "loss", "valid_for_best"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_log_metrics(log_path, min_valid_loss):
    rows = []
    for line in Path(log_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        pretrain = PRETRAIN_RE.search(line)
        if pretrain:
            rows.append({
                "step": 0,
                "kind": "pretrain_evaluation",
                "loss": float(pretrain.group(1)),
                "valid_for_best": True,
            })
            continue

        match = LOSS_RE.search(line)
        if not match:
            continue
        loss = float(match.group(3))
        kind = match.group(2)
        rows.append({
            "step": int(match.group(1)),
            "kind": kind,
            "loss": loss,
            "valid_for_best": kind != "evaluation" or loss > min_valid_loss,
        })
    return rows


def find_log_file(source_run_dir):
    run_dir = Path(source_run_dir)
    for name in ["logs", "train.log"]:
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return None


def copy_if_exists(src, dst, force):
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        return False
    if dst.exists() and not force:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return True


def backfill_one(model_root, name, force, min_valid_loss):
    group_dir = Path(model_root) / name
    best_eval_path = group_dir / "best_eval.json"
    if not best_eval_path.exists():
        print(f"[{name}] skip: {best_eval_path} not found")
        return False

    best_eval = load_json(best_eval_path)
    run_instance = best_eval.get("run_instance")
    copied_any = False

    if run_instance:
        run_dir = group_dir / run_instance
        copied_any |= copy_if_exists(run_dir / "metrics.csv", group_dir / "best_metrics.csv", force)
        copied_any |= copy_if_exists(run_dir / "metrics.jsonl", group_dir / "best_metrics.jsonl", force)
        copied_any |= copy_if_exists(run_dir / "run_info.json", group_dir / "best_run_info.json", force)
        copied_any |= copy_if_exists(run_dir / "pretrain_eval.json", group_dir / "best_pretrain_eval.json", force)

    metrics_csv = group_dir / "best_metrics.csv"
    metrics_jsonl = group_dir / "best_metrics.jsonl"
    if metrics_csv.exists() and metrics_jsonl.exists() and not force:
        print(f"[{name}] best metrics already available")
        return True

    source_run_dir = best_eval.get("source_run_dir")
    log_path = find_log_file(source_run_dir) if source_run_dir else None
    if log_path is None:
        print(f"[{name}] warning: could not find run metrics or Hydra log for backfill")
        return copied_any

    rows = parse_log_metrics(log_path, min_valid_loss)
    if not rows:
        print(f"[{name}] warning: no loss rows parsed from {log_path}")
        return copied_any

    group_dir.mkdir(parents=True, exist_ok=True)
    dump_csv(rows, metrics_csv)
    dump_jsonl(rows, metrics_jsonl)
    print(f"[{name}] reconstructed best_metrics from {log_path}")
    return True


def run_test_eval(args):
    command = [
        sys.executable,
        str(SCRIPT_DIR / "test_eval.py"),
        "--output-root",
        str(args.model_root),
        "--qa-data",
        str(args.qa_data),
        "--qar-data",
        str(args.qar_data),
        "--pretrained",
        args.pretrained,
        "--eval-batches",
        str(args.eval_batches),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
    ]
    print("Running:", " ".join(command))
    subprocess.check_call(command, cwd=str(REPO_DIR))


def main():
    parser = argparse.ArgumentParser(
        description="Backfill best_metrics and test_result files for existing SFT runs."
    )
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--qa-data", default=str(SCRIPT_DIR / "data" / "QA"))
    parser.add_argument("--qar-data", default=str(SCRIPT_DIR / "data" / "QAR"))
    parser.add_argument("--pretrained", default="louaaron/sedd-medium")
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-valid-loss", type=float, default=1.0e-8)
    parser.add_argument("--force", action="store_true", help="Overwrite existing backfilled files.")
    parser.add_argument("--skip-test", action="store_true", help="Only backfill best_metrics files.")
    args = parser.parse_args()

    model_root = Path(args.model_root)
    for name in ["QA", "QAR"]:
        backfill_one(model_root, name, args.force, args.min_valid_loss)

    if not args.skip_test:
        run_test_eval(args)


if __name__ == "__main__":
    main()
