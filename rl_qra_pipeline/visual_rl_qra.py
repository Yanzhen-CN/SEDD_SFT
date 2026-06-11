from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

DEFAULT_RUN_ROOT = SCRIPT_DIR / "modelparameter" / "rl_QRA"
DEFAULT_OUT_DIR = SCRIPT_DIR / "visual" / "rl_QRA"


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: object) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def plot_metric(rows: List[Dict[str, str]], metric: str, out_path: Path, title: str) -> bool:
    xs: List[float] = []
    ys: List[float] = []

    for i, row in enumerate(rows):
        y = to_float(row.get(metric))
        if y is None:
            continue
        x = to_float(row.get("step"))
        xs.append(float(i if x is None else x))
        ys.append(y)

    if not ys:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(xs, ys, linewidth=1.2)
    plt.xlabel("step")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def summarize_metrics(path: Path) -> Dict[str, object]:
    rows = read_csv(path)
    if not rows:
        return {"metrics_path": str(path), "rows": 0}

    tail = rows[-min(100, len(rows)) :]
    summary: Dict[str, object] = {
        "run_dir": str(path.parent),
        "metrics_path": str(path),
        "rows": len(rows),
    }

    for key in [
        "loss",
        "target_prob",
        "target_logp",
        "model_reward",
        "best_reward",
        "reward_gap",
        "candidate_entropy",
        "eval_loss",
    ]:
        vals = [to_float(row.get(key)) for row in tail]
        vals = [v for v in vals if v is not None]
        if vals:
            summary[f"last100_{key}"] = sum(vals) / len(vals)
            summary[f"final_{key}"] = vals[-1]

    eval_json = path.parent / "eval.json"
    if eval_json.exists():
        try:
            obj = json.loads(eval_json.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                for key in ["eval_loss", "best_loss", "final_loss"]:
                    if key in obj:
                        summary[key] = obj[key]
        except Exception:
            pass

    return summary


def write_summary(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def collect_metric_files(run_root: Path) -> List[Path]:
    files: List[Path] = []

    best = run_root / "best_metrics.csv"
    if best.exists():
        files.append(best)

    runs_dir = run_root / "runs"
    if runs_dir.exists():
        files.extend(sorted(runs_dir.glob("*/metrics.csv")))

    # Backward compatible with older timestamp folders under root.
    files.extend(sorted(p for p in run_root.glob("*/metrics.csv") if "/runs/" not in str(p)))

    # Deduplicate while preserving order.
    seen = set()
    out = []
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            out.append(p)
            seen.add(rp)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize RL-QRA metrics.")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    metric_files = collect_metric_files(run_root)

    if not metric_files:
        raise SystemExit(f"No metrics.csv found under {run_root}")

    metrics = [
        "loss",
        "target_prob",
        "target_logp",
        "model_reward",
        "best_reward",
        "reward_gap",
        "candidate_entropy",
        "eval_loss",
    ]

    summaries: List[Dict[str, object]] = []
    for path in metric_files:
        rows = read_csv(path)
        run_name = "global_best" if path.name == "best_metrics.csv" else path.parent.name
        summaries.append({"run": run_name, **summarize_metrics(path)})

        for metric in metrics:
            plot_metric(
                rows,
                metric,
                out_dir / run_name / f"{metric}.png",
                f"{run_name}: {metric}",
            )

    # Sort summary by eval_loss if available, otherwise by last100 loss.
    def sort_key(row: Dict[str, object]):
        for key in ("eval_loss", "last100_eval_loss", "last100_loss", "final_loss"):
            val = to_float(row.get(key))
            if val is not None:
                return val
        return float("inf")

    summaries.sort(key=sort_key)
    write_summary(summaries, out_dir / "summary.csv")
    (out_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Found {len(metric_files)} metrics files.")
    print(f"Wrote visuals to: {out_dir}")
    print(f"Wrote summary: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
