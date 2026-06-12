from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

DEFAULT_RUN_ROOT = SCRIPT_DIR / "modelparameter" / "rl_QRA"
DEFAULT_OUT_DIR = SCRIPT_DIR / "visual" / "rl_QRA"

LOSS_SERIES = ["loss", "eval_loss", "full_eval_loss"]
REWARD_SERIES = ["rollout_reward", "eval_rollout_reward", "full_eval_rollout_reward", "reward_best_metric_value"]
SUMMARY_KEYS = LOSS_SERIES + REWARD_SERIES + [
    "rollout_anchor_loss", "eval_rollout_anchor_loss", "full_eval_rollout_anchor_loss",
    "best_metric_value", "is_best_eval", "is_best_reward_eval",
]


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


def collect_metric_files(run_root: Path) -> List[Tuple[str, Path]]:
    files: List[Tuple[str, Path]] = []
    # Root global-best curve copied from the selected run.
    best = run_root / "best_metrics.csv"
    if best.exists():
        files.append(("global_best", best))
    if run_root.exists():
        for p in sorted(run_root.glob("*/metrics.csv")):
            files.append((p.parent.name, p))
        legacy = run_root / "runs"
        if legacy.exists():
            for p in sorted(legacy.glob("*/metrics.csv")):
                files.append((p.parent.name, p))
    seen = set()
    out: List[Tuple[str, Path]] = []
    for name, path in files:
        rp = path.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append((name, path))
    return out


def plot_series(rows: List[Dict[str, str]], series_names: List[str], out_path: Path, title: str, ylabel: str) -> bool:
    plotted = False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for metric in series_names:
        xs: List[float] = []
        ys: List[float] = []
        for i, row in enumerate(rows):
            y = to_float(row.get(metric))
            if y is None:
                continue
            x = to_float(row.get("step"))
            xs.append(float(i if x is None else x))
            ys.append(y)
        if ys:
            plt.plot(xs, ys, linewidth=1.2, label=metric)
            plotted = True
    if not plotted:
        plt.close()
        return False
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def summarize_metric_file(run_name: str, path: Path) -> Dict[str, object]:
    rows = read_csv(path)
    summary: Dict[str, object] = {"run": run_name, "metrics_path": str(path), "rows": len(rows), "run_dir": str(path.parent)}
    if not rows:
        return summary
    tail = rows[-min(100, len(rows)):]
    for key in SUMMARY_KEYS:
        vals = [to_float(r.get(key)) for r in tail]
        vals = [v for v in vals if v is not None]
        if vals:
            summary[f"last100_{key}"] = sum(vals) / len(vals)
            summary[f"final_{key}"] = vals[-1]
    # Add compact final json info if present.
    json_candidates = []
    if path.name == "best_metrics.csv":
        json_candidates += [path.parent / "best_metrics.json", path.parent / "best_eval.json"]
    json_candidates += [path.parent / "metrics.json", path.parent / "eval.json"]
    for jp in json_candidates:
        if not jp.exists():
            continue
        try:
            obj = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        for key in ["eval_loss", "full_eval_loss", "eval_count", "eval_split", "best_metric_name", "best_metric_value"]:
            if key in obj:
                summary[key] = obj[key]
        rb = obj.get("reward_best")
        if isinstance(rb, dict):
            summary["reward_best_metric_name"] = rb.get("best_metric_name")
            summary["reward_best_metric_value"] = rb.get("best_metric_value")
            summary["reward_best_eval_loss"] = rb.get("eval_loss")
            summary["reward_best_rollout_reward"] = rb.get("rollout_reward")
        break
    return summary


def write_summary(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fields:
                fields.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def visualize(run_root: Path, out_dir: Path) -> List[Dict[str, object]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, object]] = []
    for run_name, metrics_path in collect_metric_files(run_root):
        rows = read_csv(metrics_path)
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in run_name)[:160]
        summaries.append(summarize_metric_file(run_name, metrics_path))
        plot_series(rows, LOSS_SERIES, out_dir / f"{safe}_loss.png", f"{run_name}: loss curves", "loss")
        plot_series(rows, REWARD_SERIES, out_dir / f"{safe}_reward.png", f"{run_name}: reward curves", "reward")
    def sort_key(row: Dict[str, object]):
        for k in ("full_eval_loss", "last100_full_eval_loss", "eval_loss", "last100_eval_loss", "final_loss"):
            v = to_float(row.get(k))
            if v is not None:
                return v
        return float("inf")
    summaries.sort(key=sort_key)
    write_summary(summaries, out_dir / "training_summary.csv")
    (out_dir / "training_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "visual_summary.json").write_text(json.dumps({
        "run_root": str(run_root),
        "out_dir": str(out_dir),
        "num_metric_runs": len(summaries),
        "plots": "one *_loss.png and one *_reward.png per run/global_best",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact RL visualizer: only loss and reward curves.")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT), help="Usually rl_qra_pipeline/modelparameter/rl_QRA")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    summaries = visualize(Path(args.run_root), Path(args.out_dir))
    print(f"Wrote compact visuals to: {args.out_dir}")
    print(f"Runs visualized: {len(summaries)}")


if __name__ == "__main__":
    main()
