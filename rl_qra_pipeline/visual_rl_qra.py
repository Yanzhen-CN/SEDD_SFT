from __future__ import annotations

"""Compact visualizer for RL-QRA.

It produces two kinds of plots:
1. Training curves from modelparameter/rl_<start>/<run_id>/metrics.csv
   - one loss plot and one reward plot per run/global best
2. Test comparison plots from modelparameter/test_result/test_rl_qra.csv
   - one bar chart for test loss and one bar chart for test rollout reward

The goal is to match the answer-pipeline style: a small set of directly useful
figures rather than one figure per metric.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

DEFAULT_RUN_ROOTS = [
    SCRIPT_DIR / "modelparameter" / "rl_pretrain",
    SCRIPT_DIR / "modelparameter" / "rl_QRA",
]
DEFAULT_TEST_RESULT_DIR = SCRIPT_DIR / "modelparameter" / "test_result"
DEFAULT_OUT_DIR = SCRIPT_DIR / "visualization" / "rl_qra"

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


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(text))[:160] or "run"


def collect_metric_files(run_root: Path) -> List[Tuple[str, Path]]:
    files: List[Tuple[str, Path]] = []
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


def group_name_from_root(run_root: Path) -> str:
    return safe_name(run_root.name or "rl_group")


def choose_metric_series(rows: List[Dict[str, str]], preferred: Sequence[str]) -> Tuple[str, List[Tuple[float, float]]]:
    """Return the first available metric series from a preference list."""
    for metric in preferred:
        pts: List[Tuple[float, float]] = []
        for i, row in enumerate(rows):
            y = to_float(row.get(metric))
            if y is None:
                continue
            x = to_float(row.get("step"))
            pts.append((float(i if x is None else x), y))
        if pts:
            return metric, pts
    return "", []


def plot_group_overlay(
    metric_files: List[Tuple[str, Path]],
    preferred: Sequence[str],
    out_path: Path,
    title: str,
    ylabel: str,
) -> bool:
    """One compact curve plot per RL group.

    Each line is one run/global-best file.  For each file we prefer full-eval
    metrics, then mini-eval metrics, then train metrics.  This keeps the figure
    answer-pipeline-like: one loss curve image and one reward curve image per
    RL family, rather than many files per metric.
    """
    plotted = False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for run_name, path in metric_files:
        rows = read_csv(path)
        metric, pts = choose_metric_series(rows, preferred)
        if not pts:
            continue
        pts = sorted(pts, key=lambda x: x[0])
        label = run_name if metric == preferred[0] else f"{run_name} ({metric})"
        plt.plot([x for x, _ in pts], [y for _, y in pts], linewidth=1.2, label=label)
        plotted = True
    if not plotted:
        plt.close()
        return False
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def visualize_training(run_roots: Sequence[Path], out_dir: Path) -> List[Dict[str, object]]:
    """Visualize both RL groups by default: rl_pretrain and rl_QRA."""
    summaries: List[Dict[str, object]] = []
    loss_preferred = ["full_eval_loss", "eval_loss", "loss"]
    reward_preferred = ["full_eval_rollout_reward", "eval_rollout_reward", "rollout_reward", "reward_best_metric_value"]

    for run_root in run_roots:
        group = group_name_from_root(run_root)
        metric_files = collect_metric_files(run_root)
        for run_name, metrics_path in metric_files:
            item = summarize_metric_file(f"{group}/{run_name}", metrics_path)
            item["group"] = group
            summaries.append(item)
        plot_group_overlay(
            metric_files,
            loss_preferred,
            out_dir / f"{group}_loss.png",
            f"{group}: validation/training loss curves",
            "loss",
        )
        plot_group_overlay(
            metric_files,
            reward_preferred,
            out_dir / f"{group}_reward.png",
            f"{group}: rollout reward curves",
            "reward",
        )

    def sort_key(row: Dict[str, object]):
        for k in ("group", "full_eval_loss", "last100_full_eval_loss", "eval_loss", "last100_eval_loss", "final_loss"):
            if k == "group":
                continue
            v = to_float(row.get(k))
            if v is not None:
                return (str(row.get("group", "")), v)
        return (str(row.get("group", "")), float("inf"))

    summaries.sort(key=sort_key)
    write_summary(summaries, out_dir / "training_summary.csv")
    (out_dir / "training_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    return summaries


def plot_test_bar(rows: List[Dict[str, str]], metric: str, out_path: Path, title: str, ylabel: str, higher_better: bool) -> bool:
    pairs = []
    for r in rows:
        if str(r.get("status", "ok")) not in {"", "ok"}:
            continue
        v = to_float(r.get(metric))
        if v is None:
            continue
        pairs.append((str(r.get("model", "model")), v))
    if not pairs:
        return False
    labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def visualize_test_results(test_result_dir: Path, out_dir: Path) -> Dict[str, object]:
    csv_path = test_result_dir / "test_rl_qra.csv"
    rows = read_csv(csv_path)
    # Backward compatibility only: old temporary script wrote test_rl_qa.csv.
    # New workflow should run test_rl_qra.py and produce test_rl_qra.csv.
    if not rows:
        legacy = test_result_dir / "test_rl_qa.csv"
        legacy_rows = read_csv(legacy)
        if legacy_rows:
            csv_path = legacy
            rows = legacy_rows
    summary: Dict[str, object] = {"test_result_csv": str(csv_path), "rows": len(rows)}
    if not rows:
        return summary
    plot_test_bar(rows, "test_loss", out_dir / "test_loss_compare.png", "Test set loss comparison", "test loss", higher_better=False)
    plot_test_bar(rows, "test_rollout_reward", out_dir / "test_reward_compare.png", "Test set rollout reward comparison", "test rollout reward", higher_better=True)
    write_summary([{k: v for k, v in r.items()} for r in rows], out_dir / "test_summary.csv")
    (out_dir / "test_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    # compact winners
    valid = [r for r in rows if str(r.get("status", "ok")) in {"", "ok"}]
    if valid:
        best_loss = min(valid, key=lambda r: to_float(r.get("test_loss")) if to_float(r.get("test_loss")) is not None else float("inf"))
        best_reward = max(valid, key=lambda r: to_float(r.get("test_rollout_reward")) if to_float(r.get("test_rollout_reward")) is not None else -float("inf"))
        summary["best_by_test_loss"] = best_loss.get("model")
        summary["best_by_test_reward"] = best_reward.get("model")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact RL-QRA visualizer for training curves and test-set model comparison.")
    parser.add_argument("--run-root", action="append", default=None, help="RL root to visualize. Can be repeated. Default: rl_pretrain and rl_QRA.")
    parser.add_argument("--test-result-dir", default=str(DEFAULT_TEST_RESULT_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--no-training", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    training_summary: List[Dict[str, object]] = []
    test_summary: Dict[str, object] = {"skipped": True}

    if not args.no_training:
        run_roots = [Path(x) for x in args.run_root] if args.run_root else list(DEFAULT_RUN_ROOTS)
        training_summary = visualize_training(run_roots, out_dir)
    if not args.no_test:
        test_summary = visualize_test_results(Path(args.test_result_dir), out_dir)

    final = {
        "run_roots": [str(x) for x in (args.run_root or DEFAULT_RUN_ROOTS)],
        "test_result_dir": str(args.test_result_dir),
        "out_dir": str(out_dir),
        "num_training_runs": len(training_summary),
        "test": test_summary,
        "plots": [
            "rl_pretrain_loss.png",
            "rl_pretrain_reward.png",
            "rl_QRA_loss.png",
            "rl_QRA_reward.png",
            "test_loss_compare.png",
            "test_reward_compare.png",
        ],
    }
    (out_dir / "visual_summary.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote visuals to: {out_dir}")
    print(f"Training runs visualized: {len(training_summary)}")
    if not args.no_test:
        print(f"Test rows visualized: {test_summary.get('rows', 0)}")


if __name__ == "__main__":
    main()
