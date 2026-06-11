from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

DEFAULT_RUN_ROOT = SCRIPT_DIR / "modelparameter" / "rl_QRA"
DEFAULT_OUT_DIR = SCRIPT_DIR / "visual" / "rl_QRA"


ROLLOUT_METRICS = [
    "loss",
    "rollout_loss",
    "rollout_reward",
    "rollout_reward_min",
    "rollout_reward_max",
    "rollout_reward_std",
    "rollout_entropy",
    "rollout_logprob",
    "rollout_anchor_loss",
]
LEGACY_METRICS = [
    "target_logp",
    "model_reward",
    "best_reward",
    "reward_gap",
    "candidate_entropy",
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


def norm_answer(text: str) -> str:
    s = str(text or "")
    s = s.replace("[MASK]", "□")
    s = s.replace("\\mathrm{~m}", "m").replace("\\mathrm{m}", "m")
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\s+", "", s)
    if len(s) > 1 and s[-1] in ".;":
        s = s[:-1]
    return s


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


def plot_multi_series(series: Dict[str, List[Tuple[float, float]]], out_path: Path, title: str, ylabel: str) -> bool:
    series = {k: v for k, v in series.items() if v}
    if not series:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for label, pts in series.items():
        pts = sorted(pts, key=lambda x: x[0])
        plt.plot([x for x, _ in pts], [y for _, y in pts], linewidth=1.2, label=label)
    plt.xlabel("generation step")
    plt.ylabel(ylabel)
    plt.title(title)
    if len(series) <= 12:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def summarize_metrics(path: Path) -> Dict[str, object]:
    rows = read_csv(path)
    if not rows:
        return {"metrics_path": str(path), "rows": 0}
    tail = rows[-min(100, len(rows)) :]
    summary: Dict[str, object] = {"run_dir": str(path.parent), "metrics_path": str(path), "rows": len(rows)}
    for key in ROLLOUT_METRICS + LEGACY_METRICS:
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
                for key in ["eval_loss", "eval_count", "eval_split", "rollout_reward", "rollout_entropy"]:
                    if key in obj:
                        summary[key] = obj[key]
        except Exception:
            pass
    return summary


def write_table(rows: List[Dict[str, object]], path: Path) -> None:
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
    files.extend(sorted(p for p in run_root.glob("*/metrics.csv") if "/runs/" not in str(p)))
    seen = set()
    out = []
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            out.append(p)
            seen.add(rp)
    return out


def newest_sample_chain_dir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / "detail.csv").exists()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def read_chain_detail(chain_dir: Path) -> List[Dict[str, str]]:
    return read_csv(chain_dir / "detail.csv")


def aggregate_chain(rows: List[Dict[str, str]]) -> Tuple[Dict[str, List[Tuple[float, float]]], Dict[str, List[Tuple[float, float]]]]:
    mask_values: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)
    exact_values: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)
    for row in rows:
        dataset = row.get("dataset", "")
        model = row.get("model", "")
        step = to_float(row.get("step"))
        if step is None:
            continue
        key = (dataset, model, int(step))
        m = to_float(row.get("mask_count"))
        if m is not None:
            mask_values[key].append(m)
        ans = norm_answer(row.get("answer", ""))
        gt = norm_answer(row.get("gt_answer", ""))
        if gt:
            exact_values[key].append(1.0 if ans == gt else 0.0)
    mask_series: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    exact_series: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for (dataset, model, step), vals in mask_values.items():
        mask_series[f"{dataset}/{model}"].append((float(step), sum(vals) / len(vals)))
    for (dataset, model, step), vals in exact_values.items():
        exact_series[f"{dataset}/{model}"].append((float(step), sum(vals) / len(vals)))
    return dict(mask_series), dict(exact_series)


def visualize_metrics(run_root: Path, out_dir: Path) -> List[Dict[str, object]]:
    metric_files = collect_metric_files(run_root)
    summaries: List[Dict[str, object]] = []
    for path in metric_files:
        rows = read_csv(path)
        run_name = "global_best" if path.name == "best_metrics.csv" else path.parent.name
        summaries.append({"run": run_name, **summarize_metrics(path)})
        for metric in ROLLOUT_METRICS + LEGACY_METRICS:
            plot_metric(rows, metric, out_dir / "training" / run_name / f"{metric}.png", f"{run_name}: {metric}")
    def sort_key(row: Dict[str, object]):
        for key in ("eval_loss", "last100_loss", "final_loss"):
            val = to_float(row.get(key))
            if val is not None:
                return val
        return float("inf")
    summaries.sort(key=sort_key)
    write_table(summaries, out_dir / "training_summary.csv")
    (out_dir / "training_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    return summaries


def visualize_chain(chain_dir: Optional[Path], out_dir: Path) -> Dict[str, object]:
    if chain_dir is None or not chain_dir.exists():
        return {"chain_dir": str(chain_dir) if chain_dir else "", "rows": 0}
    rows = read_chain_detail(chain_dir)
    mask_series, exact_series = aggregate_chain(rows)
    chain_out = out_dir / "sample_chain"
    plot_multi_series(mask_series, chain_out / "mean_mask_count.png", "Sample chain: mean remaining mask count", "mean mask count")
    plot_multi_series(exact_series, chain_out / "exact_rate_by_step.png", "Sample chain: exact answer rate by step", "exact rate")
    summary = {"chain_dir": str(chain_dir), "rows": len(rows), "series": sorted(set(mask_series) | set(exact_series))}
    (chain_out / "chain_visual_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (chain_out / "chain_visual_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize S1K rollout slot-alignment RL training metrics and sample-chain diagnostics.")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT), help="Usually rl_qra_pipeline/modelparameter/rl_QRA")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--chain-dir", default="", help="Optional experiment/sample_chain/<run> directory. Default: latest under experiment/sample_chain.")
    parser.add_argument("--no-chain", action="store_true")
    args = parser.parse_args()
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = visualize_metrics(run_root, out_dir)
    chain_summary = {"skipped": True}
    if not args.no_chain:
        chain_dir = Path(args.chain_dir) if args.chain_dir else newest_sample_chain_dir(REPO_DIR / "experiment" / "sample_chain")
        chain_summary = visualize_chain(chain_dir, out_dir)

    final = {"run_root": str(run_root), "out_dir": str(out_dir), "num_metric_runs": len(summaries), "chain": chain_summary}
    (out_dir / "visual_summary.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote visuals to: {out_dir}")
    print(f"Training runs visualized: {len(summaries)}")
    if not args.no_chain:
        print(f"Chain visualized: {chain_summary.get('chain_dir', '')}")


if __name__ == "__main__":
    main()
