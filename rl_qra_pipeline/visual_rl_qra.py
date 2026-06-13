from __future__ import annotations

"""Compact visualizer for RL-QRA.

Outputs:
1. Training curves from modelparameter/rl_<start>/<run_id>/metrics.csv
   - per-group loss curves
   - per-group reward curves
   - per-group combined loss/reward overlay with twin y-axes
2. Test comparison plots from modelparameter/test_result/test_rl_qra.csv
   - test loss bar chart with pretrain/QRA start baselines and reward_best loss lines
   - test reward bar chart with pretrain/QRA start baselines and reward_best reward lines
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

PRETRAIN_BASELINE_NAMES = {"pretrain", "pretrain_start", "start_pretrain"}
QRA_BASELINE_NAMES = {"qra", "qra_start", "start_qra", "QRA"}

# Keep the visual language simple and high-contrast.
LOSS_COLOR = "tab:blue"
REWARD_COLOR = "tab:orange"
PRETRAIN_COLOR = "tab:blue"
QRA_COLOR = "tab:orange"
OTHER_COLOR = "0.68"
LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["", "o", "s", "^"]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
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


def model_family(model_name: str) -> str:
    s = str(model_name or "").lower()
    if "qra" in s:
        return "QRA"
    if "pretrain" in s:
        return "pretrain"
    if "qa" in s:
        return "QA"
    return "other"


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


def choose_metric_series(rows: List[Dict[str, str]], preferred: Sequence[str]) -> Tuple[str, List[Tuple[float, float]]]:
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


def plot_group_loss_reward_overlay(
    metric_files: List[Tuple[str, Path]],
    loss_preferred: Sequence[str],
    reward_preferred: Sequence[str],
    out_path: Path,
    title: str,
) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax_loss = plt.subplots(figsize=(10.8, 5.8))
    ax_reward = ax_loss.twinx()
    plotted = False

    for idx, (run_name, path) in enumerate(metric_files):
        rows = read_csv(path)
        loss_metric, loss_pts = choose_metric_series(rows, loss_preferred)
        reward_metric, reward_pts = choose_metric_series(rows, reward_preferred)
        linestyle = LINE_STYLES[idx % len(LINE_STYLES)]
        marker = MARKERS[idx % len(MARKERS)]
        markevery = max(1, len(rows) // 12) if rows else None

        if loss_pts:
            loss_pts = sorted(loss_pts, key=lambda x: x[0])
            label = f"loss:{run_name}" if loss_metric == loss_preferred[0] else f"loss:{run_name} ({loss_metric})"
            ax_loss.plot(
                [x for x, _ in loss_pts],
                [y for _, y in loss_pts],
                color=LOSS_COLOR,
                linewidth=2.2,
                linestyle=linestyle,
                marker=marker or None,
                markersize=3.2,
                markevery=markevery,
                alpha=0.92,
                label=label,
            )
            plotted = True
        if reward_pts:
            reward_pts = sorted(reward_pts, key=lambda x: x[0])
            label = f"reward:{run_name}" if reward_metric == reward_preferred[0] else f"reward:{run_name} ({reward_metric})"
            ax_reward.plot(
                [x for x, _ in reward_pts],
                [y for _, y in reward_pts],
                color=REWARD_COLOR,
                linewidth=2.2,
                linestyle=linestyle,
                marker=marker or None,
                markersize=3.2,
                markevery=markevery,
                alpha=0.92,
                label=label,
            )
            plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("loss", color=LOSS_COLOR)
    ax_reward.set_ylabel("reward", color=REWARD_COLOR)
    ax_loss.tick_params(axis="y", labelcolor=LOSS_COLOR)
    ax_reward.tick_params(axis="y", labelcolor=REWARD_COLOR)
    ax_loss.grid(True, alpha=0.22)
    ax_loss.set_title(title)

    handles1, labels1 = ax_loss.get_legend_handles_labels()
    handles2, labels2 = ax_reward.get_legend_handles_labels()
    if handles1 or handles2:
        ax_loss.legend(handles1 + handles2, labels1 + labels2, fontsize=7, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
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
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def group_name_from_root(run_root: Path) -> str:
    return safe_name(run_root.name or "rl_group")


def visualize_training(run_roots: Sequence[Path], out_dir: Path) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    loss_preferred = ["full_eval_loss", "eval_loss", "loss"]
    reward_preferred = ["full_eval_rollout_reward", "eval_rollout_reward", "rollout_reward", "reward_best_metric_value"]

    for run_root in run_roots:
        group = group_name_from_root(run_root)
        for stale in (out_dir / f"{group}_loss.png", out_dir / f"{group}_reward.png"):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass
        metric_files = collect_metric_files(run_root)
        for run_name, metrics_path in metric_files:
            item = summarize_metric_file(f"{group}/{run_name}", metrics_path)
            item["group"] = group
            summaries.append(item)
        plot_group_loss_reward_overlay(
            metric_files,
            loss_preferred,
            reward_preferred,
            out_dir / f"{group}_loss_reward_overlay.png",
            f"{group}: loss/reward overlay",
        )

    def sort_key(row: Dict[str, object]):
        for k in ("full_eval_loss", "last100_full_eval_loss", "eval_loss", "last100_eval_loss", "final_loss"):
            v = to_float(row.get(k))
            if v is not None:
                return (str(row.get("group", "")), v)
        return (str(row.get("group", "")), float("inf"))

    summaries.sort(key=sort_key)
    write_summary(summaries, out_dir / "training_summary.csv")
    (out_dir / "training_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    return summaries


def find_baseline_values(rows: List[Dict[str, str]], metric: str) -> Dict[str, float]:
    baselines: Dict[str, float] = {}
    for r in rows:
        if str(r.get("status", "ok")) not in {"", "ok"}:
            continue
        model_raw = str(r.get("model", ""))
        model = model_raw.lower()
        v = to_float(r.get(metric))
        if v is None:
            continue
        if model in PRETRAIN_BASELINE_NAMES or model_raw in PRETRAIN_BASELINE_NAMES:
            baselines.setdefault("pretrain start", v)
        if model in {x.lower() for x in QRA_BASELINE_NAMES} or model_raw in QRA_BASELINE_NAMES:
            baselines.setdefault("QRA start", v)
    return baselines


def plot_test_bar(
    rows: List[Dict[str, str]],
    metric: str,
    out_path: Path,
    title: str,
    ylabel: str,
    higher_better: bool,
    extra_hlines: Optional[List[Tuple[str, float, str, str]]] = None,  # (label, value, color, linestyle)
) -> bool:
    """Draw test bar chart; optionally add extra horizontal lines with custom colors."""
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
    bar_colors = []
    for label in labels:
        family = model_family(label)
        if family == "pretrain":
            bar_colors.append(PRETRAIN_COLOR)
        elif family == "QRA":
            bar_colors.append(QRA_COLOR)
        else:
            bar_colors.append(OTHER_COLOR)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10.5, 5.2))
    plt.bar(labels, values, color=bar_colors, alpha=0.82)

    baseline_values = find_baseline_values(rows, metric)
    baseline_style = {
        "pretrain start": (PRETRAIN_COLOR, "--"),
        "QRA start": (QRA_COLOR, "-."),
    }
    for name, value in baseline_values.items():
        color, linestyle = baseline_style.get(name, ("0.3", ":"))
        plt.axhline(value, color=color, linestyle=linestyle, linewidth=1.4, label=f"{name}: {value:.4g}")

    if extra_hlines:
        for label, value, color, linestyle in extra_hlines:
            if value is not None:
                plt.axhline(value, color=color, linestyle=linestyle, linewidth=1.2, label=f"{label}: {value:.4g}")

    plt.xticks(rotation=25, ha="right")
    plt.ylabel(ylabel)
    direction = "higher is better" if higher_better else "lower is better"
    plt.title(f"{title} ({direction})")
    if baseline_values or extra_hlines:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def visualize_test_results(test_result_dir: Path, out_dir: Path) -> Dict[str, object]:
    csv_path = test_result_dir / "test_rl_qra.csv"
    rows = read_csv(csv_path)
    if not rows:
        legacy = test_result_dir / "test_rl_qa.csv"
        legacy_rows = read_csv(legacy)
        if legacy_rows:
            csv_path = legacy
            rows = legacy_rows
    summary: Dict[str, object] = {"test_result_csv": str(csv_path), "rows": len(rows)}
    if not rows:
        return summary

    valid_rows = [r for r in rows if str(r.get("status", "ok")) in {"", "ok"}]
    pretrain_rows = [r for r in valid_rows if model_family(r.get("model", "")) == "pretrain"]
    qra_rows = [r for r in valid_rows if model_family(r.get("model", "")) == "QRA"]

    # Find best reward (max test_rollout_reward) for each family
    pretrain_best_reward_loss = None
    qra_best_reward_loss = None
    pretrain_best_reward_value = None
    qra_best_reward_value = None

    if pretrain_rows:
        best = max(pretrain_rows, key=lambda r: to_float(r.get("test_rollout_reward")) or -float("inf"))
        pretrain_best_reward_loss = to_float(best.get("test_loss"))
        pretrain_best_reward_value = to_float(best.get("test_rollout_reward"))
    if qra_rows:
        best = max(qra_rows, key=lambda r: to_float(r.get("test_rollout_reward")) or -float("inf"))
        qra_best_reward_loss = to_float(best.get("test_loss"))
        qra_best_reward_value = to_float(best.get("test_rollout_reward"))

    # Build extra lines for loss plot: (label, value, color, linestyle)
    extra_loss_lines = []
    if pretrain_best_reward_loss is not None:
        extra_loss_lines.append(("pretrain best reward loss", pretrain_best_reward_loss, PRETRAIN_COLOR, ":"))
    if qra_best_reward_loss is not None:
        extra_loss_lines.append(("QRA best reward loss", qra_best_reward_loss, QRA_COLOR, ":"))

    # Build extra lines for reward plot: (label, value, color, linestyle)
    extra_reward_lines = []
    if pretrain_best_reward_value is not None:
        extra_reward_lines.append(("pretrain best reward", pretrain_best_reward_value, PRETRAIN_COLOR, ":"))
    if qra_best_reward_value is not None:
        extra_reward_lines.append(("QRA best reward", qra_best_reward_value, QRA_COLOR, ":"))

    # Loss bar chart
    plot_test_bar(
        rows,
        "test_loss",
        out_dir / "test_loss_compare.png",
        "Test set loss comparison",
        "test loss",
        higher_better=False,
        extra_hlines=extra_loss_lines,
    )

    # Reward bar chart
    plot_test_bar(
        rows,
        "test_rollout_reward",
        out_dir / "test_reward_compare.png",
        "Test set rollout reward comparison",
        "test rollout reward",
        higher_better=True,
        extra_hlines=extra_reward_lines,
    )

    write_summary([{k: v for k, v in r.items()} for r in rows], out_dir / "test_summary.csv")
    (out_dir / "test_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

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
            "rl_pretrain_loss_reward_overlay.png",
            "rl_QRA_loss_reward_overlay.png",
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