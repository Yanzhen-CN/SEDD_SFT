import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from rl_utils import load_config


DEFAULT_CONFIGS = [
    "sft_rl_pipeline/rl_config.yaml",
    "sft_rl_pipeline/rl_config_pretrained.yaml",
]


def read_metrics(path):
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            parsed = dict(row)
            parsed["step"] = int(float(row["step"]))
            parsed["loss"] = float(row["loss"]) if row.get("loss") else None
            parsed["reward"] = float(row["reward"]) if row.get("reward") else None
            rows.append(parsed)
    return rows


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def experiment_from_config(config_path):
    config = load_config(config_path)
    out_root = Path(config["results"]["output_dir"])
    return {
        "name": out_root.name,
        "config": str(config_path),
        "out_root": out_root,
        "run_dirs": sorted(path for path in out_root.glob("*") if path.is_dir() and (path / "metrics.csv").exists()),
    }


def plot_learning_curves(experiments, figure_dir):
    plt.figure(figsize=(11, 5))
    plotted = False
    for exp in experiments:
        for run_dir in exp["run_dirs"]:
            rows = read_metrics(run_dir / "metrics.csv")
            train = [row for row in rows if row["split"] == "train"]
            valid = [row for row in rows if row["split"] == "validation"]
            label = f"{exp['name']}/{run_dir.name}"
            if train:
                plt.plot([row["step"] for row in train], [row["loss"] for row in train], alpha=0.35, linestyle="--", label=f"{label} train")
                plotted = True
            if valid:
                plt.plot([row["step"] for row in valid], [row["loss"] for row in valid], marker="o", label=f"{label} valid")
                plotted = True
    if not plotted:
        return None
    plt.xlabel("step")
    plt.ylabel("masked DWDSE loss")
    plt.title("RL learning curves")
    plt.legend(fontsize=7)
    plt.tight_layout()
    path = figure_dir / "rl_learning_curves.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_reward_curves(experiments, figure_dir):
    plt.figure(figsize=(11, 5))
    plotted = False
    for exp in experiments:
        for run_dir in exp["run_dirs"]:
            rows = [row for row in read_metrics(run_dir / "metrics.csv") if row["split"] == "train" and row["reward"] is not None]
            if not rows:
                continue
            plt.plot([row["step"] for row in rows], [row["reward"] for row in rows], marker=".", label=f"{exp['name']}/{run_dir.name}")
            plotted = True
    if not plotted:
        return None
    plt.xlabel("step")
    plt.ylabel("batch reward")
    plt.title("Reward curves")
    plt.legend(fontsize=7)
    plt.tight_layout()
    path = figure_dir / "rl_reward_curves.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_run_bars(experiments, figure_dir):
    labels = []
    start_losses = []
    best_losses = []
    for exp in experiments:
        for run_dir in exp["run_dirs"]:
            rows = read_metrics(run_dir / "metrics.csv")
            start = next((row["loss"] for row in rows if row["split"] == "pretrain_validation"), None)
            best = read_json(run_dir / "best_eval.json", {})
            if start is None or not best:
                continue
            labels.append(f"{exp['name']}/{run_dir.name}")
            start_losses.append(start)
            best_losses.append(float(best["validation_loss"]))
    if not labels:
        return None

    x = range(len(labels))
    width = 0.36
    plt.figure(figsize=(max(10, len(labels) * 1.2), 5))
    plt.bar([i - width / 2 for i in x], start_losses, width=width, label="start")
    plt.bar([i + width / 2 for i in x], best_losses, width=width, label="best")
    plt.xticks(list(x), labels, rotation=35, ha="right")
    plt.ylabel("validation loss")
    plt.title("Each RL run: start vs best")
    plt.legend()
    plt.tight_layout()
    path = figure_dir / "rl_run_start_vs_best.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def metric_from_rows(path, split):
    if not Path(path).exists():
        return None
    for row in read_metrics(path):
        if row["split"] == split:
            return row["loss"]
    return None


def plot_final_comparison(experiments, figure_dir):
    bars = []
    for exp in experiments:
        best_metrics = exp["out_root"] / "best_metrics.csv"
        start = metric_from_rows(best_metrics, "pretrain_validation")
        best = read_json(exp["out_root"] / "best_eval.json", {})
        if start is not None:
            bars.append((f"{exp['name']} start", start))
        if best and "validation_loss" in best:
            bars.append((f"{exp['name']} best", float(best["validation_loss"])))
    if not bars:
        return None

    labels = [label for label, _ in bars]
    values = [value for _, value in bars]
    plt.figure(figsize=(max(8, len(labels) * 1.4), 5))
    plt.bar(labels, values)
    plt.ylabel("validation loss")
    plt.title("Final comparison")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    path = figure_dir / "rl_final_comparison.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_test_comparison(figure_dir, test_path=Path("sft_rl_pipeline/test_results/rl_test_results.csv")):
    if not test_path.exists():
        return None
    rows = []
    with open(test_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append((row["name"], float(row["test_loss"])))
    if not rows:
        return None

    labels = [label for label, _ in rows]
    values = [value for _, value in rows]
    plt.figure(figsize=(max(7, len(labels) * 1.3), 5))
    plt.bar(labels, values)
    plt.ylabel("test loss")
    plt.title("QAR test loss comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = figure_dir / "rl_test_comparison.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def write_summary(experiments, figure_dir):
    summary = []
    for exp in experiments:
        best = read_json(exp["out_root"] / "best_eval.json", {})
        start = metric_from_rows(exp["out_root"] / "best_metrics.csv", "pretrain_validation")
        summary.append(
            {
                "name": exp["name"],
                "config": exp["config"],
                "output_dir": str(exp["out_root"]),
                "num_runs": len(exp["run_dirs"]),
                "start_validation_loss": start,
                "best_validation_loss": best.get("validation_loss"),
                "best_step": best.get("step"),
                "best_run_dir": best.get("run_dir"),
                "best_checkpoint": best.get("checkpoint_path"),
            }
        )
    path = figure_dir / "rl_visual_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Visualize all SFT-RL experiments from saved metrics.")
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--output-dir", default="sft_rl_pipeline/visualization")
    args = parser.parse_args()

    experiments = [experiment_from_config(Path(path)) for path in args.configs]
    figure_dir = Path(args.output_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        plot_learning_curves(experiments, figure_dir),
        plot_reward_curves(experiments, figure_dir),
        plot_run_bars(experiments, figure_dir),
        plot_final_comparison(experiments, figure_dir),
        plot_test_comparison(figure_dir),
        write_summary(experiments, figure_dir),
    ]
    for path in outputs:
        if path:
            print(path)


if __name__ == "__main__":
    main()
