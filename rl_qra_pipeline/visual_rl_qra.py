import argparse
import csv
import json
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "rl_qra_config.yaml"


def repo_path(path):
    p = Path(path)
    return p if p.is_absolute() else REPO_DIR / p


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def best_metrics_path(root, name):
    root = Path(root)
    direct = root / name / "best_metrics.csv"
    if direct.exists():
        return direct
    best_eval = root / name / "best_eval.json"
    if best_eval.exists():
        info = json.loads(best_eval.read_text(encoding="utf-8"))
        run_dir = info.get("run_dir")
        if run_dir and Path(run_dir).exists():
            path = Path(run_dir) / "metrics.csv"
            if path.exists():
                return path
    return root / name / "metrics.csv"


def step_bin_end(step, bin_size):
    if step <= 0:
        return 0
    return ((step - 1) // bin_size + 1) * bin_size


def aggregate(rows, bin_size):
    buckets = {}
    for row in rows:
        b = step_bin_end(row["step"], bin_size)
        buckets.setdefault(b, []).append(row["loss"])
    return [{"step": b, "loss": sum(vals) / len(vals), "count": len(vals)} for b, vals in sorted(buckets.items())]


def plot_curves(root, out_dir, names, bin_size=25, show_raw=False):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plotted = False
    for name in names:
        path = best_metrics_path(root, name)
        if not path.exists():
            print(f"Skip {name}: {path} not found")
            continue
        rows = []
        for row in read_csv(path):
            try:
                rows.append({"step": int(row["step"]), "loss": float(row["loss"])})
            except (KeyError, ValueError):
                pass
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r["step"])
        if show_raw:
            plt.scatter([r["step"] for r in rows], [r["loss"] for r in rows], s=8, alpha=0.18)
        mean_rows = aggregate(rows, bin_size)
        plt.plot(
            [r["step"] for r in mean_rows],
            [r["loss"] for r in mean_rows],
            marker="o",
            linewidth=1.8,
            label=name,
        )
        plotted = True

    if not plotted:
        return
    plt.title("RL-QRA Guided Ratio Loss")
    plt.xlabel("step")
    plt.ylabel("guided local ratio loss")
    plt.grid(alpha=0.25)
    plt.legend()
    out = Path(out_dir) / "rl_qra_learning_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Saved {out}")


def plot_test(test_dir, out_dir):
    import matplotlib.pyplot as plt

    path = Path(test_dir) / "test_results.csv"
    if not path.exists():
        print(f"Skip test: {path} not found")
        return
    rows = []
    for row in read_csv(path):
        try:
            rows.append({"dataset": row["dataset"], "model": row["model"], "loss": float(row["loss"])})
        except (KeyError, ValueError):
            pass
    if not rows:
        return
    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    models = list(dict.fromkeys(row["model"] for row in rows))
    values = {(row["dataset"], row["model"]): row["loss"] for row in rows}
    width = 0.8 / max(1, len(models))
    x = list(range(len(datasets)))

    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        offsets = [pos - 0.4 + width / 2 + i * width for pos in x]
        plt.bar(offsets, [values.get((dataset, model), 0) for dataset in datasets], width=width, label=model)
    plt.xticks(x, datasets)
    plt.ylabel("target-token score entropy loss")
    plt.title("RL-QRA Test Loss Comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    out = Path(out_dir) / "rl_qra_test_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot RL-QRA training and test results.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config = load_config(args.config)
    root = repo_path(config["output"].get("root_dir", "rl_qra_pipeline/modelparameter"))
    out_dir = repo_path(config.get("visual", {}).get("output_dir", "rl_qra_pipeline/visualization"))
    test_dir = repo_path(config.get("eval", {}).get("output_dir", "rl_qra_pipeline/test_result"))
    bin_size = int(config.get("visual", {}).get("bin_size", 25))
    show_raw = bool(config.get("visual", {}).get("show_raw", False))
    plot_curves(root, out_dir, ["rl_pretrain", "rl_QRA"], bin_size=bin_size, show_raw=show_raw)
    plot_test(test_dir, out_dir)


if __name__ == "__main__":
    main()
