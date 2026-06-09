import argparse
import csv
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR / "modelparameter"
DEFAULT_OUT = SCRIPT_DIR / "visualization"


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def best_metrics_path(root, name):
    run_root = Path(root) / name
    best_eval = run_root / "best_eval.json"
    if best_eval.exists():
        with open(best_eval, "r", encoding="utf-8") as f:
            run_instance = json.load(f).get("run_instance")
        if run_instance:
            full_run_metrics = run_root / run_instance / "metrics.csv"
            if full_run_metrics.exists():
                return full_run_metrics
    direct = run_root / "best_metrics.csv"
    if direct.exists():
        return direct
    return run_root / "metrics.csv"


def step_bin_end(step, bin_size):
    if step <= 0:
        return 0
    return ((step - 1) // bin_size + 1) * bin_size


def aggregate_by_step_mean(rows, split, bin_size):
    buckets = {}
    for row in rows:
        if row["split"] != split:
            continue
        b = step_bin_end(row["step"], bin_size)
        buckets.setdefault(b, []).append(row["loss"])
    return [{"step": b, "loss": sum(vals) / len(vals), "count": len(vals)} for b, vals in sorted(buckets.items())]


def plot_curve(root, name, out_dir, bin_size=25, show_raw=False):
    import matplotlib.pyplot as plt

    path = best_metrics_path(root, name)
    if not path.exists():
        print(f"Skip {name}: {path} not found")
        return
    rows = []
    for row in read_csv(path):
        try:
            rows.append({"step": int(row["step"]), "split": row["split"], "loss": float(row["loss"])})
        except (KeyError, ValueError):
            pass
    plt.figure(figsize=(10, 6))
    train_raw = sorted([r for r in rows if r["split"] == "train"], key=lambda r: r["step"])
    train_mean = aggregate_by_step_mean(rows, "train", bin_size)
    if show_raw and train_raw:
        plt.scatter([r["step"] for r in train_raw], [r["loss"] for r in train_raw], s=8, alpha=0.20, label=f"train raw ({len(train_raw)} points)")
    if train_mean:
        plt.plot([r["step"] for r in train_mean], [r["loss"] for r in train_mean], marker="o", markersize=4, linewidth=1.8, label=f"train mean per {bin_size} steps")
    validation = sorted([r for r in rows if r["split"] == "validation"], key=lambda r: r["step"])
    if validation:
        plt.plot([r["step"] for r in validation], [r["loss"] for r in validation], marker="o", markersize=5, linewidth=2.0, label="validation")
    baseline = [r for r in rows if r["split"] == "pretrain_validation"]
    if baseline:
        plt.axhline(baseline[0]["loss"], linestyle="--", linewidth=1.8, label=f"pretrained validation = {baseline[0]['loss']:.4f}")
    plt.title(f"{name} QRA SFT Loss")
    plt.xlabel("step")
    plt.ylabel("train-token score entropy loss")
    plt.grid(alpha=0.25)
    plt.legend()
    out = Path(out_dir) / f"{name.lower()}_qra_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Saved {out}")


def plot_test(root, out_dir):
    import matplotlib.pyplot as plt

    path = Path(root) / "test_result" / "test_results.csv"
    if not path.exists():
        print(f"Skip test: {path} not found")
        return
    rows = []
    for row in read_csv(path):
        try:
            rows.append({"dataset": row["dataset"], "model": row["model"], "loss": float(row["loss"])})
        except (KeyError, ValueError):
            pass
    datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    models = list(dict.fromkeys(r["model"] for r in rows))
    values = {(r["dataset"], r["model"]): r["loss"] for r in rows}
    width = 0.8 / max(1, len(models))
    x = list(range(len(datasets)))
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        offsets = [pos - 0.4 + width / 2 + i * width for pos in x]
        plt.bar(offsets, [values.get((dataset, model), 0) for dataset in datasets], width=width, label=model)
    plt.xticks(x, datasets)
    plt.ylabel("train-token score entropy loss")
    plt.title("QRA SFT Test Loss")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    out = Path(out_dir) / "qra_test_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot QRA reasoning-conditioned SFT curves and test comparison.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--qra", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--bin-size", type=int, default=25)
    parser.add_argument("--show-raw", action="store_true")
    args = parser.parse_args()
    plot_all = not any([args.qra, args.test])
    if plot_all or args.qra:
        plot_curve(args.root, "QRA", args.out_dir, bin_size=args.bin_size, show_raw=args.show_raw)
    if plot_all or args.test:
        plot_test(args.root, args.out_dir)


if __name__ == "__main__":
    main()
