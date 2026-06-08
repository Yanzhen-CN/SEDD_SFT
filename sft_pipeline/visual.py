import argparse
import csv
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = SCRIPT_DIR / "modelparameter"
DEFAULT_TEST_CSV = DEFAULT_MODEL_ROOT / "test_result" / "test_results.csv"
DEFAULT_OUT_DIR = SCRIPT_DIR / "visualization"


def read_csv(path):
    """Read CSV and return list of dicts."""
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def find_best_metrics(run_root):
    """
    Find the best metrics.csv file for a run (QA or QAR).
    Priority:
      1. best_metrics.csv (if exists)
      2. run_instance subdirectory from best_eval.json
      3. fallback to metrics.csv in run_root
    """
    run_root = Path(run_root)
    direct = run_root / "best_metrics.csv"
    if direct.exists():
        return direct

    best_eval = run_root / "best_eval.json"
    if best_eval.exists():
        with open(best_eval, "r", encoding="utf-8") as f:
            data = json.load(f)
            run_instance = data.get("run_instance")
        if run_instance:
            candidate = run_root / run_instance / "metrics.csv"
            if candidate.exists():
                return candidate

    return run_root / "metrics.csv"


def step_bin_end(step, bin_size):
    """
    Map steps 1..bin_size to bin_size,
    steps bin_size+1..2*bin_size to 2*bin_size, etc.
    Step 0 stays 0.
    """
    if step <= 0:
        return 0
    return ((step - 1) // bin_size + 1) * bin_size


def aggregate_by_step_mean(rows, split, bin_size):
    """
    Average all rows of a split within each fixed step window.
    """
    buckets = {}
    for row in rows:
        if row["split"] != split:
            continue
        b = step_bin_end(row["step"], bin_size)
        buckets.setdefault(b, []).append(row["loss"])

    aggregated = []
    for b in sorted(buckets):
        losses = buckets[b]
        aggregated.append(
            {
                "step": b,
                "loss": sum(losses) / len(losses),
                "count": len(losses),
            }
        )
    return aggregated


def plot_curve(name, model_root, out_dir, bin_size=25, show_raw=False):
    import matplotlib.pyplot as plt

    run_root = Path(model_root) / name
    metrics_path = find_best_metrics(run_root)
    if not metrics_path.exists():
        print(f"Skip {name}: metrics file not found at {metrics_path}")
        return

    rows = []
    for row in read_csv(metrics_path):
        try:
            step = int(row["step"])
            kind = row.get("kind", row.get("split", ""))
            loss = float(row["loss"])
            # Map kind to split name used in plotting
            if kind == "training":
                split = "train"
            elif kind == "evaluation":
                split = "validation"
            elif kind == "pretrain_evaluation":
                split = "pretrain_validation"
            else:
                split = kind  # fallback
            rows.append({"step": step, "split": split, "loss": loss})
        except (KeyError, ValueError):
            continue

    if not rows:
        print(f"Skip {name}: no valid data in {metrics_path}")
        return

    plt.figure(figsize=(10, 6))

    # Raw train points
    train_raw = [r for r in rows if r["split"] == "train"]
    if show_raw and train_raw:
        train_raw_sorted = sorted(train_raw, key=lambda r: r["step"])
        plt.scatter(
            [r["step"] for r in train_raw_sorted],
            [r["loss"] for r in train_raw_sorted],
            s=8,
            alpha=0.20,
            label=f"train raw ({len(train_raw_sorted)} points)",
        )

    # Binned train mean
    train_mean = aggregate_by_step_mean(rows, "train", bin_size)
    if train_mean:
        plt.plot(
            [r["step"] for r in train_mean],
            [r["loss"] for r in train_mean],
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=f"train mean per {bin_size} steps ({len(train_mean)} points)",
        )

    # Validation points
    validation = [r for r in rows if r["split"] == "validation"]
    if validation:
        validation_sorted = sorted(validation, key=lambda r: r["step"])
        plt.plot(
            [r["step"] for r in validation_sorted],
            [r["loss"] for r in validation_sorted],
            marker="o",
            markersize=5,
            linewidth=2.0,
            label=f"validation ({len(validation_sorted)} points)",
        )

    # Pretrained baseline
    baseline = [r for r in rows if r["split"] == "pretrain_validation"]
    if baseline:
        plt.axhline(
            baseline[0]["loss"],
            linestyle="--",
            linewidth=1.8,
            label=f"pretrained validation = {baseline[0]['loss']:.4f}",
        )

    plt.title(f"{name} SFT Training Curve")
    plt.xlabel("step")
    plt.ylabel("score entropy loss")
    plt.grid(alpha=0.25)
    plt.legend()

    out_path = Path(out_dir) / f"{name.lower()}_training_curve.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

    print(f"Saved {out_path}")

    if validation:
        best_val = min(validation, key=lambda r: r["loss"])
        last_val = validation[-1]
        print(
            f"{name} validation: best step={best_val['step']} loss={best_val['loss']:.4f}; "
            f"last step={last_val['step']} loss={last_val['loss']:.4f}"
        )
    if train_mean:
        print(
            f"{name} train mean: last window step={train_mean[-1]['step']} "
            f"loss={train_mean[-1]['loss']:.4f}, points in last window={train_mean[-1]['count']}"
        )


def plot_test_comparison(test_csv, out_dir):
    import matplotlib.pyplot as plt

    test_csv = Path(test_csv)
    if not test_csv.exists():
        print(f"Skip test comparison: {test_csv} not found")
        return

    rows = []
    for row in read_csv(test_csv):
        try:
            rows.append(
                {
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "loss": float(row["loss"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue

    if not rows:
        print("No valid rows in test CSV.")
        return

    # Prefer order: QA-test, QAR-test
    dataset_order = ["QA-test", "QAR-test"]
    model_order = ["pretrained", "QA-best", "QAR-best"]

    datasets = [d for d in dataset_order if any(r["dataset"] == d for r in rows)]
    models = [m for m in model_order if any(r["model"] == m for r in rows)]

    # Add any extra datasets/models not in predefined order
    for r in rows:
        if r["dataset"] not in datasets:
            datasets.append(r["dataset"])
        if r["model"] not in models:
            models.append(r["model"])

    values = {(r["dataset"], r["model"]): r["loss"] for r in rows}

    x = list(range(len(datasets)))
    width = 0.8 / max(1, len(models))

    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        offsets = [pos - 0.4 + width / 2 + i * width for pos in x]
        y = [values.get((dataset, model), float("nan")) for dataset in datasets]
        plt.bar(offsets, y, width=width, label=model)

    plt.xticks(x, datasets)
    plt.xlabel("test split")
    plt.ylabel("score entropy loss")
    plt.title("Final Test Loss Comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out_path = Path(out_dir) / "test_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot SFT training curves (QA/QAR) and test comparison from saved results."
    )
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT), help="Root directory containing QA/ and QAR/ subdirs")
    parser.add_argument("--test-csv", default=str(DEFAULT_TEST_CSV), help="Path to test_results.csv")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for images")
    parser.add_argument("--qa", action="store_true", help="Plot QA training curve")
    parser.add_argument("--qar", action="store_true", help="Plot QAR training curve")
    parser.add_argument("--test", action="store_true", help="Plot test comparison bar chart")
    parser.add_argument("--bin-size", type=int, default=25, help="Window size for averaging train loss (default 25)")
    parser.add_argument("--show-raw", action="store_true", help="Show raw train points as faint scatter")
    args = parser.parse_args()

    plot_all = not any([args.qa, args.qar, args.test])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if plot_all or args.qa:
        plot_curve("QA", args.model_root, out_dir, bin_size=args.bin_size, show_raw=args.show_raw)
    if plot_all or args.qar:
        plot_curve("QAR", args.model_root, out_dir, bin_size=args.bin_size, show_raw=args.show_raw)
    if plot_all or args.test:
        plot_test_comparison(args.test_csv, out_dir)


if __name__ == "__main__":
    main()