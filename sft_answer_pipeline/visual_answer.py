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

    Example with bin_size=25:
      steps 1..25   -> x=25,  loss=mean(losses)
      steps 26..50  -> x=50,  loss=mean(losses)
      steps 76..100 -> x=100, loss=mean(losses)

    If the training log only contains one point per 25 steps,
    the mean is just that one point.
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


def plot_curve(root, name, out_dir, bin_size=25, show_raw=False):
    import matplotlib.pyplot as plt

    path = best_metrics_path(root, name)
    if not path.exists():
        print(f"Skip {name}: {path} not found")
        return

    rows = []
    for row in read_csv(path):
        try:
            rows.append(
                {
                    "step": int(row["step"]),
                    "split": row["split"],
                    "loss": float(row["loss"]),
                }
            )
        except (KeyError, ValueError):
            pass

    plt.figure(figsize=(10, 6))

    train_raw = [row for row in rows if row["split"] == "train"]
    train_mean = aggregate_by_step_mean(rows, "train", bin_size)

    if show_raw and train_raw:
        train_raw = sorted(train_raw, key=lambda r: r["step"])
        plt.scatter(
            [r["step"] for r in train_raw],
            [r["loss"] for r in train_raw],
            s=8,
            alpha=0.20,
            label=f"train raw ({len(train_raw)} points)",
        )

    if train_mean:
        plt.plot(
            [r["step"] for r in train_mean],
            [r["loss"] for r in train_mean],
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=f"train mean per {bin_size} steps ({len(train_mean)} points)",
        )

    validation = [row for row in rows if row["split"] == "validation"]
    if validation:
        validation = sorted(validation, key=lambda r: r["step"])
        plt.plot(
            [r["step"] for r in validation],
            [r["loss"] for r in validation],
            marker="o",
            markersize=5,
            linewidth=2.0,
            label=f"validation ({len(validation)} points)",
        )

    baseline = [row for row in rows if row["split"] == "pretrain_validation"]
    if baseline:
        plt.axhline(
            baseline[0]["loss"],
            linestyle="--",
            linewidth=1.8,
            label=f"pretrained validation = {baseline[0]['loss']:.4f}",
        )

    plt.title(f"{name} Answer SFT Loss")
    plt.xlabel("step")
    plt.ylabel("answer-token score entropy loss")
    plt.grid(alpha=0.25)
    plt.legend()

    out = Path(out_dir) / f"{name.lower()}_answer_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()

    print(f"Read metrics from {path}")
    print(f"Saved {out}")

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
            f"loss={train_mean[-1]['loss']:.4f}, "
            f"points_in_last_window={train_mean[-1]['count']}"
        )


def plot_test(root, out_dir):
    import matplotlib.pyplot as plt

    path = Path(root) / "test_result" / "test_results.csv"
    if not path.exists():
        print(f"Skip test: {path} not found")
        return

    rows = []
    for row in read_csv(path):
        try:
            rows.append(
                {
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "loss": float(row["loss"]),
                }
            )
        except (KeyError, ValueError):
            pass

    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    models = list(dict.fromkeys(row["model"] for row in rows))
    values = {(row["dataset"], row["model"]): row["loss"] for row in rows}

    width = 0.8 / max(1, len(models))
    x = list(range(len(datasets)))

    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        offsets = [pos - 0.4 + width / 2 + i * width for pos in x]
        plt.bar(
            offsets,
            [values.get((dataset, model), 0) for dataset in datasets],
            width=width,
            label=model,
        )

    plt.xticks(x, datasets)
    plt.ylabel("answer-token score entropy loss")
    plt.title("Answer SFT Test Loss")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()

    out = Path(out_dir) / "answer_test_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()

    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot answer-conditioned SFT curves and test comparison."
    )
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--qa", action="store_true")
    parser.add_argument("--qar", action="store_true")
    parser.add_argument("--test", action="store_true")

    parser.add_argument(
        "--bin-size",
        type=int,
        default=25,
        help="Average train loss within fixed step windows. Default: 25.",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Also show raw train points as faint scatter.",
    )

    args = parser.parse_args()

    plot_all = not any([args.qa, args.qar, args.test])

    if plot_all or args.qa:
        plot_curve(args.root, "QA", args.out_dir, bin_size=args.bin_size, show_raw=args.show_raw)
    if plot_all or args.qar:
        plot_curve(args.root, "QAR", args.out_dir, bin_size=args.bin_size, show_raw=args.show_raw)
    if plot_all or args.test:
        plot_test(args.root, args.out_dir)


if __name__ == "__main__":
    main()