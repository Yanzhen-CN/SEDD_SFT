import argparse
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR / "modelparameter"
DEFAULT_OUT = SCRIPT_DIR / "visualization"


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def best_metrics_path(root, name):
    direct = Path(root) / name / "best_metrics.csv"
    if direct.exists():
        return direct
    return Path(root) / name / "metrics.csv"


def plot_curve(root, name, out_dir):
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
    for split, label in [("train", "train"), ("validation", "validation")]:
        selected = [row for row in rows if row["split"] == split]
        if selected:
            plt.plot([r["step"] for r in selected], [r["loss"] for r in selected], marker="o", label=label)
    baseline = [row for row in rows if row["split"] == "pretrain_validation"]
    if baseline:
        plt.axhline(baseline[0]["loss"], linestyle="--", label="pretrained validation")
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
    parser = argparse.ArgumentParser(description="Plot answer-conditioned SFT curves and test comparison.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--qa", action="store_true")
    parser.add_argument("--qar", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    plot_all = not any([args.qa, args.qar, args.test])
    if plot_all or args.qa:
        plot_curve(args.root, "QA", args.out_dir)
    if plot_all or args.qar:
        plot_curve(args.root, "QAR", args.out_dir)
    if plot_all or args.test:
        plot_test(args.root, args.out_dir)


if __name__ == "__main__":
    main()
