import argparse
import csv
from pathlib import Path


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value}")


def read_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def read_metrics(path):
    rows = []
    for row in read_csv(path):
        try:
            row["step"] = int(row["step"])
            row["loss"] = float(row["loss"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append(row)
    return rows


def split_series(rows, kind):
    selected = [row for row in rows if row.get("kind") == kind]
    return [row["step"] for row in selected], [row["loss"] for row in selected]


def find_best_eval(rows):
    eval_rows = [
        row for row in rows
        if row.get("kind") == "evaluation" and row.get("valid_for_best") != "False"
    ]
    if not eval_rows:
        return None
    return min(eval_rows, key=lambda row: row["loss"])


def plot_training_curve(metrics_csv, output_path, title):
    import matplotlib.pyplot as plt

    rows = read_metrics(metrics_csv)
    train_x, train_y = split_series(rows, "training")
    eval_x, eval_y = split_series(rows, "evaluation")
    _, pretrain_y = split_series(rows, "pretrain_evaluation")
    best_eval = find_best_eval(rows)

    plt.figure(figsize=(10, 6))
    if train_x:
        plt.plot(train_x, train_y, label="train loss", linewidth=1.4, alpha=0.72)
    if eval_x:
        plt.plot(eval_x, eval_y, marker="o", label="validation loss", linewidth=1.8)
    if pretrain_y:
        plt.axhline(pretrain_y[0], linestyle="--", linewidth=1.2, label="pretrain baseline")
    if best_eval:
        plt.scatter(
            [best_eval["step"]],
            [best_eval["loss"]],
            s=80,
            label=f"best validation @ {best_eval['step']}",
        )

    plt.xlabel("step")
    plt.ylabel("score entropy loss")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180)
    plt.close()
    print(f"Wrote {output}")


def plot_qa_curve(metrics_csv, output_dir):
    plot_training_curve(
        metrics_csv,
        Path(output_dir) / "qa_training_curve.png",
        "QA SFT Training Curve",
    )


def plot_qar_curve(metrics_csv, output_dir):
    plot_training_curve(
        metrics_csv,
        Path(output_dir) / "qar_training_curve.png",
        "QAR SFT Training Curve",
    )


def plot_test_comparison(test_csv, output_dir):
    import matplotlib.pyplot as plt

    rows = []
    for row in read_csv(test_csv):
        try:
            rows.append({
                "dataset": row["dataset"],
                "model": row["model"],
                "loss": float(row["loss"]),
            })
        except (KeyError, TypeError, ValueError):
            continue

    if not rows:
        raise ValueError("No valid rows found in test CSV. Expected columns: dataset,model,loss")

    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    models = list(dict.fromkeys(row["model"] for row in rows))
    values = {(row["dataset"], row["model"]): row["loss"] for row in rows}

    x = list(range(len(datasets)))
    width = 0.8 / max(1, len(models))

    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        offsets = [pos - 0.4 + width / 2 + i * width for pos in x]
        y = [values.get((dataset, model), 0) for dataset in datasets]
        plt.bar(offsets, y, width=width, label=model)

    plt.xticks(x, datasets)
    plt.xlabel("test split")
    plt.ylabel("score entropy loss")
    plt.title("Final Test Loss Comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output = Path(output_dir) / "test_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180)
    plt.close()
    print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot SEDD SFT training curves and final test comparison.")
    parser.add_argument("--plot-qa", type=str_to_bool, default=False)
    parser.add_argument("--plot-qar", type=str_to_bool, default=False)
    parser.add_argument("--plot-test", type=str_to_bool, default=False)
    parser.add_argument("--qa-metrics", default=None, help="QA run metrics.csv.")
    parser.add_argument("--qar-metrics", default=None, help="QAR run metrics.csv.")
    parser.add_argument("--test-csv", default=None, help="Final test CSV with columns: dataset,model,loss.")
    parser.add_argument("--out-dir", default="sft_pipeline/reports/figures")
    args = parser.parse_args()

    if args.plot_qa:
        if not args.qa_metrics:
            raise ValueError("--qa-metrics is required when --plot-qa true")
        plot_qa_curve(args.qa_metrics, args.out_dir)

    if args.plot_qar:
        if not args.qar_metrics:
            raise ValueError("--qar-metrics is required when --plot-qar true")
        plot_qar_curve(args.qar_metrics, args.out_dir)

    if args.plot_test:
        if not args.test_csv:
            raise ValueError("--test-csv is required when --plot-test true")
        plot_test_comparison(args.test_csv, args.out_dir)

    if not any([args.plot_qa, args.plot_qar, args.plot_test]):
        raise ValueError("Nothing to plot. Set one of --plot-qa true, --plot-qar true, --plot-test true.")


if __name__ == "__main__":
    main()
