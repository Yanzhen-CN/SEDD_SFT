import argparse
import csv
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = SCRIPT_DIR / "modelparameter"
DEFAULT_TEST_CSV = DEFAULT_MODEL_ROOT / "test_result" / "test_results.csv"
DEFAULT_OUT_DIR = SCRIPT_DIR / "visualization"


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
        plt.scatter([best_eval["step"]], [best_eval["loss"]], s=80, label=f"best validation @ {best_eval['step']}")

    plt.xlabel("step")
    plt.ylabel("score entropy loss")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Saved: {output_path}")


def plot_qa_curve(model_root, out_dir):
    metrics = find_best_metrics(Path(model_root) / "QA")
    if metrics.exists():
        plot_training_curve(metrics, Path(out_dir) / "qa_training_curve.png", "QA SFT Training Curve")
    else:
        print(f"Skip QA curve: {metrics} not found")


def plot_qar_curve(model_root, out_dir):
    metrics = find_best_metrics(Path(model_root) / "QAR")
    if metrics.exists():
        plot_training_curve(metrics, Path(out_dir) / "qar_training_curve.png", "QAR SFT Training Curve")
    else:
        print(f"Skip QAR curve: {metrics} not found")


def find_best_metrics(run_root):
    run_root = Path(run_root)
    direct = run_root / "best_metrics.csv"
    if direct.exists():
        return direct

    best_eval = run_root / "best_eval.json"
    if best_eval.exists():
        with open(best_eval, "r", encoding="utf-8") as f:
            run_instance = json.load(f).get("run_instance")
        if run_instance:
            return run_root / run_instance / "metrics.csv"

    return direct


def plot_test_comparison(test_csv, output_path):
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
        raise ValueError("No valid rows in test CSV. Need columns: dataset,model,loss")

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate QA, QAR, and final test plots from saved result files.")
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--test-csv", default=str(DEFAULT_TEST_CSV))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--qa", action="store_true", help="Plot QA best training curve.")
    parser.add_argument("--qar", action="store_true", help="Plot QAR best training curve.")
    parser.add_argument("--test", action="store_true", help="Plot final test comparison.")
    args = parser.parse_args()

    plot_all = not any([args.qa, args.qar, args.test])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if plot_all or args.qa:
        plot_qa_curve(args.model_root, out_dir)
    if plot_all or args.qar:
        plot_qar_curve(args.model_root, out_dir)
    if plot_all or args.test:
        test_csv = Path(args.test_csv)
        if test_csv.exists():
            plot_test_comparison(test_csv, out_dir / "test_comparison.png")
        else:
            print(f"Skip test comparison: {test_csv} not found")


if __name__ == "__main__":
    main()
