import argparse
import csv
from pathlib import Path

# 获取当前脚本所在目录（假设 visual.py 位于 sft_pipeline/ 下）
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = SCRIPT_DIR / "modelparameter"
DEFAULT_TEST_CSV = SCRIPT_DIR / "reports" / "test_results.csv"
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
    eval_rows = [row for row in rows if row.get("kind") == "evaluation" and row.get("valid_for_best") != "False"]
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
        raise ValueError("No valid rows in test CSV. Need columns: dataset, model, loss")

    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    models = list(dict.fromkeys(row["model"] for row in rows))
    values = {(row["dataset"], row["model"]): row["loss"] for row in rows}
    x = list(range(len(datasets)))
    width = 0.8 / max(1, len(models))

    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        offsets = [pos - 0.4 + width/2 + i*width for pos in x]
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
    parser = argparse.ArgumentParser(description="Generate all available plots (QA, QAR, test comparison) by default.")
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT), help="Root dir containing QA/, QAR/ subdirs with metrics.csv")
    parser.add_argument("--test-csv", default=str(DEFAULT_TEST_CSV), help="CSV for test comparison (columns: dataset,model,loss)")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output folder for images")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. QA training curve
    qa_metrics = Path(args.model_root) / "QA" / "metrics.csv"
    if qa_metrics.exists():
        plot_training_curve(qa_metrics, out_dir / "qa_training_curve.png", "QA SFT Training Curve")
    else:
        print(f"Skip QA curve: {qa_metrics} not found")

    # 2. QAR training curve
    qar_metrics = Path(args.model_root) / "QAR" / "metrics.csv"
    if qar_metrics.exists():
        plot_training_curve(qar_metrics, out_dir / "qar_training_curve.png", "QAR SFT Training Curve")
    else:
        print(f"Skip QAR curve: {qar_metrics} not found")

    # 3. Test comparison
    test_csv = Path(args.test_csv)
    if test_csv.exists():
        plot_test_comparison(test_csv, out_dir / "test_comparison.png")
    else:
        print(f"Skip test comparison: {test_csv} not found")

if __name__ == "__main__":
    main()