import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import GPT2TokenizerFast


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
for path in (REPO_DIR, REPO_DIR / "sft_answer_pipeline"):
    sys.path.insert(0, str(path))

DEFAULT_CONFIG = SCRIPT_DIR / "rl_qra_config.yaml"


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def repo_path(path):
    p = Path(path)
    return p if p.is_absolute() else REPO_DIR / p


def dump_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "model", "loss", "batches", "checkpoint"])
        writer.writeheader()
        writer.writerows(rows)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_loader(config, dataset_name, split):
    from answer_dataset import make_answer_loader

    tokenizer = GPT2TokenizerFast.from_pretrained(config["model"].get("tokenizer", "gpt2"))
    tokenizer.pad_token = tokenizer.eos_token
    data_root = repo_path(Path(config["data"]["data_dir"]).parent)
    return make_answer_loader(
        data_root / dataset_name / f"{split}.jsonl",
        tokenizer,
        int(config["model"].get("max_length", 1024)),
        int(config["model"].get("min_target_tokens", 1)),
        int(config.get("eval", {}).get("batch_size", 1)),
        False,
        int(config.get("eval", {}).get("num_workers", 0)),
        drop_overlength=bool(config["model"].get("drop_overlength", True)),
        write_report=bool(config["model"].get("write_load_reports", False)),
    )


def load_model(config, checkpoint, device):
    import graph_lib
    import noise_lib
    from model import SEDD
    from model.ema import ExponentialMovingAverage

    model = SEDD.from_pretrained(config["model"].get("pretrained", "louaaron/sedd-medium")).to(device)
    model.config.model.length = int(config["model"].get("max_length", model.config.model.length))
    ema = ExponentialMovingAverage(model.parameters(), decay=float(config["training"].get("ema", 0.9999)))
    ckpt_path = repo_path(checkpoint)
    if not ckpt_path.exists():
        print(f"Skip {ckpt_path}: not found")
        return None
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state.get("model", state), strict=True)
    if isinstance(state, dict) and "ema" in state:
        ema.load_state_dict(state["ema"])
    graph = graph_lib.get_graph(model.config, device)
    noise = noise_lib.get_noise(model.config).to(device)
    return model, graph, noise, ema, str(ckpt_path)


def eval_model(config, model_name, model_tuple, datasets, device):
    from answer_losses import evaluate_answer_loss

    model, graph, noise, ema, checkpoint = model_tuple
    eval_batches = int(config.get("eval", {}).get("eval_batches", 0))
    rows = []
    for dataset_name in datasets:
        loader = make_loader(config, dataset_name, "test")
        loss, batches = evaluate_answer_loss(model, ema, noise, graph, loader, device, eval_batches=eval_batches)
        row = {"dataset": dataset_name, "model": model_name, "loss": loss, "batches": batches, "checkpoint": checkpoint}
        rows.append(row)
        print(row, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL-QRA checkpoints on QRA test split.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and not bool(config.get("cpu", False)) else "cpu")
    datasets = list(config.get("eval", {}).get("datasets", ["QRA"]))
    rows = []

    for model_name, model_cfg in (config.get("compare_models") or {}).items():
        checkpoint = model_cfg.get("checkpoint") if isinstance(model_cfg, dict) else model_cfg
        model_tuple = load_model(config, checkpoint, device)
        if model_tuple is None:
            continue
        try:
            rows.extend(eval_model(config, model_name, model_tuple, datasets, device))
        finally:
            del model_tuple
            cleanup()

    out_dir = repo_path(config.get("eval", {}).get("output_dir", "rl_qra_pipeline/test_result"))
    write_csv(rows, out_dir / "test_results.csv")
    dump_json(rows, out_dir / "test_results.json")

    output_root = repo_path(config["output"].get("root_dir", "rl_qra_pipeline/modelparameter"))
    for model_name in ("rl_pretrain", "rl_QRA"):
        model_rows = [row for row in rows if row["model"] == model_name]
        if model_rows:
            dump_json(model_rows, output_root / model_name / "best_test_result.json")
            write_csv(model_rows, output_root / model_name / "best_test_result.csv")
    print(f"Wrote {out_dir / 'test_results.csv'}")


if __name__ == "__main__":
    main()
