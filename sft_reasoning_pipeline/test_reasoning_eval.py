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
ANSWER_PIPELINE_DIR = REPO_DIR / "sft_answer_pipeline"
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(ANSWER_PIPELINE_DIR))

DEFAULT_CONFIG = SCRIPT_DIR / "reasoning_config.yaml"


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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


def selected_runs(config):
    selected = config.get("run", {}).get("selected", "QRA")
    if selected == "all":
        return list(config.get("runs", {}).keys()) or ["QRA"]
    if isinstance(selected, list):
        return selected
    return [selected]


def make_loader(config, dataset_name, split, batch_size):
    from answer_dataset import make_answer_loader

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    data_root = Path(config["data"].get("output_dir", SCRIPT_DIR / "data"))
    return make_answer_loader(
        data_root / dataset_name / f"{split}.jsonl",
        tokenizer,
        int(config["model"].get("max_length", 512)),
        int(config["model"].get("min_target_tokens_by_mode", {}).get(dataset_name, config["model"].get("min_target_tokens", 1))),
        batch_size,
        False,
        int(config.get("eval", {}).get("num_workers", 0)),
        drop_overlength=bool(config["model"].get("drop_overlength", True)),
        write_report=bool(config["model"].get("write_load_reports", True)),
    )


def load_pretrained(config, device):
    import graph_lib
    import noise_lib
    from model import SEDD
    from model.ema import ExponentialMovingAverage

    model_name = config["model"].get("pretrained", "louaaron/sedd-medium")
    model = SEDD.from_pretrained(model_name).to(device)
    model.config.model.length = int(config["model"].get("max_length", model.config.model.length))
    graph = graph_lib.get_graph(model.config, device)
    noise = noise_lib.get_noise(model.config).to(device)
    ema = ExponentialMovingAverage(model.parameters(), decay=float(config["training"].get("ema", 0.9999)))
    return model, graph, noise, ema, model_name


def load_best(config, run_name, device):
    import graph_lib
    import noise_lib
    from model import SEDD
    from model.ema import ExponentialMovingAverage

    model_dir = Path(config["results"].get("output_dir", SCRIPT_DIR / "modelparameter")) / run_name
    ckpt_path = model_dir / "best.pth"
    if not ckpt_path.exists():
        print(f"Skip {run_name}-best: {ckpt_path} not found")
        return None
    model = SEDD.from_pretrained(config["model"].get("pretrained", "louaaron/sedd-medium")).to(device)
    model.config.model.length = int(config["model"].get("max_length", model.config.model.length))
    ema = ExponentialMovingAverage(model.parameters(), decay=float(config["training"].get("ema", 0.9999)))
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    ema.load_state_dict(state["ema"])
    graph = graph_lib.get_graph(model.config, device)
    noise = noise_lib.get_noise(model.config).to(device)
    return model, graph, noise, ema, str(ckpt_path)


def eval_model(config, model_name, model_tuple, datasets, device):
    from answer_losses import evaluate_answer_loss

    model, graph, noise, ema, checkpoint = model_tuple
    batch_size = int(config.get("eval", {}).get("batch_size", 1))
    eval_batches = int(config.get("eval", {}).get("eval_batches", 0))
    rows = []
    for dataset_name in datasets:
        loader = make_loader(config, dataset_name, "test", batch_size)
        loss, batches = evaluate_answer_loss(model, ema, noise, graph, loader, device, eval_batches=eval_batches)
        row = {"dataset": dataset_name, "model": model_name, "loss": loss, "batches": batches, "checkpoint": checkpoint}
        rows.append(row)
        print(row, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate reasoning-conditioned QRA SFT checkpoints on test split.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = [config.get("runs", {}).get(run, {}).get("dataset", run) for run in selected_runs(config)] or ["QRA"]

    rows = []
    model_tuple = None
    try:
        model_tuple = load_pretrained(config, device)
        rows.extend(eval_model(config, "pretrained", model_tuple, datasets, device))
    finally:
        if model_tuple is not None:
            del model_tuple
        cleanup()

    for run_name in selected_runs(config):
        model_tuple = load_best(config, run_name, device)
        if model_tuple is None:
            continue
        try:
            rows.extend(eval_model(config, f"{run_name}-best", model_tuple, datasets, device))
        finally:
            del model_tuple
            cleanup()

    output_root = Path(config["results"].get("output_dir", SCRIPT_DIR / "modelparameter"))
    result_dir = output_root / "test_result"
    write_csv(rows, result_dir / "test_results.csv")
    dump_json(rows, result_dir / "test_results.json")
    print(f"Wrote {result_dir / 'test_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
