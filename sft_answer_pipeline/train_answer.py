import argparse
import csv
import datetime as dt
import json
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from transformers import GPT2TokenizerFast


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_DIR))

DEFAULT_CONFIG = SCRIPT_DIR / "answer_config.yaml"


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def append_csv(row, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "split", "loss", "batches", "lr"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def selected_runs(config, explicit_run):
    if explicit_run:
        selected = explicit_run
    else:
        selected = config.get("run", {}).get("selected", "QA")
    if selected == "all":
        return ["QA", "QAR"]
    if isinstance(selected, list):
        return selected
    return [selected]


def build_loaders(config, run_name):
    from answer_dataset import make_answer_loader

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    data_root = Path(config["data"].get("output_dir", SCRIPT_DIR / "data"))
    dataset_name = config["runs"][run_name].get("dataset", run_name)
    data_dir = data_root / dataset_name
    max_length = int(config["model"].get("max_length", 512))
    min_target_tokens = int(config["model"].get("min_target_tokens", 32))
    train_cfg = config["training"]

    train_loader = make_answer_loader(
        data_dir / "train.jsonl",
        tokenizer,
        max_length,
        min_target_tokens,
        int(train_cfg.get("batch_size", 1)),
        True,
        int(train_cfg.get("num_workers", 0)),
    )
    valid_loader = make_answer_loader(
        data_dir / "validation.jsonl",
        tokenizer,
        max_length,
        min_target_tokens,
        int(config.get("eval", {}).get("batch_size", train_cfg.get("batch_size", 1))),
        False,
        int(config.get("eval", {}).get("num_workers", 0)),
    )
    return tokenizer, train_loader, valid_loader, str(data_dir)


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


def current_lr(base_lr, step, warmup):
    if warmup <= 0:
        return base_lr
    return base_lr * min(step / warmup, 1.0)


def save_checkpoint(path, model, ema, optimizer, step, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "meta": meta,
    }, path)


def sync_global_best(run_dir, global_dir):
    global_dir = Path(global_dir)
    global_dir.mkdir(parents=True, exist_ok=True)
    for name in ["best.pth", "best_eval.json", "metrics.csv", "metrics.jsonl", "run_info.json"]:
        src = Path(run_dir) / name
        if not src.exists():
            continue
        dst_name = name if name.startswith("best") else f"best_{name}"
        shutil.copyfile(src, global_dir / dst_name)


def train_one(config, run_name):
    from answer_losses import evaluate_answer_loss, get_answer_loss_fn

    seed = int(config["training"].get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and config["runs"][run_name].get("cuda_visible_devices"):
        print("CUDA_VISIBLE_DEVICES should be set by run_answer_sft.py before this process starts.")

    tokenizer, train_loader, valid_loader, data_dir = build_loaders(config, run_name)
    model, graph, noise, ema, pretrained_name = load_pretrained(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("lr", 3e-6)),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))
    loss_fn = get_answer_loss_fn(noise, graph, train=True)

    output_root = Path(config["results"].get("output_dir", SCRIPT_DIR / "modelparameter"))
    run_instance = dt.datetime.now().strftime("%Y.%m.%d_%H%M%S")
    run_dir = output_root / run_name / run_instance
    global_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    run_info = {
        "run_name": run_name,
        "run_instance": run_instance,
        "data_dir": data_dir,
        "pretrained": pretrained_name,
        "max_length": int(config["model"].get("max_length", 512)),
        "loss": "score entropy on target tokens only; prompt tokens are fixed conditioning context",
        "device": str(device),
    }
    dump_json(run_info, run_dir / "run_info.json")

    steps = int(config["training"].get("steps", 1500))
    accum = int(config["training"].get("accum", 1))
    log_freq = int(config["training"].get("log_freq", 25))
    eval_freq = int(config["training"].get("eval_freq", 100))
    save_freq = int(config["training"].get("save_freq", 500))
    eval_batches = int(config["training"].get("eval_batches", 0))
    base_lr = float(config["training"].get("lr", 3e-6))
    warmup = int(config["training"].get("warmup", 0))
    grad_clip = float(config["training"].get("grad_clip", 1.0))
    best_loss = math.inf

    pretrain_loss, pretrain_batches = evaluate_answer_loss(
        model, ema, noise, graph, valid_loader, device, eval_batches=eval_batches
    )
    pretrain_record = {"step": 0, "split": "pretrain_validation", "loss": pretrain_loss, "batches": pretrain_batches, "lr": 0.0}
    append_csv(pretrain_record, run_dir / "metrics.csv")
    append_jsonl(pretrain_record, run_dir / "metrics.jsonl")
    dump_json(pretrain_record, run_dir / "pretrain_eval.json")
    print(f"[{run_name}] pretrain_validation_loss={pretrain_loss:.6f} batches={pretrain_batches}")

    train_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    running = 0.0
    running_count = 0
    model.train()

    for step in range(1, steps + 1):
        lr = current_lr(base_lr, step, warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr

        for _ in range(accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            answer_mask = batch["answer_mask"].to(device, non_blocking=True)
            with autocast(enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                loss = loss_fn(model, input_ids, answer_mask).mean() / accum
            scaler.scale(loss).backward()
            running += float(loss.detach().item()) * accum
            running_count += 1

        scaler.unscale_(optimizer)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model.parameters())

        if step % log_freq == 0 or step == 1:
            train_loss = running / max(1, running_count)
            row = {"step": step, "split": "train", "loss": train_loss, "batches": running_count, "lr": lr}
            append_csv(row, run_dir / "metrics.csv")
            append_jsonl(row, run_dir / "metrics.jsonl")
            print(f"[{run_name}] step={step} train_loss={train_loss:.6f} lr={lr:.3e}")
            running = 0.0
            running_count = 0

        if step % eval_freq == 0 or step == steps:
            valid_loss, used_batches = evaluate_answer_loss(
                model, ema, noise, graph, valid_loader, device, eval_batches=eval_batches
            )
            row = {"step": step, "split": "validation", "loss": valid_loss, "batches": used_batches, "lr": lr}
            append_csv(row, run_dir / "metrics.csv")
            append_jsonl(row, run_dir / "metrics.jsonl")
            print(f"[{run_name}] step={step} validation_loss={valid_loss:.6f}")

            if valid_loss < best_loss:
                best_loss = valid_loss
                meta = {**run_info, "step": step, "validation_loss": valid_loss, "pretrain_validation_loss": pretrain_loss}
                save_checkpoint(run_dir / "best.pth", model, ema, optimizer, step, meta)
                dump_json(meta, run_dir / "best_eval.json")
                global_best_path = global_dir / "best_eval.json"
                old_global = math.inf
                if global_best_path.exists():
                    old_global = float(json.loads(global_best_path.read_text(encoding="utf-8")).get("validation_loss", math.inf))
                if valid_loss < old_global:
                    sync_global_best(run_dir, global_dir)
                    append_jsonl(meta, global_dir / "improvement_log.jsonl")
                    print(f"[{run_name}] new global best validation_loss={valid_loss:.6f} step={step}")

        if save_freq > 0 and step % save_freq == 0:
            save_checkpoint(run_dir / f"step_{step}.pth", model, ema, optimizer, step, run_info)

    global_best_path = global_dir / "best_eval.json"
    if global_best_path.exists():
        global_best = json.loads(global_best_path.read_text(encoding="utf-8"))
        if global_best.get("run_instance") == run_instance:
            sync_global_best(run_dir, global_dir)

    print(f"[{run_name}] done. run_dir={run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train answer-conditioned SEDD SFT.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run", default=None, choices=["QA", "QAR"])
    args = parser.parse_args()
    config = load_config(args.config)
    for run_name in selected_runs(config, args.run):
        train_one(config, run_name)


if __name__ == "__main__":
    main()
