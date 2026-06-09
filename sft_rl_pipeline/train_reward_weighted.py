import argparse
import csv
import datetime as dt
import json
import math
import shutil
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from transformers import GPT2TokenizerFast

from reward import score_answer
from rl_utils import DEFAULT_CONFIG, load_config, load_policy, set_seed


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "split", "loss", "reward", "weighted_loss", "batches", "lr"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def current_lr(base_lr, step, warmup):
    if warmup <= 0:
        return base_lr
    return base_lr * min(step / warmup, 1.0)


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def update_global_best(out_root, run_dir, checkpoint_path, best_info):
    global_info_path = out_root / "best_eval.json"
    old = load_json(global_info_path, {})
    old_loss = float(old.get("validation_loss", math.inf))
    if float(best_info["validation_loss"]) >= old_loss:
        return False
    best_info = dict(best_info)
    best_info["run_dir"] = str(run_dir)
    best_info["checkpoint_path"] = str(out_root / "best_RL_QAR.pth")
    shutil.copy2(checkpoint_path, out_root / "best_RL_QAR.pth")
    shutil.copy2(run_dir / "metrics.csv", out_root / "best_metrics.csv")
    dump_json(global_info_path, best_info)
    return True


def _assistant_completion_from_segments(segments, order):
    parts = []
    seen_assistant = False
    for name in order:
        seg = segments.get(name)
        if seg is None:
            continue
        if seen_assistant:
            parts.append(seg.get("text", ""))
        if name == "assistant_label":
            seen_assistant = True
    if parts:
        return "".join(parts)
    return "".join(segments[name].get("text", "") for name in order if name in segments)


def reward_weights(batch, reward_cfg, device):
    vals = []
    for segments, order, target in zip(batch.get("segments", []), batch.get("segment_orders", []), batch["targets"]):
        # Reward-weighted SFT, not online RL: reward is computed on the reference
        # assistant completion and used as a stable sample weight.  With anchored
        # QAR, batch["targets"] only contains generated contents and omits fixed
        # labels such as Reasoning:/Answer:, so we reconstruct the completion.
        completion = _assistant_completion_from_segments(segments, order) or target
        vals.append(score_answer(completion, completion, reward_cfg)["score"])
    rewards = torch.tensor(vals, dtype=torch.float32, device=device)
    min_weight = float(reward_cfg.get("train_weight_min", 0.5))
    max_weight = float(reward_cfg.get("train_weight_max", 1.5))
    weights = min_weight + rewards.clamp(0.0, 1.0) * (max_weight - min_weight)
    return rewards, weights


def main():
    parser = argparse.ArgumentParser(description="Reward-weighted masked DWDSE continuation from QAR/pretrained.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()
    config = load_config(args.config)

    from answer_dataset import make_answer_loader
    from answer_losses import evaluate_answer_loss, get_answer_loss_fn

    set_seed(int(config["training"].get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(config["results"].get("output_dir", "sft_rl_pipeline/modelparameter/RL-QAR"))
    timestamp = dt.datetime.now().strftime("%Y.%m.%d_%H%M%S")
    run_dir_name = timestamp if not args.run_name else f"{timestamp}_{args.run_name}"
    run_dir = out_root / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log = run_dir / "train.log"

    def log(message):
        print(message, flush=True)
        with open(run_log, "a", encoding="utf-8") as f:
            f.write(str(message) + "\n")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    data_dir = Path(config["data"]["data_dir"])
    max_length = int(config["model"].get("max_length", 512))
    min_target_tokens = int(config["model"].get("min_target_tokens", 32))
    drop_overlength = bool(config["model"].get("drop_overlength", True))
    write_reports = bool(config["model"].get("write_load_reports", True))

    train_loader = make_answer_loader(
        data_dir / "train.jsonl",
        tokenizer,
        max_length,
        min_target_tokens,
        int(config["training"].get("batch_size", 1)),
        True,
        int(config["training"].get("num_workers", 0)),
        drop_overlength=drop_overlength,
        write_report=write_reports,
    )
    valid_loader = make_answer_loader(
        data_dir / "validation.jsonl",
        tokenizer,
        max_length,
        min_target_tokens,
        1,
        False,
        int(config["training"].get("num_workers", 0)),
        drop_overlength=drop_overlength,
        write_report=write_reports,
    )

    model, graph, noise, ema, checkpoint = load_policy(config, device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("lr", 1e-6)),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))
    loss_fn = get_answer_loss_fn(noise, graph, train=True)

    dump_json(
        run_dir / "run_info.json",
        {
            "init_checkpoint": checkpoint,
            "objective": "reward-weighted masked DWDSE; reference reward used as sample weight, not online RL",
            "target_format": "Anchored assistant completion = Reasoning: <generated reasoning> / Answer: <generated answer>",
            "data_dir": str(data_dir),
            "config": args.config,
            "device": str(device),
            "run_name": args.run_name,
        },
    )

    steps = int(config["training"].get("steps", 300))
    accum = int(config["training"].get("accum", 8))
    log_freq = int(config["training"].get("log_freq", 10))
    eval_freq = int(config["training"].get("eval_freq", 50))
    eval_batches = int(config["training"].get("eval_batches", 0))
    base_lr = float(config["training"].get("lr", 1e-6))
    warmup = int(config["training"].get("warmup", 50))
    grad_clip = float(config["training"].get("grad_clip", 1.0))
    best = math.inf
    best_run_is_global = False

    pre_loss, pre_batches = evaluate_answer_loss(model, ema, noise, graph, valid_loader, device, eval_batches)
    append_csv(run_dir / "metrics.csv", {"step": 0, "split": "pretrain_validation", "loss": pre_loss, "reward": "", "weighted_loss": "", "batches": pre_batches, "lr": 0.0})
    log(f"run_dir={run_dir}")
    log(f"pretrain_validation_loss={pre_loss:.6f}")

    train_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_weighted = 0.0
    running_reward = 0.0
    running_count = 0

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
            train_mask = batch["train_mask"].to(device, non_blocking=True)
            rewards, weights = reward_weights(batch, config.get("reward", {}), device)
            with autocast(enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                per_sample = loss_fn(model, input_ids, train_mask)
                weighted = (per_sample * weights).mean() / accum
            scaler.scale(weighted).backward()
            running_loss += float(per_sample.mean().detach().item())
            running_weighted += float((per_sample * weights).mean().detach().item())
            running_reward += float(rewards.mean().detach().item())
            running_count += 1

        scaler.unscale_(optimizer)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model.parameters())

        if step % log_freq == 0 or step == 1:
            row = {
                "step": step,
                "split": "train",
                "loss": running_loss / max(1, running_count),
                "reward": running_reward / max(1, running_count),
                "weighted_loss": running_weighted / max(1, running_count),
                "batches": running_count,
                "lr": lr,
            }
            append_csv(run_dir / "metrics.csv", row)
            log(f"step={step} loss={row['loss']:.6f} weighted={row['weighted_loss']:.6f} reward={row['reward']:.3f}")
            running_loss = running_weighted = running_reward = 0.0
            running_count = 0

        if step % eval_freq == 0 or step == steps:
            val, batches = evaluate_answer_loss(model, ema, noise, graph, valid_loader, device, eval_batches)
            append_csv(run_dir / "metrics.csv", {"step": step, "split": "validation", "loss": val, "reward": "", "weighted_loss": "", "batches": batches, "lr": lr})
            log(f"step={step} validation_loss={val:.6f}")
            if val < best:
                best = val
                state = {"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(), "step": step}
                checkpoint_path = run_dir / "best_RL_QAR.pth"
                best_info = {"step": step, "validation_loss": val, "init_checkpoint": checkpoint}
                torch.save(state, checkpoint_path)
                dump_json(run_dir / "best_eval.json", best_info)
                if update_global_best(out_root, run_dir, checkpoint_path, best_info):
                    best_run_is_global = True
                    log(f"new global best validation_loss={val:.6f} at step={step}")

    if best_run_is_global:
        shutil.copy2(run_dir / "metrics.csv", out_root / "best_metrics.csv")
    log(f"done: {run_dir}")


if __name__ == "__main__":
    main()
