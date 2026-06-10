from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import yaml
from transformers import GPT2TokenizerFast

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

for path in (SCRIPT_DIR, REPO_DIR, REPO_DIR / "sft_answer_pipeline"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import graph_lib  # noqa: E402
import noise_lib  # noqa: E402
from model import SEDD  # noqa: E402
from model.ema import ExponentialMovingAverage  # noqa: E402
from answer_dataset import AnswerSegmentDataset  # noqa: E402
from guided_ratio_update import guided_ratio_loss  # noqa: E402

DEFAULT_CONFIG = SCRIPT_DIR / "rl_qra_config.yaml"


def load_config(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_repo_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else REPO_DIR / p


def selected_starts(cfg: Dict, explicit_start: str | None) -> List[str]:
    starts = cfg.get("starts") or {}
    available = list(starts.keys())
    selected = explicit_start or cfg.get("run", {}).get("selected", "all")
    if selected == "all":
        return available
    if isinstance(selected, list):
        names = [str(x) for x in selected]
    else:
        names = [str(selected)]
    unknown = [name for name in names if name not in starts]
    if unknown:
        raise ValueError(f"Unknown start(s): {unknown}. Available starts: {available}")
    return names


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(cfg: Dict) -> torch.device:
    if bool(cfg.get("cpu", False)) or not torch.cuda.is_available():
        return torch.device("cpu")
    cuda_device = cfg.get("run", {}).get("cuda_device", None)
    if cuda_device is None or str(cuda_device).strip() == "":
        return torch.device("cuda")
    return torch.device(f"cuda:{cuda_device}")


def cycle_samples(samples: List[Dict], seed: int) -> Iterable[Dict]:
    rng = random.Random(seed)
    order = list(samples)
    while True:
        rng.shuffle(order)
        for item in order:
            yield item


def load_samples(cfg: Dict, split: str, tokenizer) -> List[Dict]:
    data_dir = resolve_repo_path(cfg["data"]["data_dir"])
    path = data_dir / f"{split}.jsonl"
    ds = AnswerSegmentDataset(
        path,
        tokenizer,
        int(cfg["model"].get("max_length", 1024)),
        min_target_tokens=int(cfg["model"].get("min_target_tokens", 1)),
        drop_overlength=bool(cfg["model"].get("drop_overlength", True)),
        write_report=bool(cfg["model"].get("write_load_reports", False)),
    )
    limit = int(cfg.get("data", {}).get(f"{split}_limit", 0) or 0)
    return ds.samples if limit <= 0 else ds.samples[:limit]


def load_policy(cfg: Dict, device: torch.device, init_checkpoint: str | None = None):
    pretrained = cfg["model"].get("pretrained", "louaaron/sedd-medium")
    model = SEDD.from_pretrained(pretrained).to(device)
    max_length = int(cfg["model"].get("max_length", getattr(model.config.model, "length", 1024)))
    model.config.model.length = max_length
    ema = ExponentialMovingAverage(model.parameters(), decay=float(cfg["training"].get("ema", 0.9999)))

    ckpt_value = init_checkpoint or cfg["model"].get("init_checkpoint", "")
    loaded_from = pretrained
    if ckpt_value:
        ckpt_path = resolve_repo_path(ckpt_value)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state.get("model", state), strict=True)
            if isinstance(state, dict) and "ema" in state:
                try:
                    ema.load_state_dict(state["ema"])
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                except Exception:
                    pass
            loaded_from = str(ckpt_path)
            print(f"Loaded start checkpoint: {ckpt_path}", flush=True)
        else:
            print(f"Warning: init checkpoint not found: {ckpt_path}; using pretrained weights.", flush=True)

    graph = graph_lib.get_graph(model.config, device)
    noise = noise_lib.get_noise(model.config).to(device)
    return model, graph, noise, ema, loaded_from


def append_csv(path: Path, row: Dict) -> None:
    fields = ["step", "loss", "lr", "guided_states", "guided_targets", "pos_logp", "neg_prob"]
    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_checkpoint(path: Path, model, ema, optimizer, cfg: Dict, step: int, metrics: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "step": step,
            "metrics": metrics,
        },
        path,
    )


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def sync_best(run_dir: Path, out_base: Path) -> None:
    copy_if_exists(run_dir / "best_RL_QRA_guided.pth", out_base / "best.pth")
    copy_if_exists(run_dir / "best_eval.json", out_base / "best_eval.json")
    copy_if_exists(run_dir / "metrics.csv", out_base / "best_metrics.csv")
    copy_if_exists(run_dir / "run_info.json", out_base / "best_run_info.json")


def train(cfg: Dict, start_name: str = "QRA", run_name: str = "guided_ratio") -> Path:
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = choose_device(cfg)
    print(f"[{start_name}] device={device}", flush=True)

    start_cfg = (cfg.get("starts") or {}).get(start_name, {})
    output_name = start_cfg.get("output_name", f"rl_{start_name}")
    init_checkpoint = start_cfg.get("init_checkpoint", cfg.get("model", {}).get("init_checkpoint", ""))
    out_root = resolve_repo_path(cfg["output"].get("root_dir", "rl_qra_pipeline/modelparameter"))
    out_base = out_root / output_name
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base / f"{stamp}_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = GPT2TokenizerFast.from_pretrained(cfg["model"].get("tokenizer", "gpt2"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, graph, noise, ema, loaded_from = load_policy(cfg, device, init_checkpoint=init_checkpoint)
    samples = load_samples(cfg, cfg.get("data", {}).get("split", "train"), tokenizer)
    if not samples:
        raise RuntimeError("No training samples loaded.")
    sample_iter = cycle_samples(samples, seed)

    lr = float(cfg["training"].get("lr", 1e-6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(cfg["training"].get("weight_decay", 0.0)))
    batch_size = int(cfg["training"].get("batch_size", 1))
    steps = int(cfg["training"].get("steps", 100))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))
    save_every = int(cfg["training"].get("save_every", 50))
    log_every = int(cfg["training"].get("log_every", 1))

    dump_json(
        out_dir / "run_info.json",
        {
            "start_name": start_name,
            "loaded_from": loaded_from,
            "num_samples": len(samples),
            "device": str(device),
            "output_base": str(out_base),
            "config": cfg,
        },
    )
    metrics_path = out_dir / "metrics.csv"

    best_loss = float("inf")
    for step in range(1, steps + 1):
        model.train(True)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        agg = {"guided_states": 0.0, "guided_targets": 0.0, "pos_logp": 0.0, "neg_prob": 0.0}
        valid = 0
        for _ in range(batch_size):
            sample = next(sample_iter)
            try:
                loss_i, stats_i = guided_ratio_loss(model, graph, noise, tokenizer, sample, cfg, device)
            except Exception as exc:
                print(f"[warn] skip sample {sample.get('id', '')}: {exc}", flush=True)
                continue
            losses.append(loss_i)
            valid += 1
            for k in agg:
                agg[k] += float(stats_i.get(k, 0.0))
        if not losses:
            continue
        loss = torch.stack(losses).mean()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        ema.update(model.parameters())

        row = {"step": step, "loss": float(loss.detach().item()), "lr": lr}
        for k, v in agg.items():
            row[k] = v / max(1, valid)
        if step % log_every == 0:
            append_csv(metrics_path, row)
            print(f"step={step} loss={row['loss']:.4f} pos_logp={row['pos_logp']:.4f} neg_prob={row['neg_prob']:.4f} targets={row['guided_targets']:.1f}", flush=True)
        if row["loss"] < best_loss:
            best_loss = row["loss"]
            save_checkpoint(out_dir / "best_RL_QRA_guided.pth", model, ema, optimizer, cfg, step, row)
            dump_json(
                out_dir / "best_eval.json",
                {
                    "start_name": start_name,
                    "step": step,
                    "loss": row["loss"],
                    "loaded_from": loaded_from,
                    "run_dir": str(out_dir),
                    "checkpoint": str(out_dir / "best_RL_QRA_guided.pth"),
                    "metrics": row,
                },
            )
            sync_best(out_dir, out_base)
        if save_every > 0 and step % save_every == 0:
            save_checkpoint(out_dir / "last_RL_QRA_guided.pth", model, ema, optimizer, cfg, step, row)

    save_checkpoint(out_dir / "last_RL_QRA_guided.pth", model, ema, optimizer, cfg, steps, {"loss": best_loss})
    copy_if_exists(out_dir / "last_RL_QRA_guided.pth", out_base / "last.pth")
    sync_best(out_dir, out_base)
    print(f"Done. Outputs: {out_dir}", flush=True)
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-name", type=str, default="guided_ratio")
    parser.add_argument("--start", type=str, default=None, help="Start point to train: pretrain, QRA, or all.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    failures = []
    for start_name in selected_starts(cfg, args.start):
        try:
            train(cfg, start_name=start_name, run_name=args.run_name)
        except Exception as exc:
            failures.append((start_name, repr(exc)))
            raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if failures:
        raise SystemExit(f"Run(s) failed: {failures}")


if __name__ == "__main__":
    main()
