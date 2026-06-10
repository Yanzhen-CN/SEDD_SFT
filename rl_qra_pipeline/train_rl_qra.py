from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any

import numpy as np
import torch
import yaml
from transformers import GPT2TokenizerFast

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
for p in (REPO_DIR, REPO_DIR / "sft_answer_pipeline", REPO_DIR / "sft_rl_pipeline", SCRIPT_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

import graph_lib  # noqa: E402
import noise_lib  # noqa: E402
from model import SEDD  # noqa: E402
from model.ema import ExponentialMovingAverage  # noqa: E402
from answer_dataset import AnswerSegmentDataset  # noqa: E402

from guided_ratio_update import guided_ratio_loss  # noqa: E402

DEFAULT_CONFIG = SCRIPT_DIR / "rl_qra_config.yaml"

METRIC_FIELDS = [
    "step",
    "loss",
    "lr",
    "guided_states",
    "guided_targets",
    "rrpi_loss",
    "target_logp",
    "target_prob",
    "model_reward",
    "best_reward",
    "reward_gap",
    "candidate_entropy",
    # Compatibility with old scripts.
    "pos_logp",
    "neg_prob",
]


def load_config(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_repo_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else REPO_DIR / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deep_update(base: Dict, updates: Dict) -> Dict:
    out = copy.deepcopy(base)
    for k, v in (updates or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def select_start_config(raw_cfg: Dict, start_name: str | None) -> Tuple[Dict, str]:
    """Return a flat runtime config for one selected start."""
    cfg = copy.deepcopy(raw_cfg)
    starts = cfg.get("starts") or {}
    selected = start_name or cfg.get("run", {}).get("selected") or cfg.get("start") or "QRA"

    if starts:
        if selected not in starts:
            raise KeyError(f"Unknown --start {selected!r}. Available starts: {sorted(starts.keys())}")
        start_cfg = starts[selected] or {}

        cfg.setdefault("model", {})
        cfg.setdefault("data", {})
        cfg.setdefault("output", {})

        if start_cfg.get("init_checkpoint") is not None:
            cfg["model"]["init_checkpoint"] = start_cfg["init_checkpoint"]
        if start_cfg.get("pretrained") is not None:
            cfg["model"]["pretrained"] = start_cfg["pretrained"]
        if start_cfg.get("data_dir") is not None:
            cfg["data"]["data_dir"] = start_cfg["data_dir"]
        if start_cfg.get("split") is not None:
            cfg["data"]["split"] = start_cfg["split"]
        if start_cfg.get("output_dir") is not None:
            cfg["output"]["dir"] = start_cfg["output_dir"]
        if start_cfg.get("output_name") is not None:
            cfg["output"]["name"] = start_cfg["output_name"]

        cfg = deep_update(cfg, start_cfg.get("overrides", {}))
    else:
        selected = selected or "default"

    return cfg, selected


def cycle_samples(samples: List[Dict], seed: int) -> Iterable[Dict]:
    rng = random.Random(seed)
    order = list(samples)
    while True:
        rng.shuffle(order)
        for item in order:
            yield item


def load_samples(cfg: Dict, split: str, tokenizer) -> List[Dict]:
    data_dir_value = cfg.get("data", {}).get("data_dir")
    if not data_dir_value:
        raise ValueError("Missing data.data_dir after resolving start config.")
    data_dir = resolve_repo_path(data_dir_value)
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")
    ds = AnswerSegmentDataset(
        path,
        tokenizer,
        int(cfg["model"].get("max_length", 1024)),
        min_target_tokens=int(cfg["model"].get("min_target_tokens", 1)),
        drop_overlength=bool(cfg["model"].get("drop_overlength", True)),
        write_report=bool(cfg["model"].get("write_load_reports", False)),
    )
    limit = int(cfg.get("data", {}).get(f"{split}_limit", 0) or cfg.get("data", {}).get("train_limit", 0) or 0)
    return ds.samples if limit <= 0 else ds.samples[:limit]


def load_policy(cfg: Dict, device: torch.device):
    pretrained = cfg.get("model", {}).get("pretrained", "louaaron/sedd-medium")
    model = SEDD.from_pretrained(pretrained).to(device)
    max_length = int(cfg["model"].get("max_length", getattr(model.config.model, "length", 1024)))
    model.config.model.length = max_length
    ema = ExponentialMovingAverage(model.parameters(), decay=float(cfg["training"].get("ema", 0.9999)))

    ckpt_value = cfg.get("model", {}).get("init_checkpoint", "")
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
                except Exception as exc:
                    print(f"[warn] failed to load EMA from checkpoint: {exc}", flush=True)
            loaded_from = str(ckpt_path)
            print(f"Loaded start checkpoint: {ckpt_path}", flush=True)
        else:
            raise FileNotFoundError(
                f"init checkpoint not found: {ckpt_path}\n"
                "Fix starts.<name>.init_checkpoint in rl_qra_config.yaml, or pass --start to select another start."
            )

    graph = graph_lib.get_graph(model.config, device)
    noise = noise_lib.get_noise(model.config).to(device)
    return model, graph, noise, ema, loaded_from


def append_csv(path: Path, row: Dict[str, Any]) -> None:
    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in METRIC_FIELDS})


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


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


def sync_checkpoint(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def print_debug_record(step: int, record: Dict[str, Any], max_rows: int = 8) -> None:
    print(f"\n[RRPI DEBUG step={step}]", flush=True)
    print(f"sample_id={record.get('sample_id', '')} type={record.get('answer_type', '')} state={record.get('state_kind', '')}", flush=True)
    print(f"GT: {record.get('gt_answer', '')}", flush=True)
    print(f"state_answer: {record.get('current_state_answer', '')}", flush=True)
    print(
        f"pos={record.get('position')} ans_idx={record.get('answer_index')} "
        f"target={record.get('target_token_text')!r} "
        f"p_target={float(record.get('target_prob', 0.0)):.6g} "
        f"logp_target={float(record.get('target_logp', 0.0)):.4f}",
        flush=True,
    )
    print(
        f"best_reward={float(record.get('best_reward', 0.0)):.3f} "
        f"model_choice_reward={float(record.get('model_choice_reward', 0.0)):.3f} "
        f"reward_gap={float(record.get('reward_gap', 0.0)):.3f}",
        flush=True,
    )
    print("candidate_table: token | reward | q_target | pi_model | answer | source", flush=True)
    for row in (record.get("candidate_table") or [])[:max_rows]:
        token = row.get("token_text", "")
        answer = row.get("candidate_answer", "")
        mark = "*" if row.get("is_gt") else " "
        print(
            f" {mark} {token!r:12s} | R={float(row.get('reward', 0.0)):+.3f} "
            f"q={float(row.get('q_target', 0.0)):.3f} "
            f"pi={float(row.get('pi_model', 0.0)):.6g} | {answer!r} | {row.get('source', '')}",
            flush=True,
        )
    print("", flush=True)


def train(cfg: Dict, run_name: str = "rrpi", start_name: str = "QRA") -> Path:
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not bool(cfg.get("cpu", False)) else "cpu")

    out_root = resolve_repo_path(cfg.get("output", {}).get("dir", f"rl_qra_pipeline/modelparameter/rl_{start_name}"))
    output_name = cfg.get("output", {}).get("name", f"rl_{start_name}")
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"{stamp}_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = GPT2TokenizerFast.from_pretrained(cfg["model"].get("tokenizer", "gpt2"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[{start_name}] device={device}", flush=True)
    model, graph, noise, ema, loaded_from = load_policy(cfg, device)
    split = cfg.get("data", {}).get("split", "train")
    samples = load_samples(cfg, split, tokenizer)
    if not samples:
        raise RuntimeError("No training samples loaded.")
    sample_iter = cycle_samples(samples, seed)

    lr = float(cfg["training"].get("lr", 2e-6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(cfg["training"].get("weight_decay", 0.0)))
    batch_size = int(cfg["training"].get("batch_size", 1))
    steps = int(cfg["training"].get("steps", 1000))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))
    save_every = int(cfg["training"].get("save_every", 100))
    log_every = int(cfg["training"].get("log_every", 1))
    rrpi_cfg = cfg.get("rrpi", cfg.get("guided", {}))
    debug_every = int(rrpi_cfg.get("debug_every", 20))
    max_debug_rows = int(rrpi_cfg.get("max_debug_rows", 8))

    dump_json(
        out_dir / "run_info.json",
        {
            "algorithm": "RRPI: Ratio-Reward Policy Improvement",
            "start": start_name,
            "output_name": output_name,
            "loaded_from": loaded_from,
            "num_samples": len(samples),
            "data_split": split,
            "device": str(device),
            "config": cfg,
        },
    )
    metrics_path = out_dir / "metrics.csv"
    debug_path = out_dir / "rrpi_debug.jsonl"

    best_loss = float("inf")
    best_path = out_dir / "best_RL_QRA_rrpi.pth"
    last_path = out_dir / "last_RL_QRA_rrpi.pth"

    numeric_keys = [k for k in METRIC_FIELDS if k not in {"step", "loss", "lr"}]
    last_debug_records: List[Dict[str, Any]] = []

    for step in range(1, steps + 1):
        model.train(True)
        optimizer.zero_grad(set_to_none=True)
        losses: List[torch.Tensor] = []
        agg = {k: 0.0 for k in numeric_keys}
        debug_records: List[Dict[str, Any]] = []
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
            for k in numeric_keys:
                if k in stats_i:
                    try:
                        agg[k] += float(stats_i.get(k, 0.0))
                    except Exception:
                        pass
            debug_records.extend(stats_i.get("debug_records", []) or [])
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
        last_debug_records = debug_records or last_debug_records

        if step % log_every == 0:
            append_csv(metrics_path, row)
            print(
                f"step={step} loss={row['loss']:.4f} "
                f"target_logp={row.get('target_logp', 0.0):.4f} "
                f"p_target={row.get('target_prob', 0.0):.6g} "
                f"modelR={row.get('model_reward', 0.0):+.3f} "
                f"bestR={row.get('best_reward', 0.0):+.3f} "
                f"gap={row.get('reward_gap', 0.0):.3f} "
                f"targets={row.get('guided_targets', 0.0):.1f}",
                flush=True,
            )

        if debug_records:
            for rec in debug_records:
                append_jsonl(debug_path, {"step": step, **rec})
        if debug_every > 0 and step % debug_every == 0 and last_debug_records:
            print_debug_record(step, last_debug_records[0], max_rows=max_debug_rows)

        if row["loss"] < best_loss:
            best_loss = row["loss"]
            save_checkpoint(best_path, model, ema, optimizer, cfg, step, row)
            sync_checkpoint(best_path, out_root / "best.pth")
        if save_every > 0 and step % save_every == 0:
            save_checkpoint(last_path, model, ema, optimizer, cfg, step, row)
            sync_checkpoint(last_path, out_root / "last.pth")

    save_checkpoint(last_path, model, ema, optimizer, cfg, steps, {"loss": best_loss})
    sync_checkpoint(last_path, out_root / "last.pth")
    sync_checkpoint(best_path, out_root / "best.pth")
    print(f"Done. Outputs: {out_dir}", flush=True)
    print(f"Synced checkpoints: {out_root / 'best.pth'} and {out_root / 'last.pth'}", flush=True)
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-name", type=str, default="rrpi")
    parser.add_argument("--start", type=str, default=None, help="Start key under config.starts, e.g. pretrain, QA, QRA")
    args = parser.parse_args()
    raw_cfg = load_config(args.config)
    cfg, start_name = select_start_config(raw_cfg, args.start)
    train(cfg, args.run_name, start_name=start_name)


if __name__ == "__main__":
    main()
