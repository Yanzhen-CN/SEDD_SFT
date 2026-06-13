from __future__ import annotations

"""Evaluate RL-QRA checkpoints on the test set.

Default comparison models are exactly six:
    pretrain, pretrain_mini_loss_best, pretrain_reward_best,
    QRA, QRA_mini_loss_best, QRA_reward_best

Outputs are written under:
    rl_qra_pipeline/modelparameter/test_result/

Files:
    test_rl_qra.csv                 # latest compact comparison table
    test_rl_qra.json                # latest structured report
    test_rl_qra_<timestamp>.csv     # timestamped archive
    test_rl_qra_<timestamp>.json    # timestamped archive
    test_rl_qra.log                 # latest terminal-style summary
"""

import argparse
import copy
import csv
import datetime as dt
import gc
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import yaml
from transformers import GPT2TokenizerFast

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
for p in (REPO_DIR, SCRIPT_DIR, REPO_DIR / "sft_answer_pipeline", REPO_DIR / "sft_rl_pipeline"):
    p_str = str(p)
    if p.exists() and p_str not in sys.path:
        sys.path.insert(0, p_str)

# Reuse the training/eval implementation so test metrics match training/select_best exactly.
from train_rl_qra import (  # noqa: E402
    choose_device,
    evaluate_loss,
    load_config,
    load_policy,
    load_samples,
    parse_eval_limit,
    resolve_repo_path,
    select_start_config,
)

DEFAULT_CONFIG = SCRIPT_DIR / "rl_qra_config.yaml"
DEFAULT_MODELS = [
    "pretrain",
    "pretrain_mini_loss_best",
    "pretrain_reward_best",
    "QRA",
    "QRA_mini_loss_best",
    "QRA_reward_best",
]
DEFAULT_OUT_DIR = SCRIPT_DIR / "modelparameter" / "test_result"


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def selected_start_cfg(cfg: Dict[str, Any], start_name: str) -> Dict[str, Any]:
    return ((cfg.get("starts") or {}).get(start_name) or {})


def start_output_dir(cfg: Dict[str, Any], start_name: str) -> Path:
    start_cfg = selected_start_cfg(cfg, start_name)
    return resolve_repo_path(start_cfg.get("output_dir", f"rl_qra_pipeline/modelparameter/rl_{start_name}"))


def model_pretrained_name(cfg: Dict[str, Any], start_name: str = "pretrain") -> str:
    start_cfg = selected_start_cfg(cfg, start_name)
    return str(start_cfg.get("pretrained") or (cfg.get("model") or {}).get("pretrained") or "louaaron/sedd-medium")


def init_checkpoint_path(cfg: Dict[str, Any], start_name: str) -> Optional[Path]:
    value = selected_start_cfg(cfg, start_name).get("init_checkpoint", "")
    if value is None or str(value).strip().lower() in {"", "none", "null", "pretrained", "hf", "cache"}:
        return None
    path = resolve_repo_path(value)
    return path if path.exists() else None


def latest_run_checkpoint(run_root: Path, filename: str = "last.pth") -> Optional[Path]:
    if not run_root.exists():
        return None
    candidates: List[Path] = []
    for p in run_root.iterdir():
        if not p.is_dir():
            continue
        ckpt = p / filename
        if ckpt.exists():
            candidates.append(ckpt)
            continue
        # Backward compatibility for earlier run naming.
        legacy_map = {
            "last.pth": "last_run.pth",
            "best.pth": "best_run.pth",
            "best_reward.pth": "best_reward_run.pth",
            "best_mini.pth": "best_mini_run.pth",
        }
        legacy_name = legacy_map.get(filename)
        if legacy_name:
            legacy = p / legacy_name
            if legacy.exists():
                candidates.append(legacy)
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x.stat().st_mtime)[-1]


def reward_checkpoint_or_fallback(root: Path) -> Tuple[Optional[Path], str, str]:
    ckpt = root / "best_reward.pth"
    if ckpt.exists():
        return ckpt, "rl_root_best_reward", ""
    fallback = latest_run_checkpoint(root, "last.pth")
    if fallback is not None:
        return fallback, "latest_run_last_fallback_missing_best_reward", f"missing {ckpt}; fallback to {fallback}"
    return None, "missing", f"missing {ckpt} and no run last checkpoint under {root}"


def resolve_model_plan(cfg: Dict[str, Any], model_names: Sequence[str]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for raw_name in model_names:
        name_l = raw_name.strip().lower()
        if not name_l:
            continue

        if name_l in {"pretrain", "pretrain_start", "start_pretrain"}:
            specs.append({
                "name": "pretrain",
                "checkpoint": None,
                "pretrained": model_pretrained_name(cfg, "pretrain"),
                "source": "start_pretrained_or_cache",
                "warning": "",
            })
            continue

        pre_root = start_output_dir(cfg, "pretrain")
        if name_l in {"pretrain_mini_loss_best", "pretrain_loss_best", "rl_pretrain", "rl_pretrain_best"}:
            specs.append({
                "name": "pretrain_mini_loss_best",
                "checkpoint": pre_root / "best.pth",
                "pretrained": model_pretrained_name(cfg, "pretrain"),
                "source": "rl_root_loss_best",
                "warning": "",
            })
            continue
        if name_l in {"pretrain_reward_best", "rl_pretrain_reward", "rl_pretrain_best_reward"}:
            ckpt, source, warning = reward_checkpoint_or_fallback(pre_root)
            specs.append({
                "name": "pretrain_reward_best",
                "checkpoint": ckpt,
                "pretrained": model_pretrained_name(cfg, "pretrain"),
                "source": source,
                "warning": warning,
            })
            continue

        if name_l in {"qra", "qra_start", "start_qra"}:
            specs.append({
                "name": "QRA",
                "checkpoint": init_checkpoint_path(cfg, "QRA"),
                "pretrained": model_pretrained_name(cfg, "QRA"),
                "source": "start_init_checkpoint",
                "warning": "",
            })
            continue

        qra_root = start_output_dir(cfg, "QRA")
        if name_l in {"qra_mini_loss_best", "qra_loss_best", "rl_qra", "rl_qra_best"}:
            specs.append({
                "name": "QRA_mini_loss_best",
                "checkpoint": qra_root / "best.pth",
                "pretrained": model_pretrained_name(cfg, "QRA"),
                "source": "rl_root_loss_best",
                "warning": "",
            })
            continue
        if name_l in {"qra_reward_best", "rl_qra_reward", "rl_qra_best_reward"}:
            ckpt, source, warning = reward_checkpoint_or_fallback(qra_root)
            specs.append({
                "name": "QRA_reward_best",
                "checkpoint": ckpt,
                "pretrained": model_pretrained_name(cfg, "QRA"),
                "source": source,
                "warning": warning,
            })
            continue

        raise ValueError(f"Unknown model name: {raw_name}")
    return specs


def metric_value(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def add_ranks(rows: List[Dict[str, Any]]) -> None:
    loss_sorted = sorted([r for r in rows if metric_value(r, "test_loss", float("inf")) < float("inf")], key=lambda r: metric_value(r, "test_loss", float("inf")))
    reward_sorted = sorted(rows, key=lambda r: metric_value(r, "test_rollout_reward", -float("inf")), reverse=True)
    for i, row in enumerate(loss_sorted, start=1):
        row["loss_rank"] = i
    for i, row in enumerate(reward_sorted, start=1):
        row["reward_rank"] = i


def make_eval_cfg(base_cfg: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("model", {})
    cfg.setdefault("training", {})
    cfg["model"]["pretrained"] = spec.get("pretrained") or cfg["model"].get("pretrained", "louaaron/sedd-medium")
    ckpt = spec.get("checkpoint")
    cfg["model"]["init_checkpoint"] = str(ckpt) if ckpt else None
    return cfg


def test_one_model(spec: Dict[str, Any], base_cfg: Dict[str, Any], tokenizer, samples: List[Dict[str, Any]], device: torch.device, limit: Optional[int]) -> Dict[str, Any]:
    ckpt = spec.get("checkpoint")
    row: Dict[str, Any] = {
        "model": spec.get("name"),
        "source": spec.get("source"),
        "checkpoint": str(ckpt) if ckpt else "",
        "pretrained": spec.get("pretrained", ""),
        "warning": spec.get("warning", ""),
    }
    if ckpt is not None and not Path(ckpt).exists():
        row.update({"status": "missing_checkpoint", "test_loss": float("inf"), "test_count": 0})
        return row
    try:
        model_cfg = make_eval_cfg(base_cfg, spec)
        model, graph, noise, ema, loaded_from = load_policy(model_cfg, device)
        row["loaded_from"] = loaded_from
        eval_info = evaluate_loss(model, graph, noise, tokenizer, samples, model_cfg, device, limit=limit)
        row.update({
            "status": "ok",
            "test_loss": float(eval_info.get("eval_loss", float("inf"))),
            "test_count": int(eval_info.get("eval_count", 0)),
            "test_rollout_loss": float(eval_info.get("rollout_loss", 0.0)),
            "test_rollout_reward": float(eval_info.get("rollout_reward", 0.0)),
            "test_rollout_reward_std": float(eval_info.get("rollout_reward_std", 0.0)),
            "test_rollout_entropy": float(eval_info.get("rollout_entropy", 0.0)),
            "test_rollout_logprob": float(eval_info.get("rollout_logprob", 0.0)),
            "test_rollout_anchor_loss": float(eval_info.get("rollout_anchor_loss", 0.0)),
        })
        del model, graph, noise, ema
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return row
    except Exception as exc:
        row.update({"status": "error", "error": repr(exc), "test_loss": float("inf"), "test_count": 0})
        return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate six RL-QRA models on test split and report loss/reward.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--start", default="QRA", help="Used only to resolve data/model defaults. Usually QRA.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", default="all", help="all/full/-1 or positive integer")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    raw_cfg = load_config(args.config)
    cfg, start_name = select_start_config(raw_cfg, args.start)
    device = choose_device(cfg, cli_gpu=args.gpu, cli_cpu=args.cpu)
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = GPT2TokenizerFast.from_pretrained(cfg["model"].get("tokenizer", "gpt2"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    samples = load_samples(cfg, args.split, tokenizer)
    limit = parse_eval_limit(args.limit, default=len(samples))
    specs = resolve_model_plan(raw_cfg, [x.strip() for x in args.models.split(",") if x.strip()])

    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    rows: List[Dict[str, Any]] = []
    print(f"[test_rl_qra] device={device} split={args.split} loaded={len(samples)} limit={'all' if limit is None else limit}", flush=True)
    for spec in specs:
        if spec.get("warning"):
            print(f"[warn] {spec.get('name')}: {spec.get('warning')}", flush=True)
        print(f"[test_rl_qra] eval {spec.get('name')} source={spec.get('source')} ckpt={spec.get('checkpoint')}", flush=True)
        row = test_one_model(spec, cfg, tokenizer, samples, device, limit)
        row.update({"split": args.split, "limit": "all" if limit is None else int(limit)})
        rows.append(row)
        print(
            f"  -> status={row.get('status')} loss={metric_value(row, 'test_loss', float('inf')):.6g} "
            f"reward={metric_value(row, 'test_rollout_reward', 0.0):+.4f} count={row.get('test_count', 0)}",
            flush=True,
        )

    add_ranks(rows)
    fields = [
        "model", "status", "source", "split", "limit", "checkpoint", "pretrained", "loaded_from", "warning", "error",
        "test_count", "test_loss", "loss_rank", "test_rollout_loss", "test_rollout_reward", "reward_rank",
        "test_rollout_reward_std", "test_rollout_entropy", "test_rollout_logprob", "test_rollout_anchor_loss",
    ]
    write_csv(out_dir / "test_rl_qra.csv", rows, fields)
    write_csv(out_dir / f"test_rl_qra_{stamp}.csv", rows, fields)
    report = {
        "time": stamp,
        "config": str(resolve_repo_path(args.config)),
        "split": args.split,
        "loaded_samples": len(samples),
        "limit": "all" if limit is None else int(limit),
        "device": str(device),
        "models": rows,
        "best_by_loss": min(rows, key=lambda r: metric_value(r, "test_loss", float("inf"))) if rows else None,
        "best_by_reward": max(rows, key=lambda r: metric_value(r, "test_rollout_reward", -float("inf"))) if rows else None,
    }
    dump_json(out_dir / "test_rl_qra.json", report)
    dump_json(out_dir / f"test_rl_qra_{stamp}.json", report)

    with open(out_dir / "test_rl_qra.log", "w", encoding="utf-8") as f:
        f.write(f"test_rl_qra | split={args.split} | samples={len(samples)} | limit={report['limit']} | device={device}\n")
        f.write("model | loss | reward | count | checkpoint\n")
        for r in rows:
            f.write(
                f"{r.get('model')} | {metric_value(r, 'test_loss', float('inf')):.6g} | "
                f"{metric_value(r, 'test_rollout_reward', 0.0):+.4f} | {r.get('test_count', 0)} | {r.get('checkpoint', '')}\n"
            )
        f.write(f"best_by_loss: {report['best_by_loss'].get('model') if report.get('best_by_loss') else ''}\n")
        f.write(f"best_by_reward: {report['best_by_reward'].get('model') if report.get('best_by_reward') else ''}\n")

    print(f"[test_rl_qra] wrote {out_dir / 'test_rl_qra.csv'}", flush=True)
    print(f"[test_rl_qra] wrote {out_dir / 'test_rl_qra.json'}", flush=True)


if __name__ == "__main__":
    main()
