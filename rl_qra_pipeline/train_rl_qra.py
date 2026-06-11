from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import os
import random
import shutil
import time
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
from rollout_chain_update import rollout_chain_loss  # noqa: E402

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
    # Rollout-chain RL metrics.
    "rollout_loss",
    "rollout_reward",
    "rollout_reward_min",
    "rollout_reward_max",
    "rollout_reward_std",
    "rollout_entropy",
    "rollout_logprob",
    "rollout_anchor_loss",
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


def compute_rl_loss(model, graph, noise, tokenizer, sample: Dict, cfg: Dict, device: torch.device):
    """Select the RL objective.

    rrpi/local: old local candidate-policy update.
    rollout_chain: true reverse-generation policy gradient.
    """
    mode = str(cfg.get("rl", {}).get("mode", cfg.get("rrpi", {}).get("mode", "rrpi")))
    if mode in {"rollout", "rollout_chain", "chain"}:
        return rollout_chain_loss(model, graph, noise, tokenizer, sample, cfg, device)
    return guided_ratio_loss(model, graph, noise, tokenizer, sample, cfg, device)


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
    if "chain" in record:
        print(f"\n[ROLLOUT DEBUG step={step}]", flush=True)
        print(f"sample_id={record.get('sample_id', '')} type={record.get('answer_type', '')}", flush=True)
        print(f"GT: {record.get('gt_answer', '')}", flush=True)
        print("k | t | pos | token | reward | adv | action_align | skelΔ | exactΔ | before -> after", flush=True)
        for row in (record.get("chain") or [])[:max_rows]:
            print(
                f"{int(row.get('step', 0)):02d} | {float(row.get('t', 0.0)):.2f} | "
                f"{int(row.get('action_pos', -1)):02d} | {row.get('token', '')!r:8s} | "
                f"R={float(row.get('reward', 0.0)):+.3f} | A={float(row.get('advantage', 0.0)):+.3f} | "
                f"align={float(row.get('r_action_align', 0.0)):+.3f} | "
                f"sk={float(row.get('r_skeleton_delta', 0.0)):+.3f} | "
                f"ex={float(row.get('r_exact_delta', 0.0)):+.3f} | "
                f"{row.get('before', '')!r} -> {row.get('after', '')!r}",
                flush=True,
            )
        print("", flush=True)
        return

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



def choose_device(cfg: Dict, cli_gpu: int | None = None, cli_cpu: bool = False) -> torch.device:
    if cli_cpu or bool(cfg.get("cpu", False)):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if cli_gpu is not None:
        return torch.device(f"cuda:{int(cli_gpu)}")
    cfg_gpu = cfg.get("run", {}).get("cuda_device", None)
    if cfg_gpu is not None:
        return torch.device(f"cuda:{int(cfg_gpu)}")
    return torch.device("cuda")


def acquire_lock(lock_path: Path, timeout_s: float = 300.0, poll_s: float = 0.25):
    start = time.time()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()} time={time.time()}\n".encode("utf-8"))
            os.close(fd)
            return
        except FileExistsError:
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Timeout waiting for lock: {lock_path}")
            time.sleep(poll_s)


def release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def mean_dict(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = set().union(*(r.keys() for r in rows))
    out: Dict[str, float] = {}
    for k in keys:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except Exception:
                pass
        if vals:
            out[k] = float(sum(vals) / len(vals))
    return out


def evaluate_loss(model, graph, noise, tokenizer, samples: List[Dict], cfg: Dict, device: torch.device, limit: int = 128) -> Dict[str, Any]:
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg.setdefault("rrpi", {})
    eval_cfg.setdefault("rollout", {})
    eval_cfg["rrpi"]["transition_train"] = False
    eval_cfg["rrpi"]["debug_records_per_step"] = 0
    eval_cfg["rrpi"]["debug_every"] = 0
    eval_cfg["rollout"]["debug_records_per_step"] = 0
    eval_cfg["rollout"]["mode"] = "greedy"
    eval_cfg["rollout"]["num_rollouts"] = 1
    if limit and limit > 0:
        samples = samples[: min(limit, len(samples))]
    model.eval()
    losses: List[float] = []
    stats_rows: List[Dict[str, float]] = []
    with torch.no_grad():
        for sample in samples:
            try:
                loss_i, stats_i = compute_rl_loss(model, graph, noise, tokenizer, sample, eval_cfg, device)
            except Exception:
                continue
            losses.append(float(loss_i.detach().item()))
            stats_rows.append({k: float(v) for k, v in stats_i.items() if isinstance(v, (int, float))})
    out = mean_dict(stats_rows)
    out["eval_loss"] = float(sum(losses) / max(1, len(losses))) if losses else float("inf")
    out["eval_count"] = int(len(losses))
    return out


def load_eval_samples(cfg: Dict, tokenizer, train_samples: List[Dict]) -> Tuple[List[Dict], str]:
    data_cfg = cfg.get("data", {})
    requested = data_cfg.get("eval_split", "val")
    candidates = []
    if requested:
        candidates.append(str(requested))
    candidates.extend(["val", "valid", "validation", "test"])
    seen = set()
    for split in candidates:
        if split in seen:
            continue
        seen.add(split)
        try:
            samples = load_samples(cfg, split, tokenizer)
            if samples:
                return samples, split
        except Exception:
            pass
    # Fallback keeps the script runnable, but this should be avoided for final reporting.
    limit = int(data_cfg.get("eval_limit", 128) or 128)
    return train_samples[: min(limit, len(train_samples))], "train_fallback"


def update_global_best(
    out_root: Path,
    run_dir: Path,
    best_ckpt: Path,
    eval_info: Dict[str, Any],
    metrics_json: Dict[str, Any],
    run_info_path: Path,
    metrics_path: Path,
) -> bool:
    """Root stays clean: only global-best files are kept at out_root.

    The individual run keeps all artifacts under out_root/runs/<run_id>/.
    At the end of a run, compare eval_loss against root best_metrics.json.
    If better, atomically update the root best files.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    lock = out_root / ".global_best.lock"
    acquire_lock(lock)
    improved = False
    try:
        current_path = out_root / "best_metrics.json"
        current_best = float("inf")
        if current_path.exists():
            try:
                current = json.loads(current_path.read_text(encoding="utf-8"))
                current_best = float(current.get("eval_loss", current.get("best_eval_loss", float("inf"))))
            except Exception:
                current_best = float("inf")

        new_loss = float(eval_info.get("eval_loss", float("inf")))
        if new_loss < current_best:
            improved = True
            if best_ckpt.exists():
                shutil.copy2(best_ckpt, out_root / "best.pth")
            shutil.copy2(metrics_path, out_root / "best_metrics.csv")
            dump_json(out_root / "best_eval.json", eval_info)
            dump_json(out_root / "best_metrics.json", metrics_json)
            shutil.copy2(run_info_path, out_root / "best_run_info.json")

            log_row = {
                "time": dt.datetime.now().isoformat(timespec="seconds"),
                "run_dir": str(run_dir),
                "old_eval_loss": current_best,
                "new_eval_loss": new_loss,
                "improved": True,
            }
            append_jsonl(out_root / "improvement_log.jsonl", log_row)
    finally:
        release_lock(lock)
    return improved


def train(cfg: Dict, run_name: str = "rollout_slotalign", start_name: str = "QRA", cli_gpu: int | None = None, cli_cpu: bool = False) -> Path:
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = choose_device(cfg, cli_gpu=cli_gpu, cli_cpu=cli_cpu)

    out_root = resolve_repo_path(cfg.get("output", {}).get("dir", f"rl_qra_pipeline/modelparameter/rl_{start_name}"))
    output_name = cfg.get("output", {}).get("name", f"rl_{start_name}")
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_run = str(run_name).replace("/", "_").replace(" ", "_")
    run_id = f"{stamp}_{safe_run}_pid{os.getpid()}"
    # Keep root clean. All per-run artifacts live under runs/<run_id>/.
    out_dir = out_root / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = GPT2TokenizerFast.from_pretrained(cfg["model"].get("tokenizer", "gpt2"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(
        f"[{start_name}] device={device} cli_gpu={cli_gpu} cfg_gpu={cfg.get('run', {}).get('cuda_device', None)} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        flush=True,
    )
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
    rollout_cfg = cfg.get("rollout", {})
    mode_for_debug = str(cfg.get("rl", {}).get("mode", rrpi_cfg.get("mode", "rrpi")))
    if mode_for_debug in {"rollout", "rollout_chain", "chain"}:
        debug_every = int(rollout_cfg.get("debug_every", rrpi_cfg.get("debug_every", 20)))
        max_debug_rows = int(rollout_cfg.get("max_debug_steps", rrpi_cfg.get("max_debug_rows", 8)))
    else:
        debug_every = int(rrpi_cfg.get("debug_every", 20))
        max_debug_rows = int(rrpi_cfg.get("max_debug_rows", 8))

    run_info = {
        "algorithm": str(cfg.get("rl", {}).get("mode", "rrpi")),
        "start": start_name,
        "output_name": output_name,
        "run_id": run_id,
        "run_dir": str(out_dir),
        "loaded_from": loaded_from,
        "num_samples": len(samples),
        "data_split": split,
        "device": str(device),
        "cli_gpu": cli_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "config": cfg,
    }
    run_info_path = out_dir / "run_info.json"
    dump_json(run_info_path, run_info)
    metrics_path = out_dir / "metrics.csv"
    debug_path = out_dir / "rollout_debug.jsonl"

    best_loss = float("inf")
    best_step = 0
    best_row: Dict[str, Any] = {}
    best_path = out_dir / "best_run.pth"
    last_path = out_dir / "last_run.pth"

    numeric_keys = [k for k in METRIC_FIELDS if k not in {"step", "loss", "lr"}]
    last_debug_records: List[Dict[str, Any]] = []
    last_row: Dict[str, Any] = {}

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
                loss_i, stats_i = compute_rl_loss(model, graph, noise, tokenizer, sample, cfg, device)
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
        last_row = row
        last_debug_records = debug_records or last_debug_records

        if step % log_every == 0:
            append_csv(metrics_path, row)
            print(
                f"step={step} loss={row['loss']:.4f} "
                f"target_logp={row.get('target_logp', 0.0):.4f} "
                f"modelR={row.get('model_reward', 0.0):+.3f} "
                f"Rstd={row.get('rollout_reward_std', 0.0):.3f} "
                f"entropy={row.get('rollout_entropy', row.get('candidate_entropy', 0.0)):.3f} "
                f"anchor={row.get('rollout_anchor_loss', 0.0):.4f} "
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
            best_step = step
            best_row = dict(row)
            save_checkpoint(best_path, model, ema, optimizer, cfg, step, row)
        if save_every > 0 and step % save_every == 0:
            save_checkpoint(last_path, model, ema, optimizer, cfg, step, row)

    if not last_path.exists():
        save_checkpoint(last_path, model, ema, optimizer, cfg, steps, last_row or {"loss": best_loss})
    if not best_path.exists():
        save_checkpoint(best_path, model, ema, optimizer, cfg, best_step or steps, best_row or last_row or {"loss": best_loss})

    # Evaluate the run's own best training checkpoint, then compete for root global best.
    try:
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state.get("model", state), strict=True)
        if isinstance(state, dict) and "ema" in state:
            try:
                ema.load_state_dict(state["ema"])
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
            except Exception:
                pass
    except Exception as exc:
        print(f"[warn] failed to reload best checkpoint for eval: {exc}", flush=True)

    eval_samples, eval_split = load_eval_samples(cfg, tokenizer, samples)
    eval_limit = int(cfg.get("data", {}).get("eval_limit", 128) or 128)
    eval_info = evaluate_loss(model, graph, noise, tokenizer, eval_samples, cfg, device, limit=eval_limit)
    eval_info.update({"eval_split": eval_split, "eval_limit": eval_limit, "run_id": run_id, "best_train_step": best_step, "best_train_loss": best_loss})
    dump_json(out_dir / "eval.json", eval_info)

    metrics_json = {
        "run_id": run_id,
        "run_dir": str(out_dir),
        "best_train_step": best_step,
        "best_train_loss": best_loss,
        "last_train_loss": float(last_row.get("loss", float("inf"))) if last_row else float("inf"),
        "eval_loss": float(eval_info.get("eval_loss", float("inf"))),
        "eval_split": eval_split,
        "eval_count": int(eval_info.get("eval_count", 0)),
        "last_row": last_row,
        "best_row": best_row,
        "eval_metrics": eval_info,
    }
    dump_json(out_dir / "metrics.json", metrics_json)

    improved = update_global_best(out_root, out_dir, best_path, eval_info, metrics_json, run_info_path, metrics_path)
    print(f"Done. Outputs: {out_dir}", flush=True)
    print(f"Run eval_loss={eval_info.get('eval_loss')} improved_global_best={improved}", flush=True)
    if improved:
        print(f"Updated root global best files under: {out_root}", flush=True)
    return out_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-name", type=str, default="rollout_slotalign")
    parser.add_argument("--start", type=str, default=None, help="Start key under config.starts, e.g. pretrain, QA, QRA")
    parser.add_argument("--gpu", type=int, default=None, help="Use cuda:<gpu>. Overrides run.cuda_device in config.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    args = parser.parse_args()
    raw_cfg = load_config(args.config)
    cfg, start_name = select_start_config(raw_cfg, args.start)
    train(cfg, args.run_name, start_name=start_name, cli_gpu=args.gpu, cli_cpu=args.cpu)


if __name__ == "__main__":
    main()
