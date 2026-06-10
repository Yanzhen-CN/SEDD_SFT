from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import os
import random
import shutil
import sys

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None
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


def choose_device(cfg: Dict, gpu_override: int | None = None) -> torch.device:
    """Choose device from CLI first, then config, then default cuda.

    Priority:
      1) --gpu N from command line
      2) run.cuda_device in yaml
      3) top-level cuda_device / gpu in yaml
      4) cuda if available, else cpu

    Notes:
      - If you run with CUDA_VISIBLE_DEVICES=2 and --gpu 0, cuda:0 means the
        first visible GPU, i.e. physical GPU 2.
      - Set cpu: true or run.cpu: true to force CPU.
    """
    run_cfg = cfg.get("run", {}) or {}
    force_cpu = bool(cfg.get("cpu", False) or run_cfg.get("cpu", False))
    if force_cpu:
        return torch.device("cpu")

    if not torch.cuda.is_available():
        print("[warn] CUDA is not available; using CPU.", flush=True)
        return torch.device("cpu")

    gpu_value = gpu_override
    if gpu_value is None:
        gpu_value = run_cfg.get("cuda_device", cfg.get("cuda_device", cfg.get("gpu", None)))

    if gpu_value is None:
        return torch.device("cuda")

    if isinstance(gpu_value, str):
        if gpu_value.strip().lower() in {"", "none", "null", "auto", "cuda"}:
            return torch.device("cuda")
        gpu_value = int(gpu_value)

    return torch.device(f"cuda:{int(gpu_value)}")


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


def safe_copy(src: Path, dst: Path) -> str:
    """Copy src to dst if src exists, and return the copied path as string."""
    if not src.exists():
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def append_run_index(index_path: Path, row: Dict[str, Any]) -> None:
    """Append one finished run into a root-level index with a simple file lock.

    Multiple training processes may finish at similar times. The fcntl lock avoids
    interleaved writes on Linux servers. If fcntl is unavailable, it still appends
    normally.
    """
    index_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "run_name",
        "start",
        "device",
        "steps",
        "best_loss",
        "final_loss",
        "last_target_prob",
        "last_target_logp",
        "last_model_reward",
        "last_best_reward",
        "last_reward_gap",
        "run_dir",
        "root_best_ckpt",
        "root_last_ckpt",
        "root_metrics",
        "root_debug",
        "root_run_info",
        "global_best_updated",
        "global_best_metric",
        "global_best_metric_name",
        "global_best_metric_mode",
        "global_best_summary",
        "finished_at",
    ]
    exists = index_path.exists()
    with open(index_path, "a", encoding="utf-8", newline="") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fieldnames})
            f.flush()
            os.fsync(f.fileno())
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)



def _atomic_copy(src: Path, dst: Path) -> str:
    """Copy through a temp file and replace atomically."""
    if not src.exists():
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + f".tmp.{os.getpid()}")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return str(dst)


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _metric_is_better(new_value: float, old_value: float | None, mode: str) -> bool:
    if old_value is None:
        return True
    if mode.lower() in {"max", "higher", "higher_is_better"}:
        return new_value > old_value
    return new_value < old_value


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def compute_selection_metric(best_loss: float, final_metrics: Dict[str, Any], cfg: Dict) -> Tuple[str, str, float]:
    """Return (metric_name, mode, metric_value) for global-best comparison.

    Default follows the SFT pipelines: lower best training loss wins.  For RRPI
    you can override with, for example:

      output:
        best_metric: last_model_reward
        best_metric_mode: max

    Supported default names include best_loss, final_loss, and any metric key in
    final_metrics such as model_reward, reward_gap, target_prob, etc.
    """
    output_cfg = cfg.get("output", {}) or {}
    metric_name = str(output_cfg.get("best_metric", "best_loss"))
    mode = str(output_cfg.get("best_metric_mode", "min"))

    if metric_name == "best_loss":
        value = float(best_loss)
    elif metric_name == "final_loss":
        value = _safe_float(final_metrics.get("loss"))
    elif metric_name.startswith("last_"):
        raw_name = metric_name[len("last_"):]
        value = _safe_float(final_metrics.get(raw_name))
    else:
        value = _safe_float(final_metrics.get(metric_name))

    if not np.isfinite(value):
        # Fallback to loss so the global-best update never crashes just because
        # a custom metric was absent from this run.
        metric_name = "best_loss"
        mode = "min"
        value = float(best_loss)
    return metric_name, mode, value


def update_root_best_if_better(
    *,
    out_root: Path,
    run_id: str,
    run_name: str,
    start_name: str,
    run_dir: Path,
    best_path: Path,
    last_path: Path,
    metrics_path: Path,
    debug_path: Path,
    run_info_path: Path,
    best_loss: float,
    final_metrics: Dict[str, Any],
    cfg: Dict,
) -> Dict[str, Any]:
    """Atomically compare this finished run with the root best and promote if better.

    This matches the usual SFT-pipeline behavior while remaining safe for
    simultaneous runs: every process trains in its own directory; at the end it
    grabs a root lock, compares its best metric with the current root best, and
    only then updates root-level best.pth plus the matching metrics/debug/info.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    lock_path = out_root / ".global_best.lock"
    summary_path = out_root / "best_summary.json"

    metric_name, metric_mode, metric_value = compute_selection_metric(best_loss, final_metrics, cfg)
    promoted = False
    previous_value = None
    previous_run = ""

    with open(lock_path, "a+", encoding="utf-8") as lock_f:
        if fcntl is not None:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            old_summary = _read_json_if_exists(summary_path)
            previous_run = str(old_summary.get("run_id", ""))
            old_metric_name = str(old_summary.get("metric_name", metric_name))
            old_metric_mode = str(old_summary.get("metric_mode", metric_mode))
            # If user changes best_metric between runs, start a new comparison
            # under the new metric rather than comparing incompatible numbers.
            if old_metric_name == metric_name and old_metric_mode == metric_mode:
                previous_value = old_summary.get("metric_value", None)
                previous_value = None if previous_value is None else float(previous_value)
            else:
                previous_value = None

            if _metric_is_better(metric_value, previous_value, metric_mode):
                _atomic_copy(best_path, out_root / "best.pth")
                _atomic_copy(last_path, out_root / "best_run_last.pth")
                _atomic_copy(metrics_path, out_root / "best_metrics.csv")
                _atomic_copy(debug_path, out_root / "best_rrpi_debug.jsonl")
                _atomic_copy(run_info_path, out_root / "best_run_info.json")

                best_summary = {
                    "run_id": run_id,
                    "run_name": run_name,
                    "start": start_name,
                    "run_dir": str(run_dir),
                    "metric_name": metric_name,
                    "metric_mode": metric_mode,
                    "metric_value": metric_value,
                    "best_loss": best_loss,
                    "final_metrics": final_metrics,
                    "best_ckpt": str(out_root / "best.pth"),
                    "best_run_last_ckpt": str(out_root / "best_run_last.pth"),
                    "best_metrics": str(out_root / "best_metrics.csv"),
                    "best_debug": str(out_root / "best_rrpi_debug.jsonl"),
                    "best_run_info": str(out_root / "best_run_info.json"),
                    "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
                }
                tmp_summary = summary_path.with_suffix(".json.tmp")
                tmp_summary.write_text(json.dumps(best_summary, ensure_ascii=False, indent=2), encoding="utf-8")
                os.replace(tmp_summary, summary_path)
                promoted = True

            # Also maintain last-finished artifacts as a separate namespace. This
            # is useful for debugging, but it never decides the global best.
            _atomic_copy(last_path, out_root / "last_finished.pth")
            _atomic_copy(metrics_path, out_root / "last_finished_metrics.csv")
            _atomic_copy(debug_path, out_root / "last_finished_rrpi_debug.jsonl")
            _atomic_copy(run_info_path, out_root / "last_finished_run_info.json")
        finally:
            if fcntl is not None:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    return {
        "promoted": promoted,
        "metric_name": metric_name,
        "metric_mode": metric_mode,
        "metric_value": metric_value,
        "previous_metric_value": previous_value,
        "previous_run_id": previous_run,
        "summary_path": str(summary_path),
    }

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


def train(cfg: Dict, run_name: str = "rrpi", start_name: str = "QRA", gpu_override: int | None = None) -> Path:
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = choose_device(cfg, gpu_override=gpu_override)

    out_root = resolve_repo_path(cfg.get("output", {}).get("dir", f"rl_qra_pipeline/modelparameter/rl_{start_name}"))
    output_name = cfg.get("output", {}).get("name", f"rl_{start_name}")
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Include pid to avoid collisions when multiple runs start in the same second.
    run_id = f"{stamp}_{run_name}_pid{os.getpid()}"
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = GPT2TokenizerFast.from_pretrained(cfg["model"].get("tokenizer", "gpt2"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    run_cfg = cfg.get("run", {}) or {}
    cfg_gpu = run_cfg.get("cuda_device", cfg.get("cuda_device", cfg.get("gpu", None)))
    print(
        f"[{start_name}] device={device} "
        f"cli_gpu={gpu_override} cfg_gpu={cfg_gpu} "
        f"cuda_visible_devices={__import__('os').environ.get('CUDA_VISIBLE_DEVICES', '')}",
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
    debug_every = int(rrpi_cfg.get("debug_every", 20))
    max_debug_rows = int(rrpi_cfg.get("max_debug_rows", 8))

    dump_json(
        out_dir / "run_info.json",
        {
            "algorithm": "RRPI: Ratio-Reward Policy Improvement",
            "run_id": run_id,
            "run_name": run_name,
            "start": start_name,
            "output_name": output_name,
            "output_root": str(out_root),
            "loaded_from": loaded_from,
            "num_samples": len(samples),
            "data_split": split,
            "device": str(device),
            "pid": os.getpid(),
            "config": cfg,
        },
    )
    metrics_path = out_dir / "metrics.csv"
    debug_path = out_dir / "rrpi_debug.jsonl"
    run_info_path = out_dir / "run_info.json"

    print(f"run_id={run_id}", flush=True)
    print(f"run_dir={out_dir}", flush=True)
    print(f"live_metrics={metrics_path}", flush=True)
    print("[parallel-safe] This run writes checkpoints only inside its own run_dir during training.", flush=True)

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
        if save_every > 0 and step % save_every == 0:
            save_checkpoint(last_path, model, ema, optimizer, cfg, step, row)

    final_metrics = row if "row" in locals() else {"loss": best_loss}
    save_checkpoint(last_path, model, ema, optimizer, cfg, steps, final_metrics)

    # Root-level export and global-best promotion.
    # Each run still keeps its full private run_dir.  In addition, it exports
    # run-specific artifacts to the root for easy plotting, then atomically
    # compares against the current root best.  Only a better run updates
    # root-level best.pth / best_metrics.csv / best_rrpi_debug.jsonl / best_run_info.json.
    root_best = out_root / f"{run_id}_best.pth"
    root_last = out_root / f"{run_id}_last.pth"
    root_metrics = out_root / f"{run_id}_metrics.csv"
    root_debug = out_root / f"{run_id}_rrpi_debug.jsonl"
    root_info = out_root / f"{run_id}_run_info.json"

    root_best_s = safe_copy(best_path, root_best)
    root_last_s = safe_copy(last_path, root_last)
    root_metrics_s = safe_copy(metrics_path, root_metrics)
    root_debug_s = safe_copy(debug_path, root_debug)
    root_info_s = safe_copy(run_info_path, root_info)

    best_update = update_root_best_if_better(
        out_root=out_root,
        run_id=run_id,
        run_name=run_name,
        start_name=start_name,
        run_dir=out_dir,
        best_path=best_path,
        last_path=last_path,
        metrics_path=metrics_path,
        debug_path=debug_path,
        run_info_path=run_info_path,
        best_loss=best_loss,
        final_metrics=final_metrics,
        cfg=cfg,
    )

    append_run_index(
        out_root / "runs_index.csv",
        {
            "run_id": run_id,
            "run_name": run_name,
            "start": start_name,
            "device": str(device),
            "steps": steps,
            "best_loss": best_loss,
            "final_loss": final_metrics.get("loss", ""),
            "last_target_prob": final_metrics.get("target_prob", ""),
            "last_target_logp": final_metrics.get("target_logp", ""),
            "last_model_reward": final_metrics.get("model_reward", ""),
            "last_best_reward": final_metrics.get("best_reward", ""),
            "last_reward_gap": final_metrics.get("reward_gap", ""),
            "run_dir": str(out_dir),
            "root_best_ckpt": root_best_s,
            "root_last_ckpt": root_last_s,
            "root_metrics": root_metrics_s,
            "root_debug": root_debug_s,
            "root_run_info": root_info_s,
            "global_best_updated": str(bool(best_update.get("promoted"))),
            "global_best_metric": best_update.get("metric_value", ""),
            "global_best_metric_name": best_update.get("metric_name", ""),
            "global_best_metric_mode": best_update.get("metric_mode", ""),
            "global_best_summary": best_update.get("summary_path", ""),
            "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
        },
    )

    print(f"Done. Outputs: {out_dir}", flush=True)
    print(f"Exported run artifacts with prefix: {out_root / run_id}", flush=True)
    print(f"Run index: {out_root / 'runs_index.csv'}", flush=True)
    if best_update.get("promoted"):
        print(
            f"[GLOBAL BEST UPDATED] {run_id} -> {out_root / 'best.pth'} "
            f"({best_update.get('metric_name')}={float(best_update.get('metric_value')):.6g}, "
            f"mode={best_update.get('metric_mode')})",
            flush=True,
        )
        print(f"Best metrics: {out_root / 'best_metrics.csv'}", flush=True)
        print(f"Best summary: {out_root / 'best_summary.json'}", flush=True)
    else:
        print(
            f"[GLOBAL BEST KEPT] current root best is better. "
            f"this_run {best_update.get('metric_name')}={float(best_update.get('metric_value')):.6g}; "
            f"previous={best_update.get('previous_metric_value')} run={best_update.get('previous_run_id')}",
            flush=True,
        )
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-name", type=str, default="rrpi")
    parser.add_argument("--start", type=str, default=None, help="Start key under config.starts, e.g. pretrain, QA, QRA")
    parser.add_argument("--gpu", type=int, default=None, help="CUDA device index, e.g. --gpu 0. Overrides config run.cuda_device.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU. Overrides --gpu and config cuda_device.")
    args = parser.parse_args()

    raw_cfg = load_config(args.config)
    cfg, start_name = select_start_config(raw_cfg, args.start)

    if args.cpu:
        cfg["cpu"] = True
        cfg.setdefault("run", {})["cpu"] = True
    if args.gpu is not None:
        cfg.setdefault("run", {})["cuda_device"] = int(args.gpu)

    train(cfg, args.run_name, start_name=start_name, gpu_override=args.gpu)


if __name__ == "__main__":
    main()
