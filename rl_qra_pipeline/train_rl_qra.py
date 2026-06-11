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
from typing import Dict, Iterable, List, Tuple, Any, Optional

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
    # Periodic validation metrics. These are populated only on eval steps.
    "eval_loss",
    "eval_count",
    "eval_rollout_loss",
    "eval_rollout_reward",
    "eval_rollout_reward_std",
    "eval_rollout_entropy",
    "eval_rollout_logprob",
    "eval_rollout_anchor_loss",
    # Sparse full-validation metrics. These are populated on full_eval steps.
    "full_eval_loss",
    "full_eval_count",
    "full_eval_rollout_loss",
    "full_eval_rollout_reward",
    "full_eval_rollout_reward_std",
    "full_eval_rollout_entropy",
    "full_eval_rollout_logprob",
    "full_eval_rollout_anchor_loss",
    "is_full_eval",
    # Mini-eval and full-eval checkpoint selection markers.
    "mini_best_metric_value",
    "is_best_mini_eval",
    "best_metric_value",
    "is_best_eval",
]


def load_config(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_repo_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else REPO_DIR / p


def is_disabled_checkpoint(value: object) -> bool:
    """True means initialize directly from SEDD.from_pretrained(...).

    This is needed for --start pretrain: other pipelines usually do not have a
    local pretrain.pth; they rely on the Hugging Face cache/download path.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null", "pretrained", "hf", "cache", "false"}
    return False


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
    if is_disabled_checkpoint(ckpt_value):
        print(f"Using pretrained/HF-cache weights: {pretrained}", flush=True)
    else:
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
        elif bool(cfg.get("model", {}).get("allow_missing_init_checkpoint", False)):
            print(f"[warn] init checkpoint not found, falling back to pretrained/HF-cache weights: {ckpt_path}", flush=True)
        else:
            raise FileNotFoundError(
                f"init checkpoint not found: {ckpt_path}\n"
                "For --start pretrain, set starts.pretrain.init_checkpoint: null/''/pretrained "
                "so SEDD.from_pretrained(...) uses the Hugging Face cache. "
                "For QA/QRA, fix starts.<name>.init_checkpoint in rl_qra_config.yaml."
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


def add_eval_prefix(row: Dict[str, Any], prefix: str, eval_info: Dict[str, Any]) -> None:
    """Copy evaluate_loss outputs into a metrics row with either eval_ or full_eval_ prefix."""
    row[f"{prefix}_loss"] = float(eval_info.get("eval_loss", float("inf")))
    row[f"{prefix}_count"] = int(eval_info.get("eval_count", 0))
    row[f"{prefix}_rollout_loss"] = float(eval_info.get("rollout_loss", 0.0))
    row[f"{prefix}_rollout_reward"] = float(eval_info.get("rollout_reward", 0.0))
    row[f"{prefix}_rollout_reward_std"] = float(eval_info.get("rollout_reward_std", 0.0))
    row[f"{prefix}_rollout_entropy"] = float(eval_info.get("rollout_entropy", 0.0))
    row[f"{prefix}_rollout_logprob"] = float(eval_info.get("rollout_logprob", 0.0))
    row[f"{prefix}_rollout_anchor_loss"] = float(eval_info.get("rollout_anchor_loss", 0.0))


def metric_from_full_eval(eval_info: Dict[str, Any], name: str, mode: str) -> float:
    """Read best metric from final/full eval info.

    full_eval_loss is stored as eval_loss inside eval.json, so we map it here.
    """
    if name == "full_eval_loss":
        return get_metric_value(eval_info, "full_eval_loss", get_metric_value(eval_info, "eval_loss", initial_best_value(mode)))
    if name.startswith("full_eval_"):
        fallback_key = name[len("full_eval_"):]
        return get_metric_value(eval_info, name, get_metric_value(eval_info, fallback_key, initial_best_value(mode)))
    return get_metric_value(eval_info, name, get_metric_value(eval_info, "eval_loss", initial_best_value(mode)))


def metric_better(new_value: float, best_value: float, mode: str = "min") -> bool:
    """Return whether a validation metric improved.

    For eval_loss use mode='min'. For a reward/exact-rate style metric use mode='max'.
    """
    try:
        new_value = float(new_value)
        best_value = float(best_value)
    except Exception:
        return False
    if not np.isfinite(new_value):
        return False
    return new_value > best_value if str(mode).lower() == "max" else new_value < best_value


def initial_best_value(mode: str = "min") -> float:
    return -float("inf") if str(mode).lower() == "max" else float("inf")


def get_metric_value(metrics: Dict[str, Any], name: str, default: float) -> float:
    try:
        return float(metrics.get(name, default))
    except Exception:
        return float(default)


def parse_eval_limit(value: Any, default: int = 64) -> Optional[int]:
    """Parse training.eval_limit.

    Supported values:
      - positive integer: evaluate that many validation samples
      - "all" / "full": evaluate the full validation split
      - -1: evaluate the full validation split
      - None / "": use default

    We intentionally do not treat 0 as "all" to avoid accidentally making
    every-step validation extremely expensive when a config leaves a zero-like
    value around from older pipelines.
    """
    if value is None:
        return int(default)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"all", "full", "complete", "entire"}:
            return None
        if v == "":
            return int(default)
        value = int(v)
    value = int(value)
    if value < 0:
        return None
    if value == 0:
        return int(default)
    return value


def eval_limit_label(limit: Optional[int]) -> str | int:
    return "all" if limit is None else int(limit)


def evaluate_loss(model, graph, noise, tokenizer, samples: List[Dict], cfg: Dict, device: torch.device, limit: Optional[int] = 128) -> Dict[str, Any]:
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


def load_eval_samples(cfg: Dict, tokenizer, train_samples: List[Dict], requested_split: Optional[str] = None) -> Tuple[List[Dict], str]:
    data_cfg = cfg.get("data", {})
    requested = requested_split if requested_split is not None else data_cfg.get("eval_split", "val")
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
    best_metric_name: str = "eval_loss",
    best_mode: str = "min",
) -> bool:
    """Root stays clean: only global-best files are kept at out_root.

    The individual run keeps all artifacts under out_root/<run_id>/.
    At the end of a run, compare the configured validation metric against root best_metrics.json.
    The default metric is eval_loss with mode=min.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    lock = out_root / ".global_best.lock"
    acquire_lock(lock)
    improved = False
    try:
        current_path = out_root / "best_metrics.json"
        current_best = initial_best_value(best_mode)
        if current_path.exists():
            try:
                current = json.loads(current_path.read_text(encoding="utf-8"))
                # Prefer the same configured metric. Fall back to legacy eval_loss.
                current_best = get_metric_value(
                    current,
                    "best_metric_value",
                    get_metric_value(current, best_metric_name, get_metric_value(current, "eval_loss", current_best)),
                )
            except Exception:
                current_best = initial_best_value(best_mode)

        new_metric = metric_from_full_eval(eval_info, best_metric_name, best_mode)
        if metric_better(new_metric, current_best, best_mode):
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
                "best_metric_name": best_metric_name,
                "best_mode": best_mode,
                "old_best_metric": current_best,
                "new_best_metric": new_metric,
                "new_eval_loss": eval_info.get("eval_loss"),
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
    # Keep root clean. All per-run artifacts live under <run_id>/.
    out_dir = out_root / run_id
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

    # Load validation samples once.  Mini eval uses a fixed cheap split
    # (usually val-mini.jsonl); full eval uses the complete validation split.
    data_cfg = cfg.get("data", {})
    eval_samples, eval_split = load_eval_samples(cfg, tokenizer, samples, data_cfg.get("eval_split", "val-mini"))
    full_eval_samples, full_eval_split = load_eval_samples(cfg, tokenizer, samples, data_cfg.get("full_eval_split", "val"))

    lr = float(cfg["training"].get("lr", 2e-6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(cfg["training"].get("weight_decay", 0.0)))
    batch_size = int(cfg["training"].get("batch_size", 1))
    steps = int(cfg["training"].get("steps", 1000))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))
    save_every = int(cfg["training"].get("save_every", 100))
    log_every = int(cfg["training"].get("log_every", 1))
    eval_every = int(cfg["training"].get("eval_every", 50) or 0)
    eval_limit = parse_eval_limit(
        cfg["training"].get("eval_limit", cfg.get("data", {}).get("eval_limit", 64)),
        default=int(cfg.get("data", {}).get("eval_limit", 64) or 64),
    )
    eval_limit_for_logs = eval_limit_label(eval_limit)
    full_eval_every = int(cfg["training"].get("full_eval_every", 0) or 0)
    full_eval_limit = parse_eval_limit(
        cfg["training"].get("full_eval_limit", "all"),
        default=int(cfg.get("data", {}).get("full_eval_limit", cfg.get("data", {}).get("eval_limit", 64)) or 64),
    )
    full_eval_limit_for_logs = eval_limit_label(full_eval_limit)
    best_metric_name = str(cfg["training"].get("best_metric", "full_eval_loss"))
    best_mode = str(cfg["training"].get("best_mode", "min")).lower()
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
        "eval_split": eval_split,
        "eval_limit": eval_limit_for_logs,
        "eval_every": eval_every,
        "full_eval_split": full_eval_split,
        "full_eval_limit": full_eval_limit_for_logs,
        "full_eval_every": full_eval_every,
        "best_metric_name": best_metric_name,
        "best_mode": best_mode,
        "device": str(device),
        "cli_gpu": cli_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "config": cfg,
    }
    run_info_path = out_dir / "run_info.json"
    dump_json(run_info_path, run_info)
    metrics_path = out_dir / "metrics.csv"
    debug_path = out_dir / "rollout_debug.jsonl"

    best_metric_value = initial_best_value(best_mode)
    best_step = 0
    best_row: Dict[str, Any] = {}
    best_eval_info: Dict[str, Any] = {}
    best_path = out_dir / "best_run.pth"
    best_mini_path = out_dir / "best_mini_run.pth"
    last_path = out_dir / "last_run.pth"
    best_mini_metric_value = initial_best_value(best_mode)
    best_mini_step = 0

    numeric_keys = [
        k for k in METRIC_FIELDS
        if k not in {
            "step", "loss", "lr", "best_metric_value", "is_best_eval",
            "mini_best_metric_value", "is_best_mini_eval", "is_full_eval",
        }
        and not k.startswith("eval_") and not k.startswith("full_eval_")
    ]
    last_debug_records: List[Dict[str, Any]] = []
    last_row: Dict[str, Any] = {}

    for step in range(1, steps + 1):
        model.train(True)
        optimizer.zero_grad(set_to_none=True)
        losses: List[torch.Tensor] = []
        agg = {k: 0.0 for k in numeric_keys}
        debug_records: List[Dict[str, Any]] = []
        valid = 0
        already_backward_count = 0
        logged_loss_values: List[float] = []
        for _ in range(batch_size):
            sample = next(sample_iter)
            try:
                loss_i, stats_i = compute_rl_loss(model, graph, noise, tokenizer, sample, cfg, device)
            except Exception as exc:
                print(f"[warn] skip sample {sample.get('id', '')}: {exc}", flush=True)
                continue
            if bool(stats_i.get("_already_backward", False)):
                already_backward_count += 1
                try:
                    logged_loss_values.append(float(loss_i.detach().item()))
                except Exception:
                    pass
            else:
                losses.append(loss_i)
                try:
                    logged_loss_values.append(float(loss_i.detach().item()))
                except Exception:
                    pass
            valid += 1
            for k in numeric_keys:
                if k in stats_i:
                    try:
                        agg[k] += float(stats_i.get(k, 0.0))
                    except Exception:
                        pass
            debug_records.extend(stats_i.get("debug_records", []) or [])
        if not losses and valid <= 0:
            continue

        if losses:
            loss = torch.stack(losses).mean()
            loss.backward()
            loss_value = float(loss.detach().item())
        else:
            # Memory-safe rollout already called backward per rollout step.
            loss_value = float(sum(logged_loss_values) / max(1, len(logged_loss_values)))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        ema.update(model.parameters())

        row = {"step": step, "loss": loss_value, "lr": lr}
        for k, v in agg.items():
            row[k] = v / max(1, valid)

        # Mini validation: fixed cheap subset, usually val-mini.jsonl.
        # It is useful for dense curves and for a fallback best_mini checkpoint.
        eval_now = eval_every > 0 and (step % eval_every == 0 or step == steps)
        if eval_now:
            eval_info_step = evaluate_loss(model, graph, noise, tokenizer, eval_samples, cfg, device, limit=eval_limit)
            eval_info_step.update({
                "eval_kind": "mini",
                "eval_split": eval_split,
                "eval_limit": eval_limit_for_logs,
                "step": step,
                "run_id": run_id,
                "best_metric_name": best_metric_name,
                "best_mode": best_mode,
            })
            add_eval_prefix(row, "eval", eval_info_step)
            mini_metric_value = get_metric_value(eval_info_step, "eval_loss", initial_best_value(best_mode))
            row["mini_best_metric_value"] = mini_metric_value
            improved_mini = metric_better(mini_metric_value, best_mini_metric_value, best_mode)
            row["is_best_mini_eval"] = 1.0 if improved_mini else 0.0
            append_jsonl(out_dir / "eval_history.jsonl", eval_info_step)
            if improved_mini:
                best_mini_metric_value = mini_metric_value
                best_mini_step = step
                save_checkpoint(
                    best_mini_path,
                    model,
                    ema,
                    optimizer,
                    cfg,
                    step,
                    {"best_metric_name": "eval_loss", "best_mode": best_mode, "best_metric_value": mini_metric_value, **row},
                )
                dump_json(out_dir / "best_mini_eval.json", eval_info_step)

        # Full validation: complete validation split. This is the official best_run criterion.
        full_eval_now = full_eval_every > 0 and (step % full_eval_every == 0 or step == steps)
        row["is_full_eval"] = 1.0 if full_eval_now else 0.0
        if full_eval_now:
            full_eval_info_step = evaluate_loss(model, graph, noise, tokenizer, full_eval_samples, cfg, device, limit=full_eval_limit)
            full_eval_info_step.update({
                "eval_kind": "full",
                "eval_split": full_eval_split,
                "eval_limit": full_eval_limit_for_logs,
                "step": step,
                "run_id": run_id,
                "best_metric_name": best_metric_name,
                "best_mode": best_mode,
            })
            full_eval_info_step["full_eval_loss"] = full_eval_info_step.get("eval_loss", float("inf"))
            add_eval_prefix(row, "full_eval", full_eval_info_step)
            metric_value = metric_from_full_eval(full_eval_info_step, best_metric_name, best_mode)
            row["best_metric_value"] = metric_value
            improved_eval = metric_better(metric_value, best_metric_value, best_mode)
            row["is_best_eval"] = 1.0 if improved_eval else 0.0
            append_jsonl(out_dir / "full_eval_history.jsonl", full_eval_info_step)
            if improved_eval:
                best_metric_value = metric_value
                best_step = step
                best_row = dict(row)
                best_eval_info = dict(full_eval_info_step)
                save_checkpoint(
                    best_path,
                    model,
                    ema,
                    optimizer,
                    cfg,
                    step,
                    {"best_metric_name": best_metric_name, "best_mode": best_mode, "best_metric_value": metric_value, **row},
                )
                dump_json(out_dir / "best_eval.json", best_eval_info)

        if eval_now or full_eval_now:
            eval_msg = f"mini_loss={row.get('eval_loss', float('nan')):.6g}" if row.get("eval_loss", "") != "" else "mini_loss=na"
            full_msg = f" full_loss={row.get('full_eval_loss', float('nan')):.6g}" if row.get("full_eval_loss", "") != "" else ""
            print(
                f"[eval step={step}] {eval_msg}{full_msg} "
                f"mini_count={row.get('eval_count', 0)} full_count={row.get('full_eval_count', 0)} "
                f"best_metric={best_metric_name} best={row.get('is_best_eval', 0.0)} mini_best={row.get('is_best_mini_eval', 0.0)}",
                flush=True,
            )

        last_row = row
        last_debug_records = debug_records or last_debug_records

        if step % log_every == 0:
            append_csv(metrics_path, row)
            eval_msg = ""
            if row.get("eval_loss", "") != "":
                eval_msg += f" eval_loss={float(row.get('eval_loss')):.4f}"
            if row.get("full_eval_loss", "") != "":
                eval_msg += f" full_eval_loss={float(row.get('full_eval_loss')):.4f}"
            print(
                f"step={step} loss={row['loss']:.4f} "
                f"target_logp={row.get('target_logp', 0.0):.4f} "
                f"modelR={row.get('model_reward', 0.0):+.3f} "
                f"Rstd={row.get('rollout_reward_std', 0.0):.3f} "
                f"entropy={row.get('rollout_entropy', row.get('candidate_entropy', 0.0)):.3f} "
                f"anchor={row.get('rollout_anchor_loss', 0.0):.4f} "
                f"targets={row.get('guided_targets', 0.0):.1f}"
                f"{eval_msg}",
                flush=True,
            )

        if debug_records:
            for rec in debug_records:
                append_jsonl(debug_path, {"step": step, **rec})
        if debug_every > 0 and step % debug_every == 0 and last_debug_records:
            print_debug_record(step, last_debug_records[0], max_rows=max_debug_rows)

        if save_every > 0 and step % save_every == 0:
            save_checkpoint(last_path, model, ema, optimizer, cfg, step, row)

    if not last_path.exists():
        save_checkpoint(last_path, model, ema, optimizer, cfg, steps, last_row or {"loss": float("inf")})

    # If sparse full eval was disabled or never produced a valid best, select by one final full-validation pass.
    if not best_path.exists():
        final_eval_for_best = evaluate_loss(model, graph, noise, tokenizer, full_eval_samples, cfg, device, limit=full_eval_limit)
        final_eval_for_best.update({
            "eval_kind": "full_final_fallback",
            "eval_split": full_eval_split,
            "eval_limit": full_eval_limit_for_logs,
            "step": steps,
            "run_id": run_id,
            "best_metric_name": best_metric_name,
            "best_mode": best_mode,
        })
        final_eval_for_best["full_eval_loss"] = final_eval_for_best.get("eval_loss", float("inf"))
        best_metric_value = metric_from_full_eval(final_eval_for_best, best_metric_name, best_mode)
        best_step = steps
        best_eval_info = dict(final_eval_for_best)
        best_row = dict(last_row or {})
        add_eval_prefix(best_row, "full_eval", final_eval_for_best)
        best_row.update({"best_metric_value": best_metric_value, "is_best_eval": 1.0})
        save_checkpoint(
            best_path,
            model,
            ema,
            optimizer,
            cfg,
            steps,
            {"best_metric_name": best_metric_name, "best_mode": best_mode, "best_metric_value": best_metric_value, **best_row},
        )
        dump_json(out_dir / "best_eval.json", best_eval_info)

    # Evaluate the run's validation-selected best checkpoint, then compete for root global best.
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
        print(f"[warn] failed to reload best checkpoint for final eval: {exc}", flush=True)

    eval_info = evaluate_loss(model, graph, noise, tokenizer, full_eval_samples, cfg, device, limit=full_eval_limit)
    eval_info.update({
        "eval_kind": "full_final_best_run",
        "eval_split": full_eval_split,
        "eval_limit": full_eval_limit_for_logs,
        "run_id": run_id,
        "best_step": best_step,
        "best_metric_name": best_metric_name,
        "best_mode": best_mode,
        "best_metric_value": metric_from_full_eval(eval_info, best_metric_name, best_mode),
    })
    eval_info["full_eval_loss"] = eval_info.get("eval_loss", float("inf"))
    best_metric_value = metric_from_full_eval(eval_info, best_metric_name, best_mode)

    # Cheap safeguard against missing a good step between sparse full evals:
    # also full-evaluate the best mini-eval checkpoint, then keep it if it wins.
    if best_mini_path.exists():
        try:
            state = torch.load(best_mini_path, map_location=device)
            model.load_state_dict(state.get("model", state), strict=True)
            if isinstance(state, dict) and "ema" in state:
                try:
                    ema.load_state_dict(state["ema"])
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                except Exception:
                    pass
            mini_full_eval = evaluate_loss(model, graph, noise, tokenizer, full_eval_samples, cfg, device, limit=full_eval_limit)
            mini_full_eval.update({
                "eval_kind": "full_final_best_mini_run",
                "eval_split": full_eval_split,
                "eval_limit": full_eval_limit_for_logs,
                "run_id": run_id,
                "best_step": best_mini_step,
                "best_metric_name": best_metric_name,
                "best_mode": best_mode,
            })
            mini_full_eval["full_eval_loss"] = mini_full_eval.get("eval_loss", float("inf"))
            mini_metric = metric_from_full_eval(mini_full_eval, best_metric_name, best_mode)
            mini_full_eval["best_metric_value"] = mini_metric
            dump_json(out_dir / "best_mini_full_eval.json", mini_full_eval)
            if metric_better(mini_metric, best_metric_value, best_mode):
                shutil.copy2(best_mini_path, best_path)
                best_metric_value = mini_metric
                best_step = best_mini_step
                eval_info = mini_full_eval
                best_eval_info = dict(mini_full_eval)
                dump_json(out_dir / "best_eval.json", best_eval_info)
                print(f"[final rerank] best_mini_run won on full eval: {best_metric_name}={mini_metric:.6g}", flush=True)
        except Exception as exc:
            print(f"[warn] failed to full-evaluate best_mini_run: {exc}", flush=True)

    eval_info["best_step"] = best_step
    eval_info["best_metric_value"] = best_metric_value
    dump_json(out_dir / "eval.json", eval_info)

    metrics_json = {
        "run_id": run_id,
        "run_dir": str(out_dir),
        "best_step": best_step,
        "best_metric_name": best_metric_name,
        "best_mode": best_mode,
        "best_metric_value": best_metric_value,
        "last_train_loss": float(last_row.get("loss", float("inf"))) if last_row else float("inf"),
        "eval_loss": float(eval_info.get("eval_loss", float("inf"))),
        "full_eval_loss": float(eval_info.get("full_eval_loss", eval_info.get("eval_loss", float("inf")))),
        "eval_split": full_eval_split,
        "eval_count": int(eval_info.get("eval_count", 0)),
        "mini_eval_split": eval_split,
        "mini_eval_count": len(eval_samples),
        "best_mini_step": best_mini_step,
        "last_row": last_row,
        "best_row": best_row,
        "best_eval_metrics": best_eval_info,
        "eval_metrics": eval_info,
    }
    dump_json(out_dir / "metrics.json", metrics_json)

    improved = update_global_best(
        out_root,
        out_dir,
        best_path,
        eval_info,
        metrics_json,
        run_info_path,
        metrics_path,
        best_metric_name=best_metric_name,
        best_mode=best_mode,
    )
    print(f"Done. Outputs: {out_dir}", flush=True)
    print(
        f"Run full_eval_loss={eval_info.get('full_eval_loss', eval_info.get('eval_loss'))} "
        f"{best_metric_name}={metric_from_full_eval(eval_info, best_metric_name, best_mode)} "
        f"best_step={best_step} improved_global_best={improved}",
        flush=True,
    )
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
