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
    "best_metric_value",
    "is_best_eval",
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

def cycle_random_batches(samples: List[Dict], seed: int, samples_per_update: int) -> Iterable[List[Dict]]:
    """Yield random per-update batches forever.

    This helper keeps make_update_batch_iter as a normal function that returns
    an iterable.  Do not inline a ``yield`` inside make_update_batch_iter,
    otherwise Python turns that function into a generator and ``return
    other_generator`` will immediately raise StopIteration with that generator
    as its value.
    """
    sample_iter = cycle_samples(samples, seed)
    while True:
        yield [next(sample_iter) for _ in range(max(1, int(samples_per_update)))]


DEFAULT_KIND_ORDER = [
    "interval",
    "decimal",
    "integer",
    "letter",
    "symbolic",
    "unit_decimal",
    "equation",
    "inequality",
]


def normalize_answer_kind(kind: Any) -> str:
    k = str(kind or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "single_integer": "integer",
        "int": "integer",
        "number": "integer",
        "single_letter": "letter",
        "choice": "letter",
        "multiple_choice": "letter",
        "unit": "unit_decimal",
        "unit_number": "unit_decimal",
        "units": "unit_decimal",
        "ineq": "inequality",
        "expr": "symbolic",
    }
    return aliases.get(k, k or "other")


def infer_answer_kind(sample: Dict[str, Any]) -> str:
    # Prefer explicit metadata if preparation code already tagged the answer.
    for key in ("answer_kind", "answer_type", "kind", "type"):
        if sample.get(key):
            return normalize_answer_kind(sample.get(key))
    for parent_key in ("meta", "base_meta", "answer_meta"):
        meta = sample.get(parent_key)
        if isinstance(meta, dict):
            for key in ("answer_kind", "answer_type", "kind", "type"):
                if meta.get(key):
                    return normalize_answer_kind(meta.get(key))

    # Synthetic ids usually contain the type name.
    sid = str(sample.get("id", "")).lower()
    for kind in DEFAULT_KIND_ORDER:
        if f"_{kind}_" in sid or sid.endswith(f"_{kind}"):
            return kind

    # Fallback heuristic from answer text.  This is only for sampling balance,
    # not for reward correctness.
    import re
    ans = str(sample.get("answer", "") or "").strip()
    compact = re.sub(r"\s+", "", ans)
    if re.fullmatch(r"[A-E]", compact):
        return "letter"
    if re.fullmatch(r"[\(\[][+-]?\d+(?:\.\d+)?,[+-]?\d+(?:\.\d+)?[\)\]]", compact):
        return "interval"
    if re.search(r"(?:<=|>=|<|>|\\leq|\\geq)", compact):
        return "inequality"
    if "=" in compact:
        return "equation"
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?[a-zA-Z\\/%°]+", compact) or re.fullmatch(r"[+-]?\d+(?:\.\d+)?[a-zA-Z].*", compact):
        return "unit_decimal"
    if re.fullmatch(r"[+-]?\d+", compact):
        return "integer"
    if re.fullmatch(r"[+-]?\d*\.\d+", compact):
        return "decimal"
    if any(ch in compact for ch in ["/", "^", "\\", "_", "{"]):
        return "symbolic"
    return "other"


def make_kind_buckets(samples: List[Dict[str, Any]], seed: int) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        kind = infer_answer_kind(sample)
        buckets.setdefault(kind, []).append(sample)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    return buckets


def cycle_kind_balanced_batches(
    samples: List[Dict[str, Any]],
    seed: int,
    samples_per_update: int,
    kind_order: List[str] | None = None,
) -> Iterable[List[Dict[str, Any]]]:
    """Yield update batches with approximately balanced answer kinds.

    For samples_per_update=8 and the default 8 kinds, this tries to draw
    one sample per kind.  For samples_per_update=16, it draws two rounds of
    the same kind order.  Missing/empty kinds are skipped and the remaining
    kinds are cycled.  Every selected sample has equal update weight later;
    this function only controls which samples enter the update.
    """
    rng = random.Random(seed)
    buckets = make_kind_buckets(samples, seed)
    configured = [normalize_answer_kind(k) for k in (kind_order or DEFAULT_KIND_ORDER)]
    ordered_kinds = [k for k in configured if buckets.get(k)]
    # Append any remaining detected kinds so they are not silently ignored.
    ordered_kinds.extend(k for k in sorted(buckets) if k not in ordered_kinds and buckets.get(k))
    if not ordered_kinds:
        random_iter = cycle_samples(samples, seed)
        while True:
            yield [next(random_iter) for _ in range(samples_per_update)]

    ptr = {k: 0 for k in ordered_kinds}
    offset = 0
    while True:
        batch: List[Dict[str, Any]] = []
        round_kinds = ordered_kinds[offset:] + ordered_kinds[:offset]
        while len(batch) < samples_per_update:
            for kind in round_kinds:
                bucket = buckets[kind]
                if ptr[kind] >= len(bucket):
                    rng.shuffle(bucket)
                    ptr[kind] = 0
                batch.append(bucket[ptr[kind]])
                ptr[kind] += 1
                if len(batch) >= samples_per_update:
                    break
        offset = (offset + 1) % max(1, len(ordered_kinds))
        yield batch



def _positive_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except Exception:
        return int(default)


def make_kind_quota_plan(train_cfg: Dict[str, Any], rollout_cfg: Dict[str, Any], samples_per_update: int) -> Tuple[List[str], Dict[str, int], str]:
    """Build a per-update kind quota plan.

    The quota sampler is meant for QRA refine: simple answer kinds should be
    over-represented and hard symbolic/equation/inequality samples should be
    capped.  The explicit sample_kind_quota is the source of truth.  If its
    total differs from samples_per_update, we scale/fill/truncate while
    respecting optional per-kind caps.
    """
    quota_cfg = train_cfg.get("sample_kind_quota", rollout_cfg.get("sample_kind_quota", {})) or {}
    weight_cfg = train_cfg.get("sample_kind_weights", rollout_cfg.get("sample_kind_weights", {})) or {}
    cap_cfg = train_cfg.get("sample_kind_max_per_update", rollout_cfg.get("sample_kind_max_per_update", {})) or {}
    kind_order = train_cfg.get("sample_kind_order", rollout_cfg.get("sample_kind_order", DEFAULT_KIND_ORDER)) or DEFAULT_KIND_ORDER
    kind_order = [normalize_answer_kind(k) for k in list(kind_order)]

    # Default simple-focused quota for 16 samples/update.  This intentionally
    # gives integer/decimal/letter more mass and caps hard kinds at 1 each.
    if not quota_cfg and not weight_cfg:
        quota_cfg = {
            "integer": 4,
            "decimal": 3,
            "letter": 3,
            "interval": 2,
            "unit_decimal": 1,
            "symbolic": 1,
            "equation": 1,
            "inequality": 1,
        }
        cap_cfg = {**{"symbolic": 1, "equation": 1, "inequality": 1}, **dict(cap_cfg)}

    quotas: Dict[str, int] = {}
    for k in kind_order:
        nk = normalize_answer_kind(k)
        if nk in quota_cfg:
            quotas[nk] = _positive_int(quota_cfg.get(nk), 0)
        elif nk in weight_cfg:
            quotas[nk] = _positive_int(weight_cfg.get(nk), 0)
        else:
            quotas[nk] = 0
    # Include any explicitly configured kinds that are not in sample_kind_order.
    for raw_k, raw_v in {**dict(weight_cfg), **dict(quota_cfg)}.items():
        nk = normalize_answer_kind(raw_k)
        if nk not in quotas:
            quotas[nk] = _positive_int(raw_v, 0)
            kind_order.append(nk)

    caps = {normalize_answer_kind(k): _positive_int(v, samples_per_update) for k, v in dict(cap_cfg).items()}

    # If the requested update size differs from the quota total, adjust using
    # weights from the same simple-first order.  This keeps 8/16/32 usable from
    # one config.
    total = sum(quotas.values())
    if total <= 0:
        quotas = {k: 1 for k in kind_order}
        total = sum(quotas.values())

    def can_add(k: str) -> bool:
        return quotas.get(k, 0) < caps.get(k, samples_per_update)

    if total < samples_per_update:
        # Fill extra slots by cycling simple-first kinds. Hard kinds with cap=1
        # will not grow unless the user raises their cap.
        idx = 0
        fill_order = [k for k in kind_order if k in quotas and quotas[k] > 0] or list(quotas)
        while sum(quotas.values()) < samples_per_update and fill_order:
            k = fill_order[idx % len(fill_order)]
            if can_add(k):
                quotas[k] += 1
            idx += 1
            if idx > samples_per_update * max(4, len(fill_order) * 4):
                # Caps may prevent filling; relax by adding to the first kind.
                quotas[fill_order[0]] += 1
    elif total > samples_per_update:
        # Truncate from the tail first, so hard/later kinds lose slots before
        # easy/front kinds. Keep one slot for a kind as long as possible.
        reduce_order = list(reversed(kind_order))
        idx = 0
        while sum(quotas.values()) > samples_per_update and reduce_order:
            k = reduce_order[idx % len(reduce_order)]
            if quotas.get(k, 0) > 0:
                quotas[k] -= 1
            idx += 1

    plan: List[str] = []
    for k in kind_order:
        plan.extend([k] * quotas.get(k, 0))
    # Fallback if a weird config produced too few entries.
    while len(plan) < samples_per_update:
        plan.append(kind_order[0] if kind_order else DEFAULT_KIND_ORDER[0])
    plan = plan[:samples_per_update]
    return plan, quotas, ",".join(f"{k}:{quotas.get(k,0)}" for k in kind_order if quotas.get(k, 0) > 0)


def cycle_kind_quota_batches(
    samples: List[Dict[str, Any]],
    seed: int,
    samples_per_update: int,
    cfg: Dict[str, Any],
) -> Iterable[List[Dict[str, Any]]]:
    """Yield per-update batches following explicit simple-focused quotas.

    This differs from kind_balanced: it can intentionally oversample easy kinds
    such as integer/decimal/letter and cap hard kinds such as equation and
    inequality to at most one sample per update.
    """
    rng = random.Random(seed)
    train_cfg = cfg.get("training", {}) or {}
    rollout_cfg = cfg.get("rollout", {}) or {}
    buckets = make_kind_buckets(samples, seed)
    plan, quotas, plan_label = make_kind_quota_plan(train_cfg, rollout_cfg, samples_per_update)

    # Keep known kinds in plan order, but if a requested kind is absent we will
    # substitute from available simple-first kinds instead of failing.
    fallback_order = [normalize_answer_kind(k) for k in (train_cfg.get("sample_fallback_order") or [
        "integer", "decimal", "letter", "interval", "unit_decimal", "symbolic", "equation", "inequality", "other"
    ])]
    fallback_order.extend(k for k in sorted(buckets) if k not in fallback_order)
    fallback_order = [k for k in fallback_order if buckets.get(k)]
    if not fallback_order:
        random_iter = cycle_samples(samples, seed)
        while True:
            yield [next(random_iter) for _ in range(samples_per_update)]

    ptr = {k: 0 for k in buckets}

    def draw_from_kind(kind: str) -> Dict[str, Any]:
        draw_kind = kind if buckets.get(kind) else None
        if draw_kind is None:
            # Substitute with the first available simple-first kind.
            draw_kind = fallback_order[0]
        bucket = buckets[draw_kind]
        if ptr[draw_kind] >= len(bucket):
            rng.shuffle(bucket)
            ptr[draw_kind] = 0
        item = bucket[ptr[draw_kind]]
        ptr[draw_kind] += 1
        return item

    step_offset = 0
    while True:
        # Rotate only within the same multiset quota so each kind gets the same
        # count per update but sample ordering varies a little.
        rotated = plan[step_offset:] + plan[:step_offset]
        batch = [draw_from_kind(k) for k in rotated[:samples_per_update]]
        step_offset = (step_offset + 1) % max(1, len(plan))
        yield batch

def make_update_batch_iter(samples: List[Dict[str, Any]], cfg: Dict[str, Any], seed: int, samples_per_update: int) -> Iterable[List[Dict[str, Any]]]:
    """Return an infinite iterable of per-update sample batches.

    Important: this function must NOT contain a ``yield`` statement.  If it
    does, Python treats the whole function as a generator; then statements like
    ``return cycle_kind_quota_batches(...)`` do not return that iterable to the
    caller, but instead stop the generator immediately.  That was the cause of
    the runtime error:

        StopIteration: <generator object cycle_kind_quota_batches ...>
    """
    train_cfg = cfg.get("training", {}) or {}
    rollout_cfg = cfg.get("rollout", {}) or {}
    selector = str(
        train_cfg.get(
            "sample_selector",
            train_cfg.get("sampler", rollout_cfg.get("sample_selector", rollout_cfg.get("sampler", "random"))),
        )
    ).strip().lower()
    selector = selector.replace("-", "_")
    if selector in {"kind_quota", "quota", "simple_focused", "easy_focused", "qra_simple_focused"}:
        return cycle_kind_quota_batches(samples, seed, samples_per_update, cfg)
    if selector in {"kind_balanced", "balanced", "answer_kind_balanced", "type_balanced"}:
        kind_order = train_cfg.get("sample_kind_order", rollout_cfg.get("sample_kind_order", DEFAULT_KIND_ORDER))
        return cycle_kind_balanced_batches(samples, seed, samples_per_update, list(kind_order or DEFAULT_KIND_ORDER))
    return cycle_random_batches(samples, seed, samples_per_update)


def summarize_batch_kinds(samples: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for sample in samples:
        kind = infer_answer_kind(sample)
        out[kind] = out.get(kind, 0) + 1
    return out


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

        new_metric = get_metric_value(eval_info, best_metric_name, get_metric_value(eval_info, "eval_loss", initial_best_value(best_mode)))
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
    # The update sampler is created after samples_per_update is resolved below.

    # Load validation samples once.  Best checkpoints are selected by mean validation metrics,
    # never by the single training sample loss from an update step.
    eval_samples, eval_split = load_eval_samples(cfg, tokenizer, samples)

    lr = float(cfg["training"].get("lr", 2e-6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(cfg["training"].get("weight_decay", 0.0)))
    train_cfg = cfg.get("training", {})
    # Effective number of different training samples used before one optimizer.step().
    # Keep backward compatibility: old configs can still use training.batch_size.
    samples_per_update = int(
        train_cfg.get(
            "samples_per_update",
            train_cfg.get("rl_samples_per_update", train_cfg.get("batch_size", 1)),
        )
    )
    samples_per_update = max(1, samples_per_update)
    # Split the effective update batch into smaller chunks for memory testing.
    # With rollout.memory_safe_replay=true this mostly controls logging/control flow,
    # because each rollout action is replayed/backwarded one by one.
    micro_batch_size = int(train_cfg.get("micro_batch_size", train_cfg.get("rl_micro_batch_size", samples_per_update)))
    micro_batch_size = max(1, min(samples_per_update, micro_batch_size))
    sample_selector = str(train_cfg.get("sample_selector", train_cfg.get("sampler", cfg.get("rollout", {}).get("sample_selector", "random")))).strip().lower()
    batch_iter = make_update_batch_iter(samples, cfg, seed, samples_per_update)
    # Backward-compatible alias used below.
    batch_size = samples_per_update
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
    best_metric_name = str(cfg["training"].get("best_metric", "eval_loss"))
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
        "samples_per_update": samples_per_update,
        "micro_batch_size": micro_batch_size,
        "sample_selector": sample_selector,
        "sample_kind_order": train_cfg.get("sample_kind_order", cfg.get("rollout", {}).get("sample_kind_order", DEFAULT_KIND_ORDER)),
        "data_split": split,
        "eval_split": eval_split,
        "eval_limit": eval_limit_for_logs,
        "eval_every": eval_every,
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
    last_path = out_dir / "last_run.pth"

    numeric_keys = [
        k for k in METRIC_FIELDS
        if k not in {"step", "loss", "lr", "best_metric_value", "is_best_eval"}
        and not k.startswith("eval_")
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
        try:
            update_samples = next(batch_iter)
        except StopIteration:
            # Defensive fallback: a batch iterator should be infinite.  If a
            # future sampler accidentally stops, recreate it instead of killing
            # a long server run.
            print("[warn] sample batch iterator stopped; recreating it", flush=True)
            batch_iter = make_update_batch_iter(samples, cfg, seed + step, samples_per_update)
            update_samples = next(batch_iter)
        batch_kind_counts = summarize_batch_kinds(update_samples)
        for micro_start in range(0, len(update_samples), micro_batch_size):
            micro_samples = update_samples[micro_start: micro_start + micro_batch_size]
            for sample in micro_samples:
                # For memory_safe_replay rollout_chain_loss calls backward internally.
                # This temporary config field keeps gradient magnitude invariant when
                # samples_per_update is changed from 1 -> 8/16/32.
                cfg_i = cfg
                if mode_for_debug in {"rollout", "rollout_chain", "chain"} and bool(rollout_cfg.get("memory_safe_replay", True)):
                    cfg_i = copy.deepcopy(cfg)
                    cfg_i.setdefault("rollout", {})["loss_normalizer"] = float(samples_per_update)
                    cfg_i["rollout"]["micro_batch_size"] = int(micro_batch_size)
                    cfg_i["rollout"]["samples_per_update"] = int(samples_per_update)
                try:
                    loss_i, stats_i = compute_rl_loss(model, graph, noise, tokenizer, sample, cfg_i, device)
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

        # Periodic validation.  This is deliberately not the single-sample training loss.
        # It is a mean over a fixed eval subset, so it is meaningful for best_run.pth and plots.
        eval_now = eval_every > 0 and (step % eval_every == 0 or step == steps)
        if eval_now:
            eval_info_step = evaluate_loss(model, graph, noise, tokenizer, eval_samples, cfg, device, limit=eval_limit)
            eval_info_step.update({
                "eval_split": eval_split,
                "eval_limit": eval_limit_for_logs,
                "step": step,
                "run_id": run_id,
                "best_metric_name": best_metric_name,
                "best_mode": best_mode,
            })
            row["eval_loss"] = float(eval_info_step.get("eval_loss", float("inf")))
            row["eval_count"] = int(eval_info_step.get("eval_count", 0))
            row["eval_rollout_loss"] = float(eval_info_step.get("rollout_loss", 0.0))
            row["eval_rollout_reward"] = float(eval_info_step.get("rollout_reward", 0.0))
            row["eval_rollout_reward_std"] = float(eval_info_step.get("rollout_reward_std", 0.0))
            row["eval_rollout_entropy"] = float(eval_info_step.get("rollout_entropy", 0.0))
            row["eval_rollout_logprob"] = float(eval_info_step.get("rollout_logprob", 0.0))
            row["eval_rollout_anchor_loss"] = float(eval_info_step.get("rollout_anchor_loss", 0.0))
            metric_value = get_metric_value(eval_info_step, best_metric_name, get_metric_value(eval_info_step, "eval_loss", initial_best_value(best_mode)))
            row["best_metric_value"] = metric_value
            improved_eval = metric_better(metric_value, best_metric_value, best_mode)
            row["is_best_eval"] = 1.0 if improved_eval else 0.0
            append_jsonl(out_dir / "eval_history.jsonl", eval_info_step)
            if improved_eval:
                best_metric_value = metric_value
                best_step = step
                best_row = dict(row)
                best_eval_info = dict(eval_info_step)
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
            print(
                f"[eval step={step}] {best_metric_name}={metric_value:.6g} mode={best_mode} "
                f"eval_loss={row['eval_loss']:.6g} evalR={row.get('eval_rollout_reward', 0.0):+.3f} "
                f"count={row.get('eval_count', 0)} best={'yes' if improved_eval else 'no'}",
                flush=True,
            )

        last_row = row
        last_debug_records = debug_records or last_debug_records

        if step % log_every == 0:
            append_csv(metrics_path, row)
            eval_msg = ""
            if row.get("eval_loss", "") != "":
                eval_msg = f" eval_loss={float(row.get('eval_loss')):.4f}"
            print(
                f"step={step} loss={row['loss']:.4f} "
                f"batch={valid}/{samples_per_update} micro={micro_batch_size} "
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

    # If periodic eval was disabled or never produced a valid best, select by one final validation pass.
    if not best_path.exists():
        final_eval_for_best = evaluate_loss(model, graph, noise, tokenizer, eval_samples, cfg, device, limit=eval_limit)
        final_eval_for_best.update({
            "eval_split": eval_split,
            "eval_limit": eval_limit_for_logs,
            "step": steps,
            "run_id": run_id,
            "best_metric_name": best_metric_name,
            "best_mode": best_mode,
        })
        best_metric_value = get_metric_value(
            final_eval_for_best,
            best_metric_name,
            get_metric_value(final_eval_for_best, "eval_loss", initial_best_value(best_mode)),
        )
        best_step = steps
        best_eval_info = dict(final_eval_for_best)
        best_row = dict(last_row or {})
        best_row.update({
            "eval_loss": final_eval_for_best.get("eval_loss"),
            "eval_count": final_eval_for_best.get("eval_count"),
            "best_metric_value": best_metric_value,
            "is_best_eval": 1.0,
        })
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

    eval_info = evaluate_loss(model, graph, noise, tokenizer, eval_samples, cfg, device, limit=eval_limit)
    eval_info.update({
        "eval_split": eval_split,
        "eval_limit": eval_limit_for_logs,
        "run_id": run_id,
        "best_step": best_step,
        "best_metric_name": best_metric_name,
        "best_mode": best_mode,
        "best_metric_value": best_metric_value,
    })
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
        "eval_split": eval_split,
        "eval_count": int(eval_info.get("eval_count", 0)),
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
        f"Run eval_loss={eval_info.get('eval_loss')} "
        f"{best_metric_name}={eval_info.get(best_metric_name, best_metric_value)} "
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
