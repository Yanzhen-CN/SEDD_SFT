from __future__ import annotations

"""Compare SEDD/QRA/RL answer-mask recovery chains.

Default comparison models are exactly six:
    pretrain_start, pretrain_loss_best, pretrain_reward_best,
    QRA_start, QRA_loss_best, QRA_reward_best

For each start family we compare: starting model, RL loss-best checkpoint,
and RL reward-best checkpoint.  The reward-best checkpoint is best_reward.pth;
if it is missing, the script falls back to the newest per-run last_run.pth and
records that fallback in model_plan.json/run.log.

Default output:
    experiment/sample_chain/<timestamp>_<run_name>/
        chain.csv                         # all samples / all models / all steps
        chains/<sample_id>_chain.csv       # one CSV per sample, all models in that sample
        run.log                           # terminal-style comparison log
        sample_report.txt                 # question/reasoning/GT + final answer of every model

This script fixes question/reasoning/prompt positions and starts the answer segment as [MASK]
(or □ in the visible output). It is a diagnostic chain generator for QRA-style answers.
"""

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import yaml
from transformers import GPT2TokenizerFast

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
for p in (REPO_DIR, SCRIPT_DIR, REPO_DIR / "sft_answer_pipeline", REPO_DIR / "sft_rl_pipeline"):
    p_str = str(p)
    if p.exists() and p_str not in sys.path:
        sys.path.insert(0, p_str)

import graph_lib  # noqa: E402
import noise_lib  # noqa: E402
from model import SEDD  # noqa: E402
from model.ema import ExponentialMovingAverage  # noqa: E402
from answer_dataset import AnswerSegmentDataset, ordered_segments  # noqa: E402
from state_builder import (  # noqa: E402
    encode_sample,
    mask_id_from_graph,
    project_fixed_,
    transition_probs,
)

DEFAULT_CONFIG = SCRIPT_DIR / "rl_qra_config.yaml"
DEFAULT_MODELS = [
    "pretrain_start",
    "pretrain_loss_best",
    "pretrain_reward_best",
    "QRA_start",
    "QRA_loss_best",
    "QRA_reward_best",
]


# ----------------------------- basic utilities -----------------------------

def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_DIR / p


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_int(text: str, mod: int = 10_000_000) -> int:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:12], 16) % mod


def choose_device(args: argparse.Namespace, cfg: Dict[str, Any]) -> torch.device:
    if args.cpu or bool(cfg.get("cpu", False)) or not torch.cuda.is_available():
        return torch.device("cpu")
    if args.gpu is not None:
        return torch.device(f"cuda:{int(args.gpu)}")
    cfg_gpu = (cfg.get("run") or {}).get("cuda_device", None)
    if cfg_gpu is None or str(cfg_gpu).lower() in {"", "none", "null", "auto", "cuda"}:
        return torch.device("cuda")
    return torch.device(f"cuda:{int(cfg_gpu)}")


def normalize_text(x: str) -> str:
    x = str(x or "")
    x = x.replace("\\mathrm{~m}", " m")
    x = x.replace("\\mathrm{m}", "m")
    x = x.replace("\\left", "").replace("\\right", "")
    x = re.sub(r"\s+", "", x)
    if len(x) > 1 and x[-1] in ".;":
        x = x[:-1]
    return x.strip()


def safe_name(text: str, fallback: str) -> str:
    s = str(text or fallback)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")
    return s[:120] or fallback


def short(s: str, n: int = 240) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[: n - 3] + "..."


class TeeLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")

    def print(self, *args, **kwargs) -> None:
        text = " ".join(str(a) for a in args)
        print(text, **kwargs)
        self.f.write(text + ("\n" if not text.endswith("\n") else ""))
        self.f.flush()

    def close(self) -> None:
        self.f.close()


# ---------------------------- config resolution ----------------------------

def selected_start_name(cfg: Dict[str, Any], cli_start: Optional[str]) -> str:
    return str(cli_start or (cfg.get("run") or {}).get("selected") or cfg.get("selected") or "QRA")


def selected_start_cfg(cfg: Dict[str, Any], start_name: str) -> Dict[str, Any]:
    return ((cfg.get("starts") or {}).get(start_name) or {})


def resolve_data_dir(cfg: Dict[str, Any], start_name: str) -> Path:
    data_cfg = cfg.get("data") or {}
    start_cfg = selected_start_cfg(cfg, start_name)
    data_dir = data_cfg.get("data_dir") or start_cfg.get("data_dir") or "rl_qra_pipeline/data/S1K_RL"
    return repo_path(data_dir)


def start_output_dir(cfg: Dict[str, Any], start_name: str) -> Path:
    start_cfg = selected_start_cfg(cfg, start_name)
    return repo_path(start_cfg.get("output_dir", f"rl_qra_pipeline/modelparameter/rl_{start_name}"))


def latest_run_checkpoint(run_root: Path, filename: str = "last_run.pth") -> Optional[Path]:
    """Return the newest per-run checkpoint under rl_<start>/<run_id>/filename.

    Root best.pth is intentionally not considered here.  This is for comparing
    the validation-selected best checkpoint against the latest training-state
    checkpoint from individual runs.
    """
    if not run_root.exists():
        return None
    candidates: List[Path] = []
    for p in run_root.iterdir():
        if not p.is_dir():
            continue
        ckpt = p / filename
        if ckpt.exists():
            candidates.append(ckpt)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def init_checkpoint_path(cfg: Dict[str, Any], start_name: str) -> Optional[Path]:
    start_cfg = selected_start_cfg(cfg, start_name)
    value = start_cfg.get("init_checkpoint", "")
    if value is None or str(value).strip().lower() in {"", "none", "null", "pretrained", "hf", "cache"}:
        return None
    path = repo_path(value)
    return path if path.exists() else None


def model_pretrained_name(cfg: Dict[str, Any], start_name: str = "pretrain") -> str:
    start_cfg = selected_start_cfg(cfg, start_name)
    return str(start_cfg.get("pretrained") or (cfg.get("model") or {}).get("pretrained") or "louaaron/sedd-medium")


def reward_checkpoint_or_fallback(root: Path) -> Tuple[Optional[Path], str, Optional[str]]:
    """Return root best_reward.pth, or newest run last_run.pth as fallback.

    The fallback is only for old runs before reward-best recovery existed.
    A warning string is returned so model_plan.json/run.log makes this explicit.
    """
    ckpt = root / "best_reward.pth"
    if ckpt.exists():
        return ckpt, "rl_root_best_reward", None
    fallback = latest_run_checkpoint(root, "last_run.pth")
    if fallback is not None:
        return fallback, "latest_run_last_fallback_missing_best_reward", f"missing {ckpt}; fallback to newest run last_run.pth"
    return None, "missing", f"missing {ckpt} and no */last_run.pth under {root}"


def resolve_model_plan(args: argparse.Namespace, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return model specs with fields: name, checkpoint, pretrained, source.

    The default comparison is exactly six models:
      pretrain_start        = pretrained/cache SEDD start
      pretrain_loss_best    = rl_pretrain/best.pth
      pretrain_reward_best  = rl_pretrain/best_reward.pth
      QRA_start             = starts.QRA.init_checkpoint
      QRA_loss_best         = rl_QRA/best.pth
      QRA_reward_best       = rl_QRA/best_reward.pth
    """
    if args.checkpoint:
        return [{
            "name": args.checkpoint_name or "model",
            "checkpoint": repo_path(args.checkpoint),
            "pretrained": model_pretrained_name(cfg, "pretrain"),
            "source": "manual_checkpoint",
        }]

    requested = [x.strip() for x in str(args.models).split(",") if x.strip()]
    specs: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for raw_name in requested:
        name = raw_name.strip()
        name_l = name.lower()

        # ----- pretrain family: start / loss-best / reward-best -----
        if name_l in {"pretrain", "pretrain_start", "start_pretrain"}:
            specs.append({
                "name": "pretrain_start",
                "checkpoint": None,
                "pretrained": model_pretrained_name(cfg, "pretrain"),
                "source": "start_pretrained_or_cache",
            })
            continue

        if name_l in {"pretrain_loss_best", "pretrain_best", "rl_pretrain", "rl_pretrain_best"}:
            ckpt = start_output_dir(cfg, "pretrain") / "best.pth"
            if ckpt.exists():
                specs.append({
                    "name": "pretrain_loss_best",
                    "checkpoint": ckpt,
                    "pretrained": model_pretrained_name(cfg, "pretrain"),
                    "source": "rl_root_loss_best",
                })
            else:
                warnings.append(f"skip pretrain_loss_best: missing {ckpt}")
            continue

        if name_l in {"pretrain_reward_best", "pretrain_best_reward", "rl_pretrain_reward", "rl_pretrain_last"}:
            root = start_output_dir(cfg, "pretrain")
            ckpt, source, warning = reward_checkpoint_or_fallback(root)
            if ckpt is not None:
                specs.append({
                    "name": "pretrain_reward_best",
                    "checkpoint": ckpt,
                    "pretrained": model_pretrained_name(cfg, "pretrain"),
                    "source": source,
                })
            warnings.append(f"pretrain_reward_best: {warning}" if warning else "")
            continue

        # ----- QRA family: start / loss-best / reward-best -----
        if name_l in {"qra", "qra_start", "start_qra"}:
            ckpt = init_checkpoint_path(cfg, "QRA")
            if ckpt is not None:
                specs.append({
                    "name": "QRA_start",
                    "checkpoint": ckpt,
                    "pretrained": model_pretrained_name(cfg, "QRA"),
                    "source": "start_init_checkpoint",
                })
            else:
                warnings.append("skip QRA_start: starts.QRA.init_checkpoint missing or not found")
            continue

        if name_l in {"qra_loss_best", "qra_best", "rl_qra", "rl_qra_best"}:
            ckpt = start_output_dir(cfg, "QRA") / "best.pth"
            if ckpt.exists():
                specs.append({
                    "name": "QRA_loss_best",
                    "checkpoint": ckpt,
                    "pretrained": model_pretrained_name(cfg, "QRA"),
                    "source": "rl_root_loss_best",
                })
            else:
                warnings.append(f"skip QRA_loss_best: missing {ckpt}")
            continue

        if name_l in {"qra_reward_best", "qra_best_reward", "rl_qra_reward", "rl_qra_last"}:
            root = start_output_dir(cfg, "QRA")
            ckpt, source, warning = reward_checkpoint_or_fallback(root)
            if ckpt is not None:
                specs.append({
                    "name": "QRA_reward_best",
                    "checkpoint": ckpt,
                    "pretrained": model_pretrained_name(cfg, "QRA"),
                    "source": source,
                })
            warnings.append(f"QRA_reward_best: {warning}" if warning else "")
            continue

        # ----- optional QA aliases kept for manual calls -----
        if name_l in {"qa", "qa_start"}:
            ckpt = init_checkpoint_path(cfg, "QA")
            if ckpt is not None:
                specs.append({"name": "QA_start", "checkpoint": ckpt, "pretrained": model_pretrained_name(cfg, "QA"), "source": "start_init_checkpoint"})
            else:
                warnings.append("skip QA_start: starts.QA.init_checkpoint missing or not found")
            continue

        if name_l in {"qa_loss_best", "qa_best", "rl_qa", "rl_qa_best"}:
            ckpt = start_output_dir(cfg, "QA") / "best.pth"
            if ckpt.exists():
                specs.append({"name": "QA_loss_best", "checkpoint": ckpt, "pretrained": model_pretrained_name(cfg, "QA"), "source": "rl_root_loss_best"})
            else:
                warnings.append(f"skip QA_loss_best: missing {ckpt}")
            continue

        if name_l in {"qa_reward_best", "qa_best_reward", "rl_qa_reward", "rl_qa_last"}:
            root = start_output_dir(cfg, "QA")
            ckpt, source, warning = reward_checkpoint_or_fallback(root)
            if ckpt is not None:
                specs.append({"name": "QA_reward_best", "checkpoint": ckpt, "pretrained": model_pretrained_name(cfg, "QA"), "source": source})
            warnings.append(f"QA_reward_best: {warning}" if warning else "")
            continue

        # Treat as label:path for a custom checkpoint.
        if ":" in name:
            label, raw_path = name.split(":", 1)
            ckpt = repo_path(raw_path)
            if ckpt.exists():
                specs.append({"name": label, "checkpoint": ckpt, "pretrained": model_pretrained_name(cfg), "source": "custom"})
            else:
                warnings.append(f"skip {label}: missing {ckpt}")
        else:
            warnings.append(f"skip unknown model spec: {name}")

    for w in warnings:
        if w:
            print(f"[warn] {w}", flush=True)
    if not specs:
        raise FileNotFoundError("No model checkpoints/pretrained specs available. Check root best.pth/best_reward.pth files or pass --checkpoint.")
    return specs


# ------------------------------ sample helpers -----------------------------

def get_segment_text(sample: Dict[str, Any], names: Sequence[str]) -> str:
    segs = sample.get("segments") or {}
    for name in names:
        if name in segs and isinstance(segs[name], dict):
            text = str(segs[name].get("text", ""))
            if text.strip():
                return text.strip()
    return ""


def extract_gt_answer(sample: Dict[str, Any]) -> str:
    for key in ("answer", "solution", "final_answer", "target"):
        val = sample.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    text = get_segment_text(sample, ["answer", "final_answer", "target"])
    if text:
        return text.strip()
    train_parts = []
    for _, seg in ordered_segments(sample):
        if isinstance(seg, dict) and bool(seg.get("train", False)):
            train_parts.append(str(seg.get("text", "")))
    if train_parts:
        return "".join(train_parts).strip()
    return ""


def extract_question_reasoning(sample: Dict[str, Any]) -> Tuple[str, str]:
    q = sample.get("question") or get_segment_text(sample, ["user", "question", "prompt"])
    r = sample.get("reasoning") or get_segment_text(sample, ["reasoning", "rationale", "thinking"])
    if not r:
        pieces = []
        for name, seg in ordered_segments(sample):
            name_l = str(name).lower()
            if "answer" in name_l:
                break
            text = str(seg.get("text", "")) if isinstance(seg, dict) else ""
            if "reason" in name_l or "assistant" in name_l:
                pieces.append(text)
        r = "".join(pieces).strip()
    return str(q or "").strip(), str(r or "").strip()


def answer_kind(answer: str) -> str:
    a = normalize_text(answer)
    c = a.replace(" ", "")
    if re.fullmatch(r"[A-Ea-e]", c):
        return "letter"
    if re.fullmatch(r"[+-]?\d+", c):
        return "integer"
    if re.fullmatch(r"[+-]?(?:\d+\.\d+|\.\d+)", c):
        return "decimal"
    if re.fullmatch(r"[\(\[][+-]?(?:\d+(?:\.\d+)?|\.\d+),[+-]?(?:\d+(?:\.\d+)?|\.\d+)[\)\]]", c):
        return "interval"
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*=.+", c):
        return "equation"
    if re.fullmatch(r"[+-]?(?:\d+\.\d+|\.\d+|\d+)(m|mm|cm|km|kg|g|s|ms|N|J|W|V|A|Hz|m/s|m/s\^2)", c):
        return "unit_decimal"
    if any(sym in c for sym in ["^", "/", "sqrt", "pi", "\\sqrt", "\\frac", "*", "+", "-"]) and re.search(r"[A-Za-z0-9]", c):
        return "symbolic"
    return "short_text"


def load_samples(cfg: Dict[str, Any], start_name: str, split: str, tokenizer) -> List[Dict[str, Any]]:
    data_dir = resolve_data_dir(cfg, start_name)
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Data split not found: {path}")
    model_cfg = cfg.get("model") or {}
    max_len = int(model_cfg.get("max_length", 1024))
    ds = AnswerSegmentDataset(
        path,
        tokenizer,
        max_len,
        min_target_tokens=int(model_cfg.get("min_target_tokens", 1)),
        drop_overlength=bool(model_cfg.get("drop_overlength", True)),
        write_report=False,
    )
    return ds.samples


def pick_diverse_samples(
    samples: List[Dict[str, Any]],
    sample_ids: Sequence[str],
    num_samples: int,
    per_type: int,
) -> List[Dict[str, Any]]:
    if sample_ids:
        wanted = set(sample_ids)
        chosen = [s for s in samples if str(s.get("id", "")) in wanted]
        missing = wanted - {str(s.get("id", "")) for s in chosen}
        if missing:
            print(f"[warn] missing sample ids: {sorted(missing)}", flush=True)
        return chosen[:num_samples]

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        ans = extract_gt_answer(s)
        buckets[answer_kind(ans)].append(s)

    priority = ["interval", "equation", "unit_decimal", "symbolic", "decimal", "integer", "letter", "short_text"]
    chosen: List[Dict[str, Any]] = []
    seen = set()
    for k in priority:
        for s in buckets.get(k, [])[:per_type]:
            sid = str(s.get("id", id(s)))
            if sid not in seen:
                chosen.append(s)
                seen.add(sid)
            if len(chosen) >= num_samples:
                return chosen

    for s in samples:
        sid = str(s.get("id", id(s)))
        if sid not in seen:
            chosen.append(s)
            seen.add(sid)
        if len(chosen) >= num_samples:
            break
    return chosen


# ------------------------------- model load --------------------------------

def load_model_for_trace(cfg: Dict[str, Any], spec: Dict[str, Any], device: torch.device):
    model_cfg = cfg.get("model") or {}
    training_cfg = cfg.get("training") or {}
    pretrained = str(spec.get("pretrained") or model_cfg.get("pretrained") or "louaaron/sedd-medium")
    model = SEDD.from_pretrained(pretrained).to(device)
    model.config.model.length = int(model_cfg.get("max_length", getattr(model.config.model, "length", 1024)))

    checkpoint = spec.get("checkpoint")
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        state = torch.load(checkpoint, map_location=device)
        if isinstance(state, dict) and "model" in state:
            model_state = state["model"]
        elif isinstance(state, dict) and "model_state_dict" in state:
            model_state = state["model_state_dict"]
        else:
            model_state = state
        model.load_state_dict(model_state, strict=True)

        # Use EMA weights if available, same as eval/training convention.
        if isinstance(state, dict) and "ema" in state:
            try:
                ema = ExponentialMovingAverage(model.parameters(), decay=float(training_cfg.get("ema", 0.9999)))
                ema.load_state_dict(state["ema"])
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
            except Exception as exc:
                print(f"[warn] failed to apply EMA from {checkpoint}: {exc}", flush=True)

    graph = graph_lib.get_graph(model.config, device)
    noise = noise_lib.get_noise(model.config).to(device)
    model.eval()
    return model, graph, noise


# ------------------------------- trace score -------------------------------

def visible_answer_tokens(tokenizer, ids: Sequence[int], mask_id: int, vocab_size: int) -> Tuple[str, List[str]]:
    pieces = []
    token_texts = []
    for tok in ids:
        ti = int(tok)
        if ti == int(mask_id) or ti < 0 or ti >= vocab_size:
            pieces.append("□")
            token_texts.append("□")
        else:
            txt = tokenizer.decode([ti])
            pieces.append(txt)
            token_texts.append(txt)
    return "".join(pieces), token_texts


def score_state(curr_ids: Sequence[int], gt_ids: Sequence[int], mask_id: int, t_value: float) -> Dict[str, float]:
    n = max(1, len(gt_ids))
    curr = [int(x) for x in curr_ids[: len(gt_ids)]]
    gt = [int(x) for x in gt_ids]
    mask_count = sum(1 for x in curr if x == mask_id or x < 0)
    filled = len(curr) - mask_count
    pos_match = sum(1 for x, y in zip(curr, gt) if x == y) / n

    gt_counter = Counter(gt)
    overlap = 0
    for x in curr:
        if x == mask_id or x < 0:
            continue
        if gt_counter[x] > 0:
            overlap += 1
            gt_counter[x] -= 1
    token_set = overlap / n
    exact = 1.0 if len(curr) == len(gt) and all(x == y for x, y in zip(curr, gt)) else 0.0
    fill_ratio = filled / n

    if t_value > 0.70:
        total = 0.45 * token_set + 0.30 * fill_ratio + 0.20 * pos_match + 0.05 * exact
    elif t_value > 0.30:
        total = 0.35 * pos_match + 0.35 * token_set + 0.20 * fill_ratio + 0.10 * exact
    else:
        total = 0.50 * exact + 0.30 * pos_match + 0.15 * token_set + 0.05 * fill_ratio

    return {
        "mask_count": float(mask_count),
        "filled_ratio": float(fill_ratio),
        "position_match": float(pos_match),
        "token_set_match": float(token_set),
        "exact_token_match": float(exact),
        "stage_score": float(total),
    }


# ------------------------------- trace core --------------------------------

def trace_one_sample(
    model,
    graph,
    noise,
    tokenizer,
    sample: Dict[str, Any],
    cfg: Dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    model_name: str,
) -> Dict[str, Any]:
    model_cfg = cfg.get("model") or {}
    max_length = int(model_cfg.get("max_length", 1024))
    encoded = encode_sample(sample, tokenizer, max_length, tokenizer.eos_token_id)
    answer_pos = [p for p in encoded.layout.answer_positions if p < encoded.layout.real_len]
    if not answer_pos:
        raise ValueError(f"sample {sample.get('id', '')} has no answer positions")

    mask_id = int(mask_id_from_graph(graph))
    vocab_size = int(getattr(tokenizer, "vocab_size", 50257))
    gt_ids = [int(encoded.ids[p]) for p in answer_pos]
    gt_answer = extract_gt_answer(sample)
    question, reasoning = extract_question_reasoning(sample)

    x = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    x[:, answer_pos] = mask_id
    project_fixed_(x, encoded.layout.fixed_locs, encoded.layout.fixed_ids.to(device))

    num_steps = int(args.steps)
    t_start = float(args.t_start)
    t_end = float(args.t_end)
    dt_step = (t_start - t_end) / max(1, num_steps)
    transition_kind = str(args.transition_kind)

    trace_rows: List[Dict[str, Any]] = []
    prev_score = None
    prev_answer_ids = [int(x[0, p].item()) for p in answer_pos]

    with torch.no_grad():
        for step in range(num_steps + 1):
            t_val = max(t_end, t_start - step * dt_step)
            curr_ids = [int(x[0, p].item()) for p in answer_pos]
            ans_text, token_texts = visible_answer_tokens(tokenizer, curr_ids, mask_id, vocab_size)
            score = score_state(curr_ids, gt_ids, mask_id, t_val)
            score_delta = 0.0 if prev_score is None else score["stage_score"] - prev_score
            changed = [i for i, (a, b) in enumerate(zip(prev_answer_ids, curr_ids)) if a != b]

            record: Dict[str, Any] = {
                "model": model_name,
                "sample_id": sample.get("id", ""),
                "answer_kind": answer_kind(gt_answer),
                "step": int(step),
                "t": float(t_val),
                "answer_text": ans_text,
                "answer_token_texts": token_texts,
                "answer_token_ids": curr_ids,
                "gt_answer": gt_answer,
                "gt_answer_token_ids": gt_ids,
                "changed_answer_indices": changed,
                "stage_score_delta": float(score_delta),
                **score,
            }
            trace_rows.append(record)

            if step == num_steps:
                break

            t_tensor = torch.tensor([t_val], dtype=torch.float32, device=device)
            probs = transition_probs(
                model,
                graph,
                noise,
                x,
                t_tensor,
                dt_step,
                kind=transition_kind,
                train=False,
                fixed_locs=encoded.layout.fixed_locs,
                fixed_ids=encoded.layout.fixed_ids.to(device),
            )

            prev_score = score["stage_score"]
            prev_answer_ids = curr_ids

            if args.mode == "greedy":
                next_x = probs.argmax(dim=-1)
            else:
                flat = probs.view(-1, probs.shape[-1])
                sampled = torch.multinomial(flat.float().clamp_min(0.0), num_samples=1).view_as(x)
                next_x = sampled

            if args.freeze_filled:
                keep = torch.zeros_like(x, dtype=torch.bool)
                for p in answer_pos:
                    keep[:, p] = x[:, p] != mask_id
                next_x = torch.where(keep, x, next_x)

            x = next_x.to(device)
            project_fixed_(x, encoded.layout.fixed_locs, encoded.layout.fixed_ids.to(device))

    return {
        "model": model_name,
        "sample_id": sample.get("id", ""),
        "answer_kind": answer_kind(gt_answer),
        "question": question,
        "reasoning": reasoning,
        "gt_answer": gt_answer,
        "answer_positions": answer_pos,
        "mask_id": mask_id,
        "num_steps": num_steps,
        "t_start": t_start,
        "t_end": t_end,
        "mode": args.mode,
        "freeze_filled": bool(args.freeze_filled),
        "trace": trace_rows,
    }


# ------------------------------- output helpers -----------------------------

CHAIN_FIELDS = [
    "sample_index",
    "sample_id",
    "answer_kind",
    "model",
    "step",
    "t",
    "answer",
    "gt_answer",
    "exact_text_match",
    "exact_token_match",
    "mask_count",
    "filled_ratio",
    "position_match",
    "token_set_match",
    "stage_score",
    "stage_score_delta",
    "changed_answer_indices",
    "answer_token_texts",
    "answer_token_ids",
    "gt_answer_token_ids",
]


def result_to_chain_rows(result: Dict[str, Any], sample_index: int) -> List[Dict[str, Any]]:
    rows = []
    gt_norm = normalize_text(result.get("gt_answer", ""))
    for tr in result.get("trace", []):
        ans = str(tr.get("answer_text", ""))
        rows.append({
            "sample_index": sample_index,
            "sample_id": result.get("sample_id", ""),
            "answer_kind": result.get("answer_kind", ""),
            "model": result.get("model", ""),
            "step": tr.get("step"),
            "t": f"{float(tr.get('t', 0.0)):.6f}",
            "answer": ans,
            "gt_answer": result.get("gt_answer", ""),
            "exact_text_match": 1 if normalize_text(ans) == gt_norm else 0,
            "exact_token_match": int(float(tr.get("exact_token_match", 0.0))),
            "mask_count": int(float(tr.get("mask_count", 0.0))),
            "filled_ratio": f"{float(tr.get('filled_ratio', 0.0)):.6f}",
            "position_match": f"{float(tr.get('position_match', 0.0)):.6f}",
            "token_set_match": f"{float(tr.get('token_set_match', 0.0)):.6f}",
            "stage_score": f"{float(tr.get('stage_score', 0.0)):.6f}",
            "stage_score_delta": f"{float(tr.get('stage_score_delta', 0.0)):.6f}",
            "changed_answer_indices": json.dumps(tr.get("changed_answer_indices", []), ensure_ascii=False),
            "answer_token_texts": json.dumps(tr.get("answer_token_texts", []), ensure_ascii=False),
            "answer_token_ids": json.dumps(tr.get("answer_token_ids", []), ensure_ascii=False),
            "gt_answer_token_ids": json.dumps(tr.get("gt_answer_token_ids", []), ensure_ascii=False),
        })
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str] = CHAIN_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def final_answer(result: Dict[str, Any]) -> str:
    trace = result.get("trace") or []
    if not trace:
        return ""
    return str(trace[-1].get("answer_text", ""))


def final_exact(result: Dict[str, Any]) -> bool:
    return normalize_text(final_answer(result)) == normalize_text(str(result.get("gt_answer", "")))


def write_sample_report(path: Path, sample_order: List[Dict[str, Any]], results_by_sample: Dict[str, Dict[str, Dict[str, Any]]], model_names: List[str]) -> None:
    lines: List[str] = []
    for idx, sample in enumerate(sample_order, start=1):
        sid = str(sample.get("id", ""))
        q, r = extract_question_reasoning(sample)
        gt = extract_gt_answer(sample)
        lines.append(f"sample{idx}:")
        lines.append(f"id: {sid}")
        lines.append("question:")
        lines.append(q)
        lines.append("reasoning:")
        lines.append(r)
        lines.append("answer:")
        lines.append(gt)
        for model_name in model_names:
            res = results_by_sample.get(sid, {}).get(model_name)
            if not res:
                lines.append(f"{model_name}:")
                lines.append("<missing>")
            elif "error" in res:
                lines.append(f"{model_name}:")
                lines.append(f"ERROR: {res.get('error')}")
            else:
                ans = final_answer(res)
                mark = "✓" if final_exact(res) else "✗"
                lines.append(f"{model_name}:")
                lines.append(f"{ans}    [{mark}]")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def log_sample_comparison(logger: TeeLogger, idx: int, sample: Dict[str, Any], results_by_model: Dict[str, Dict[str, Any]], model_names: List[str]) -> None:
    sid = str(sample.get("id", ""))
    gt = extract_gt_answer(sample)
    kind = answer_kind(gt)
    q, _ = extract_question_reasoning(sample)
    logger.print("=" * 88)
    logger.print(f"sample {idx} | id={sid} | kind={kind}")
    logger.print(f"GT: {gt!r}")
    if q:
        logger.print(f"Q: {short(q, 220)}")
    logger.print("final answers:")
    for model_name in model_names:
        res = results_by_model.get(model_name)
        if not res:
            logger.print(f"  {model_name:22s}: <missing>")
        elif "error" in res:
            logger.print(f"  {model_name:22s}: ERROR {res.get('error')}")
        else:
            ans = final_answer(res)
            mark = "OK" if final_exact(res) else "--"
            tr = res.get("trace") or []
            final_row = tr[-1] if tr else {}
            logger.print(
                f"  {model_name:22s}: {ans!r} | {mark} | "
                f"mask={int(float(final_row.get('mask_count', 0)))} "
                f"pos={float(final_row.get('position_match', 0.0)):.3f} "
                f"tok={float(final_row.get('token_set_match', 0.0)):.3f}"
            )


def log_overall_summary(logger: TeeLogger, model_names: List[str], results_by_sample: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    logger.print("=" * 88)
    logger.print("overall final exact summary")
    for model_name in model_names:
        total = 0
        exact = 0
        avg_mask = 0.0
        for sample_results in results_by_sample.values():
            res = sample_results.get(model_name)
            if not res or "trace" not in res:
                continue
            total += 1
            exact += int(final_exact(res))
            final_row = res["trace"][-1]
            avg_mask += float(final_row.get("mask_count", 0.0))
        if total > 0:
            logger.print(f"  {model_name:22s}: exact={exact}/{total} ({exact/total:.3f}) avg_final_mask={avg_mask/total:.3f}")
        else:
            logger.print(f"  {model_name:22s}: no valid results")


# ---------------------------------- main ------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SEDD/QRA/RL answer-mask recovery chains.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--start", default="QRA", help="Start key only used to choose data_dir; default QRA/S1K_RL.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--samples-per-type", type=int, default=2)
    parser.add_argument("--sample-id", action="append", default=[], help="Specific sample id. Can be repeated.")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma list. Default compares six: pretrain_start,pretrain_loss_best,pretrain_reward_best,QRA_start,QRA_loss_best,QRA_reward_best")
    parser.add_argument("--checkpoint", default=None, help="Trace one specific checkpoint only.")
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--run-name", default="compare_pretrain_qra_rl")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--t-start", type=float, default=0.95)
    parser.add_argument("--t-end", type=float, default=0.01)
    parser.add_argument("--transition-kind", default="analytic", choices=["analytic", "denoise"])
    parser.add_argument("--mode", default="greedy", choices=["sample", "greedy"])
    parser.add_argument("--freeze-filled", action="store_true", help="Diagnostic monotonic-fill mode; not true SEDD reverse chain.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    device = choose_device(args, cfg)
    start_name = selected_start_name(cfg, args.start)
    data_dir = resolve_data_dir(cfg, start_name)

    tokenizer = GPT2TokenizerFast.from_pretrained((cfg.get("model") or {}).get("tokenizer", "gpt2"))
    tokenizer.pad_token = tokenizer.eos_token

    samples = load_samples(cfg, start_name, args.split, tokenizer)
    chosen = pick_diverse_samples(samples, args.sample_id, int(args.num_samples), int(args.samples_per_type))
    if not chosen:
        raise RuntimeError("No samples selected for tracing.")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_DIR / "experiment" / "sample_chain" / f"{stamp}_{args.run_name}"
    chains_dir = out_dir / "chains"
    out_dir.mkdir(parents=True, exist_ok=True)
    chains_dir.mkdir(parents=True, exist_ok=True)

    logger = TeeLogger(out_dir / "run.log")
    try:
        model_specs = resolve_model_plan(args, cfg)
        model_names = [str(s["name"]) for s in model_specs]

        logger.print(f"[chain] device={device}")
        logger.print(f"[chain] data={data_dir}/{args.split}.jsonl loaded={len(samples)} selected={len(chosen)}")
        logger.print(f"[chain] models={model_names}")
        for spec in model_specs:
            logger.print(f"[chain] model_source {spec.get('name')}: {spec.get('source')} | {spec.get('checkpoint') or spec.get('pretrained')}")
        logger.print(f"[chain] out={out_dir}")
        dump_json(out_dir / "model_plan.json", model_specs)
        logger.print(f"[chain] mode={args.mode} steps={args.steps} t={args.t_start}->{args.t_end} freeze_filled={bool(args.freeze_filled)}")

        all_rows: List[Dict[str, Any]] = []
        rows_by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        results_by_sample: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

        for model_i, spec in enumerate(model_specs):
            model_name = str(spec["name"])
            ckpt = spec.get("checkpoint")
            logger.print("-" * 88)
            logger.print(f"[chain] loading {model_name}: {ckpt if ckpt else spec.get('pretrained')} ({spec.get('source')})")
            model, graph, noise = load_model_for_trace(cfg, spec, device)
            try:
                for sample_i, sample in enumerate(chosen, start=1):
                    sid = str(sample.get("id", f"sample_{sample_i}"))
                    set_seed(args.seed + sample_i * 1009 + model_i * 917 + stable_int(model_name, 997))
                    logger.print(f"[chain] {model_name} sample {sample_i}/{len(chosen)} id={sid} gt={extract_gt_answer(sample)!r}")
                    try:
                        res = trace_one_sample(model, graph, noise, tokenizer, sample, cfg, device, args, model_name)
                    except Exception as exc:
                        res = {
                            "model": model_name,
                            "sample_id": sid,
                            "error": repr(exc),
                            "gt_answer": extract_gt_answer(sample),
                            "question": extract_question_reasoning(sample)[0],
                            "reasoning": extract_question_reasoning(sample)[1],
                        }
                        logger.print(f"[warn] failed {model_name} sample {sid}: {exc}")
                    results_by_sample[sid][model_name] = res
                    if "trace" in res:
                        rows = result_to_chain_rows(res, sample_i)
                        all_rows.extend(rows)
                        rows_by_sample[sid].extend(rows)
            finally:
                del model, graph, noise
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Write requested outputs.
        write_csv(out_dir / "chain.csv", all_rows)
        used_names = set()
        for sample_i, sample in enumerate(chosen, start=1):
            sid = str(sample.get("id", f"sample_{sample_i}"))
            base = safe_name(sid, f"sample_{sample_i}")
            name = base
            if name in used_names:
                name = f"{base}_{sample_i}"
            used_names.add(name)
            write_csv(chains_dir / f"{name}_chain.csv", rows_by_sample.get(sid, []))

        write_sample_report(out_dir / "sample_report.txt", chosen, results_by_sample, model_names)

        # Terminal/log comparison after all models have been run, so each sample is grouped.
        logger.print("\n" + "#" * 88)
        logger.print("FINAL ANSWER COMPARISON")
        for sample_i, sample in enumerate(chosen, start=1):
            sid = str(sample.get("id", ""))
            log_sample_comparison(logger, sample_i, sample, results_by_sample.get(sid, {}), model_names)
        log_overall_summary(logger, model_names, results_by_sample)

        logger.print("-" * 88)
        logger.print(f"wrote: {out_dir / 'chain.csv'}")
        logger.print(f"wrote: {chains_dir}/<sample_id>_chain.csv")
        logger.print(f"wrote: {out_dir / 'run.log'}")
        logger.print(f"wrote: {out_dir / 'sample_report.txt'}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
