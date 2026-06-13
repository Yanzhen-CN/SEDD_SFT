from __future__ import annotations

"""Trace SEDD answer-mask reverse recovery chains for QRA/S1K_RL samples.

This script does NOT train. It runs a small answer-only reverse-diffusion loop and
records the intermediate answer segment after every step, so we can inspect what
SEDD-medium/QRA/RL checkpoints are doing at high/mid/low noise levels.

Default output:
    experiment/sample_chain/<timestamp>_<run_name>/
        trace.jsonl
        trace.log
        chain.csv
        chains/<sample_id>_chain.csv
        summary.json

The implementation uses the same core objects as the RL pipeline:
    - AnswerSegmentDataset
    - state_builder.encode_sample
    - state_builder.transition_probs
    - SEDD checkpoint loading

Important: this is an answer-segment trace. It fixes question/reasoning/prompt
positions and starts the answer positions as [MASK]. It is meant for diagnosis of
QRA answer recovery, not as a replacement for the full official SEDD sampler.
"""

import argparse
import csv
import datetime as dt
import gc
import json
import math
import os
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


# ----------------------------- basic utilities -----------------------------

def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_DIR / p


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def safe_name(text: str, fallback: str = "sample") -> str:
    s = str(text or fallback)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")
    return s[:120] or fallback


def make_control_chars_visible(value: object) -> str:
    """Make generated text safe for one-line CSV/log display.

    We keep the visual mask square (□), but render real control characters as
    literal escape sequences.  Thus a generated newline is shown as the two
    characters \n instead of splitting the CSV row in VSCode/Excel.
    """
    text = str(value if value is not None else "")
    out = []
    for ch in text:
        code = ord(ch)
        if ch == "\n":
            out.append(r"\n")
        elif ch == "\r":
            out.append(r"\r")
        elif ch == "\t":
            out.append(r"\t")
        elif code < 32 or code == 127:
            out.append(f"\\x{code:02x}")
        else:
            out.append(ch)
    return "".join(out)


def csv_cell(value: object) -> str:
    if isinstance(value, (list, dict, tuple)):
        value = json.dumps(value, ensure_ascii=False)
    return make_control_chars_visible(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    x = re.sub(r"\s+", " ", x).strip()
    if len(x) > 1 and x[-1] in ".;":
        x = x[:-1].strip()
    return x


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


def resolve_checkpoint_plan(args: argparse.Namespace, cfg: Dict[str, Any], start_name: str) -> List[Tuple[str, Path]]:
    """Return [(model_name, checkpoint_path), ...].

    By default compare the start checkpoint and the RL global best if they exist.
    Use --checkpoint to trace a specific checkpoint only.
    """
    if args.checkpoint:
        return [(args.checkpoint_name or "model", repo_path(args.checkpoint))]

    start_cfg = selected_start_cfg(cfg, start_name)
    out_dir = repo_path(start_cfg.get("output_dir", f"rl_qra_pipeline/modelparameter/rl_{start_name}"))
    plan: List[Tuple[str, Path]] = []

    init_ckpt = start_cfg.get("init_checkpoint")
    if init_ckpt:
        p = repo_path(init_ckpt)
        if p.exists():
            plan.append((f"{start_name}_base", p))

    rl_best = out_dir / "best.pth"
    if rl_best.exists():
        plan.append((f"rl_{start_name}", rl_best))

    if not plan:
        raise FileNotFoundError(
            "No checkpoint found. Pass --checkpoint PATH, or check starts.<start>.init_checkpoint/output_dir in config."
        )
    return plan


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
            return normalize_text(str(val))
    text = get_segment_text(sample, ["answer", "final_answer", "target"])
    if text:
        return normalize_text(text)
    # Fallback: last train=True segment.
    train_parts = []
    for _, seg in ordered_segments(sample):
        if isinstance(seg, dict) and bool(seg.get("train", False)):
            train_parts.append(str(seg.get("text", "")))
    if train_parts:
        return normalize_text("".join(train_parts))
    return ""


def extract_question_reasoning(sample: Dict[str, Any]) -> Tuple[str, str]:
    q = sample.get("question") or get_segment_text(sample, ["user", "question", "prompt"])
    r = sample.get("reasoning") or get_segment_text(sample, ["reasoning", "rationale", "thinking"])
    if not r:
        # Try to recover from segments before answer.
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
    ds = AnswerSegmentDataset(
        path,
        tokenizer,
        int(model_cfg.get("max_length", 1024)),
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

def load_model_for_trace(cfg: Dict[str, Any], checkpoint: Path, device: torch.device):
    model_cfg = cfg.get("model") or {}
    training_cfg = cfg.get("training") or {}
    pretrained = model_cfg.get("pretrained", "louaaron/sedd-medium")
    model = SEDD.from_pretrained(pretrained).to(device)
    model.config.model.length = int(model_cfg.get("max_length", getattr(model.config.model, "length", 1024)))

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

def visible_answer_tokens(
    tokenizer,
    ids: Sequence[int],
    mask_id: int,
    vocab_size: int,
) -> Tuple[str, List[str]]:
    pieces = []
    token_texts = []
    for tok in ids:
        ti = int(tok)
        if ti == int(mask_id) or ti < 0 or ti >= vocab_size:
            txt = "□"
        else:
            txt = tokenizer.decode([ti])
        txt = make_control_chars_visible(txt)
        pieces.append(txt)
        token_texts.append(txt)
    return "".join(pieces), token_texts


def score_state(
    curr_ids: Sequence[int],
    gt_ids: Sequence[int],
    mask_id: int,
    t_value: float,
) -> Dict[str, float]:
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

    # A simple t-dependent diagnostic score. It is not used for training here;
    # it helps us inspect whether the chain improves at early/middle/late steps.
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


def topk_for_positions(
    probs: torch.Tensor,
    tokenizer,
    positions: Sequence[int],
    k: int,
    max_positions: int,
) -> List[Dict[str, Any]]:
    if k <= 0 or max_positions <= 0:
        return []
    out = []
    vocab_size = int(getattr(tokenizer, "vocab_size", 50257))
    for pos in list(positions)[:max_positions]:
        vals, inds = torch.topk(probs[0, int(pos)].detach().float().cpu(), k=min(k, probs.shape[-1]))
        rows = []
        for v, i in zip(vals.tolist(), inds.tolist()):
            token_id = int(i)
            if token_id < 0 or token_id >= vocab_size:
                token = "[MASK/OUT]"
            else:
                token = make_control_chars_visible(tokenizer.decode([token_id]))
            rows.append({"token_id": token_id, "token": token, "prob": float(v)})
        out.append({"position": int(pos), "topk": rows})
    return out


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
    # Answer-only generation: question/reasoning/prompt fixed; answer starts masked.
    x[:, answer_pos] = mask_id
    project_fixed_(x, encoded.layout.fixed_locs, encoded.layout.fixed_ids.to(device))

    num_steps = int(args.steps)
    t_start = float(args.t_start)
    t_end = float(args.t_end)
    dt_step = (t_start - t_end) / max(1, num_steps)
    transition_kind = str(args.transition_kind)

    trace_rows = []
    prev_score = None
    prev_answer_ids = [int(x[0, p].item()) for p in answer_pos]

    with torch.no_grad():
        for step in range(num_steps + 1):
            # step=0 records the initial all-mask state before transition.
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

            # Do not compute probs for the final recorded row unless requested by transition loop.
            if step == num_steps:
                trace_rows.append(record)
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

            # Positions for top-k: changed positions from previous row are known only after sampling;
            # before sampling, inspect currently masked positions plus early positions.
            masked_abs = [p for p in answer_pos if int(x[0, p].item()) == mask_id]
            inspect_abs = masked_abs[: args.max_topk_positions]
            if not inspect_abs:
                inspect_abs = answer_pos[: args.max_topk_positions]
            record["topk_before_transition"] = topk_for_positions(
                probs, tokenizer, inspect_abs, int(args.topk), int(args.max_topk_positions)
            )
            trace_rows.append(record)

            prev_score = score["stage_score"]
            prev_answer_ids = curr_ids

            if args.mode == "greedy":
                next_x = probs.argmax(dim=-1)
            else:
                flat = probs.view(-1, probs.shape[-1])
                sampled = torch.multinomial(flat.float().clamp_min(0.0), num_samples=1).view_as(x)
                next_x = sampled

            # Optional diagnostic mode: once a position is non-mask, keep it.
            # This is NOT the true reverse chain, but useful to visualize monotonic filling.
            if args.freeze_filled:
                keep = torch.zeros_like(x, dtype=torch.bool)
                for p in answer_pos:
                    keep[:, p] = x[:, p] != mask_id
                next_x = torch.where(keep, x, next_x)

            # Always keep non-train/prompt positions fixed.
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


# ------------------------------- CSV output --------------------------------

CHAIN_FIELDS = [
    "sample_index", "sample_id", "answer_kind", "model", "step", "t",
    "answer", "gt_answer", "exact_text_match", "exact_token_match",
    "mask_count", "filled_ratio", "position_match", "token_set_match",
    "stage_score", "stage_score_delta", "changed_answer_indices",
    "answer_token_texts", "answer_token_ids", "gt_answer_token_ids",
]


def result_to_chain_rows(result: Dict[str, Any], sample_index: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    gt_norm = normalize_text(result.get("gt_answer", ""))
    for tr in result.get("trace", []) or []:
        ans = str(tr.get("answer_text", ""))
        rows.append({
            "sample_index": sample_index,
            "sample_id": result.get("sample_id", ""),
            "answer_kind": result.get("answer_kind", ""),
            "model": result.get("model", ""),
            "step": tr.get("step", ""),
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
            "changed_answer_indices": tr.get("changed_answer_indices", []),
            "answer_token_texts": tr.get("answer_token_texts", []),
            "answer_token_ids": tr.get("answer_token_ids", []),
            "gt_answer_token_ids": tr.get("gt_answer_token_ids", []),
        })
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str] = CHAIN_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(fields),
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_cell(row.get(k, "")) for k in fields})


# ------------------------------- markdown/log ------------------------------

def short(s: str, n: int = 900) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[:n] + " ..."


def trace_to_markdown(result: Dict[str, Any], stride: int = 1) -> str:
    lines = []
    lines.append(f"## {result['model']} | {result['sample_id']} | {result['answer_kind']}")
    lines.append("")
    lines.append(f"**GT answer:** `{result['gt_answer']}`")
    lines.append("")
    if result.get("question"):
        lines.append("**Question:**")
        lines.append("")
        lines.append(short(result["question"], 700))
        lines.append("")
    if result.get("reasoning"):
        lines.append("**Reasoning excerpt:**")
        lines.append("")
        lines.append(short(result["reasoning"], 1000))
        lines.append("")

    lines.append("| step | t | answer state | masks | pos_match | token_set | exact | stage_score | Δscore | changed |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for row in result["trace"]:
        if int(row["step"]) % max(1, stride) != 0 and int(row["step"]) != int(result["num_steps"]):
            continue
        ans = str(row["answer_text"]).replace("|", "\\|")
        if len(ans) > 160:
            ans = ans[:157] + "..."
        lines.append(
            f"| {row['step']} | {row['t']:.3f} | `{ans}` | "
            f"{int(row['mask_count'])} | {row['position_match']:.3f} | "
            f"{row['token_set_match']:.3f} | {row['exact_token_match']:.0f} | "
            f"{row['stage_score']:.3f} | {row['stage_score_delta']:+.3f} | "
            f"{row['changed_answer_indices']} |"
        )
    lines.append("")

    # Show top-k for a few early/middle/late rows.
    trace = result["trace"]
    pick_steps = sorted(set([0, len(trace) // 3, 2 * len(trace) // 3, len(trace) - 2]))
    lines.append("**Selected top-k before transition:**")
    lines.append("")
    for idx in pick_steps:
        if idx < 0 or idx >= len(trace):
            continue
        row = trace[idx]
        topk = row.get("topk_before_transition") or []
        if not topk:
            continue
        lines.append(f"- step={row['step']} t={row['t']:.3f} answer=`{str(row['answer_text']).replace('`', '')}`")
        for item in topk[:3]:
            toks = ", ".join([f"{x['token']!r}:{x['prob']:.3g}" for x in item.get("topk", [])])
            lines.append(f"  - pos {item['position']}: {toks}")
    lines.append("\n")
    return "\n".join(lines)


# ---------------------------------- main ------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Trace SEDD QRA answer-mask recovery chains.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--start", default=None, help="Start key under config.starts, e.g. QRA/pretrain.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--samples-per-type", type=int, default=2)
    parser.add_argument("--sample-id", action="append", default=[], help="Specific sample id. Can be repeated.")
    parser.add_argument("--checkpoint", default=None, help="Trace one specific checkpoint only.")
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--run-name", default="sample_chain")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # Reverse-chain diagnostic settings.
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--t-start", type=float, default=0.95)
    parser.add_argument("--t-end", type=float, default=0.01)
    parser.add_argument("--transition-kind", default="analytic", choices=["analytic", "denoise"])
    parser.add_argument("--mode", default="sample", choices=["sample", "greedy"])
    parser.add_argument("--freeze-filled", action="store_true", help="Diagnostic monotonic-fill mode; not true SEDD reverse chain.")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-topk-positions", type=int, default=4)
    parser.add_argument("--md-stride", type=int, default=1, help="Only print every N trace rows in markdown.")
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
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_jsonl = out_dir / "trace.jsonl"
    trace_log = out_dir / "trace.log"
    chain_csv = out_dir / "chain.csv"
    chains_dir = out_dir / "chains"
    chains_dir.mkdir(parents=True, exist_ok=True)

    ckpts = resolve_checkpoint_plan(args, cfg, start_name)
    summary = {
        "time": stamp,
        "config": str(args.config),
        "start": start_name,
        "data_dir": str(data_dir),
        "split": args.split,
        "device": str(device),
        "steps": args.steps,
        "t_start": args.t_start,
        "t_end": args.t_end,
        "transition_kind": args.transition_kind,
        "mode": args.mode,
        "freeze_filled": bool(args.freeze_filled),
        "checkpoints": [{"name": n, "path": str(p)} for n, p in ckpts],
        "samples": [{"id": s.get("id", ""), "answer": extract_gt_answer(s), "kind": answer_kind(extract_gt_answer(s))} for s in chosen],
    }
    dump_json(out_dir / "summary.json", summary)

    print(f"[trace] device={device}", flush=True)
    print(f"[trace] data={data_dir}/{args.split}.jsonl samples={len(samples)} selected={len(chosen)}", flush=True)
    print(f"[trace] out={out_dir}", flush=True)

    all_log = ["# SEDD QRA sample-chain trace", "", f"Output dir: `{out_dir}`", ""]
    all_csv_rows: List[Dict[str, Any]] = []
    rows_by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for model_name, ckpt_path in ckpts:
        print(f"[trace] loading {model_name}: {ckpt_path}", flush=True)
        model, graph, noise = load_model_for_trace(cfg, ckpt_path, device)
        try:
            for i, sample in enumerate(chosen):
                # Change seed per sample/model but keep deterministic across reruns.
                set_seed(args.seed + i * 1009 + abs(hash(model_name)) % 997)
                print(f"[trace] {model_name} sample {i+1}/{len(chosen)} id={sample.get('id','')} ans={extract_gt_answer(sample)!r}", flush=True)
                try:
                    res = trace_one_sample(model, graph, noise, tokenizer, sample, cfg, device, args, model_name)
                except Exception as exc:
                    res = {
                        "model": model_name,
                        "sample_id": sample.get("id", ""),
                        "error": repr(exc),
                        "gt_answer": extract_gt_answer(sample),
                    }
                    print(f"[warn] failed sample {sample.get('id','')}: {exc}", flush=True)
                append_jsonl(trace_jsonl, res)
                if "trace" in res:
                    csv_rows = result_to_chain_rows(res, sample_index=i + 1)
                    all_csv_rows.extend(csv_rows)
                    rows_by_sample[str(res.get("sample_id", sample.get("id", f"sample_{i+1}")))].extend(csv_rows)
                    all_log.append(trace_to_markdown(res, stride=int(args.md_stride)))
                else:
                    all_md.append(f"## {model_name} | {res.get('sample_id')}\n\nERROR: `{res.get('error')}`\n")
        finally:
            del model, graph, noise
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(chain_csv, all_csv_rows)
    for sid, rows in rows_by_sample.items():
        write_csv(chains_dir / f"{safe_name(sid, 'sample')}_chain.csv", rows)
    trace_log.write_text("\n".join(all_log), encoding="utf-8-sig")
    print(f"[trace] wrote {trace_jsonl}", flush=True)
    print(f"[trace] wrote {chain_csv}", flush=True)
    print(f"[trace] wrote {chains_dir}/<sample_id>_chain.csv", flush=True)
    print(f"[trace] wrote {trace_log}", flush=True)


if __name__ == "__main__":
    main()
