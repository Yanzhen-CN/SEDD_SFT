from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
for path in (REPO_DIR, REPO_DIR / "sft_answer_pipeline"):
    sys.path.insert(0, str(path))

from answer_dataset import ordered_segments, sample_text  # noqa: E402

ASSISTANT_RE = re.compile(r"(?im)(^|\n)\s*assistant\s*:")


@dataclass
class SegmentLayout:
    real_len: int
    train_mask: torch.Tensor
    fixed_locs: List[int]
    fixed_ids: torch.Tensor
    reason_positions: List[int] = field(default_factory=list)
    answer_positions: List[int] = field(default_factory=list)
    segment_spans: Dict[str, Tuple[int, int]] = field(default_factory=dict)


@dataclass
class EncodedSample:
    ids: List[int]
    train_mask: List[int]
    layout: SegmentLayout
    reference_completion: str


def completion_from_full_text(text: str) -> str:
    raw = str(text or "")
    matches = list(ASSISTANT_RE.finditer(raw))
    if not matches:
        return raw.strip()
    m = matches[-1]
    marker = m.group(0)
    colon = marker.rfind(":")
    start = m.start() + colon + 1
    return raw[start:].strip()


def reference_completion(sample: Dict) -> str:
    full = "".join(seg.get("text", "") for _, seg in ordered_segments(sample))
    comp = completion_from_full_text(full)
    return comp if comp else sample_text(sample, train=True)


def _classify_train_positions(segment_spans: Dict[str, Tuple[int, int]], segments_info: List[Tuple[str, str, bool]], train_mask: List[int]) -> Tuple[List[int], List[int]]:
    reason_positions: List[int] = []
    answer_positions: List[int] = []
    answer_anchor_end = None

    for name, text, is_train in segments_info:
        lo_name = str(name).lower()
        start, end = segment_spans[name]
        text_l = str(text).lower()
        if not is_train and "answer" in text_l:
            answer_anchor_end = end
        if is_train:
            positions = list(range(start, end))
            if "answer" in lo_name:
                answer_positions.extend(positions)
            elif "reason" in lo_name or "rationale" in lo_name or "thinking" in lo_name:
                reason_positions.extend(positions)

    train_positions = [i for i, m in enumerate(train_mask) if m]
    if not answer_positions and answer_anchor_end is not None:
        answer_positions = [i for i in train_positions if i >= answer_anchor_end]
    if not answer_positions and train_positions:
        # Conservative fallback: last third is Answer.
        cut = train_positions[int(len(train_positions) * 2 / 3)] if len(train_positions) >= 3 else train_positions[0]
        answer_positions = [i for i in train_positions if i >= cut]
    if not reason_positions:
        ans_min = min(answer_positions) if answer_positions else 10**9
        reason_positions = [i for i in train_positions if i < ans_min]

    reason_positions = sorted(set(reason_positions) - set(answer_positions))
    answer_positions = sorted(set(answer_positions))
    return reason_positions, answer_positions


def encode_sample(sample: Dict, tokenizer, max_length: int, eos_token_id: int) -> EncodedSample:
    ids: List[int] = []
    train_mask: List[int] = []
    spans: Dict[str, Tuple[int, int]] = {}
    segments_info: List[Tuple[str, str, bool]] = []

    for raw_name, seg in ordered_segments(sample):
        name = str(raw_name)
        text = seg.get("text", "")
        toks = tokenizer(text, add_special_tokens=False).input_ids
        is_train = bool(seg.get("train", False))
        start = len(ids)
        ids.extend(toks)
        train_mask.extend([1 if is_train else 0] * len(toks))
        end = len(ids)

        unique = name
        c = 1
        while unique in spans:
            c += 1
            unique = f"{name}#{c}"
        spans[unique] = (start, end)
        segments_info.append((unique, text, is_train))

    if len(ids) > int(max_length):
        raise ValueError(f"sample {sample.get('id')} over max_length: {len(ids)}>{max_length}")
    if not any(train_mask):
        raise ValueError(f"sample {sample.get('id')} has no train=True target tokens")

    real_len = len(ids)
    pad_len = int(max_length) - real_len
    if pad_len > 0:
        ids = ids + [eos_token_id] * pad_len
        train_mask = train_mask + [0] * pad_len

    fixed_locs = [idx for idx, m in enumerate(train_mask) if not m]
    reason_positions, answer_positions = _classify_train_positions(spans, segments_info, train_mask)
    layout = SegmentLayout(
        real_len=real_len,
        train_mask=torch.tensor(train_mask, dtype=torch.bool),
        fixed_locs=fixed_locs,
        fixed_ids=torch.tensor([[ids[idx] for idx in fixed_locs]], dtype=torch.long),
        reason_positions=reason_positions,
        answer_positions=answer_positions,
        segment_spans=spans,
    )
    return EncodedSample(ids=ids, train_mask=train_mask, layout=layout, reference_completion=reference_completion(sample))


def mask_id_from_graph(graph) -> int:
    for attr in ("mask_token", "mask_id", "absorbing_state"):
        if hasattr(graph, attr):
            try:
                return int(getattr(graph, attr))
            except Exception:
                pass
    if hasattr(graph, "dim"):
        return int(getattr(graph, "dim")) - 1
    if hasattr(graph, "vocab_size"):
        return int(getattr(graph, "vocab_size")) - 1
    return 50257


def project_fixed_(x: torch.Tensor, fixed_locs: Sequence[int], fixed_ids: torch.Tensor) -> torch.Tensor:
    if fixed_locs:
        x[:, list(fixed_locs)] = fixed_ids.to(x.device)
    return x


def safe_decode_ids(tokenizer, ids: Sequence[int]) -> str:
    vocab = int(getattr(tokenizer, "vocab_size", 50257))
    clean = []
    for x in ids:
        xi = int(x)
        clean.append(xi if 0 <= xi < vocab else tokenizer.eos_token_id)
    return tokenizer.decode(clean)


def decode_positions(tokenizer, ids: torch.Tensor, positions: Sequence[int]) -> str:
    if not positions:
        return ""
    arr = ids.detach().cpu().tolist()
    vals = [arr[i] for i in positions if i < len(arr)]
    return safe_decode_ids(tokenizer, vals)


def _one_hot_current(x: torch.Tensor, dim: int, dtype: torch.dtype) -> torch.Tensor:
    return F.one_hot(x.clamp_min(0), num_classes=dim).to(dtype=dtype)


def sanitize_probs(probs: torch.Tensor, current_x: torch.Tensor, fixed_locs: Sequence[int] | None = None, fixed_ids: torch.Tensor | None = None, eps: float = 1e-20) -> torch.Tensor:
    dim = probs.shape[-1]
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    denom = probs.sum(dim=-1, keepdim=True)
    fallback = _one_hot_current(current_x, dim, probs.dtype)
    probs = torch.where(denom > 0, probs / denom.clamp_min(eps), fallback)
    if fixed_locs and fixed_ids is not None:
        probs[:, list(fixed_locs), :] = 0.0
        probs[0, list(fixed_locs), fixed_ids.to(probs.device)[0]] = 1.0
    return probs


def transition_probs(model, graph, noise, x: torch.Tensor, t: torch.Tensor, step_size: float, kind: str = "analytic", train: bool = True, fixed_locs: Sequence[int] | None = None, fixed_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Return reverse transition probabilities induced by SEDD ratio field.

    This is the critical bridge from SEDD ratio output to a policy distribution:
    model(x, sigma) -> exp(log_score) -> graph.staggered_score / transition -> normalized pi(a|s).
    """
    model.train(train)
    with autocast(enabled=x.is_cuda, dtype=torch.bfloat16):
        curr_sigma = noise(t)[0]
        raw_log_score = model(x, curr_sigma.reshape(-1))
        score = raw_log_score.exp()
        if kind == "analytic":
            next_t = torch.clamp(t - float(step_size), min=0.0)
            next_sigma = noise(next_t)[0]
            dsigma = curr_sigma - next_sigma
        elif kind == "denoise":
            dsigma = curr_sigma
        else:
            raise ValueError(f"unknown transition kind: {kind}")
        stag_score = graph.staggered_score(score, dsigma)
        probs = stag_score * graph.transp_transition(x, dsigma)
        if kind == "denoise" and getattr(graph, "absorb", False):
            probs = probs.clone()
            probs[..., -1] = 0.0
    return sanitize_probs(probs.float(), x, fixed_locs=fixed_locs, fixed_ids=fixed_ids)
