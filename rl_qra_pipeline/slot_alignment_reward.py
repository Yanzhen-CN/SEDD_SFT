from __future__ import annotations

"""Stage-aware slot-alignment reward for rollout-chain RL.

The main design goal is to punish the failure mode observed in sample chains:
answers such as ``(3,4]`` becoming ``4,344``.  Token overlap is not enough.
Each rollout action is a position-token pair, so the reward first evaluates
whether the chosen token type fits the chosen GT slot type.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch


BRACKET_LEFT = {"(", "[", "{", "\\(", "\\["}
BRACKET_RIGHT = {")", "]", "}", "\\)", "\\]"}
SEPARATORS = {",", ";", ":"}
EQUALS = {"=", " ="}
OPERATORS = {"+", "-", "*", "/", "^", "≤", "≥", "<", ">", "<=", ">=", "\\le", "\\ge"}
UNITS = {
    "m", "mm", "cm", "km", "kg", "g", "s", "ms", "N", "J", "W", "V", "A", "Hz",
    " m", " mm", " cm", " kg", " g", " s", " N", " J", " W", " V", " A", " Hz",
    "m/s", "m/s^2", "\\mathrm{m}", "\\mathrm{~m}", "\\mathrm{mm}", "\\mathrm{cm}",
}
LATEX_ATOMS = {"\\sqrt", "\\frac", "\\pi", "\\omega", "\\sin", "\\cos", "sqrt", "pi"}


@dataclass
class StageCoefficients:
    action_align: float
    skeleton_delta: float
    exact_delta: float
    clean_action: float
    mask_action: float


@dataclass
class RewardBreakdown:
    total: float
    stage: str
    action_align: float
    action_clean: float
    action_mask: float
    slot_importance: float
    skeleton_before: float
    skeleton_after: float
    skeleton_delta: float
    exact_before: float
    exact_after: float
    exact_delta: float
    gt_slot_kind: str
    token_kind: str
    wrong_position_token: float


def norm_text(text: str) -> str:
    s = str(text or "")
    s = s.replace("Ġ", " ").replace("▁", " ").replace("\u00a0", " ")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\mathrm{~m}", "m").replace("\\mathrm{m}", "m")
    s = s.replace("\\mathrm{mm}", "mm").replace("\\mathrm{cm}", "cm")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def normalize_answer_text(text: str) -> str:
    return norm_text(text).replace(" ", "")


def stage_from_t(t: float) -> str:
    t = float(t)
    if t >= 0.70:
        return "early"
    if t > 0.30:
        return "middle"
    return "late"


def stage_coefficients(t: float, answer_kind: str = "") -> StageCoefficients:
    """Weights for action-level transition reward.

    action_align is intentionally the largest coefficient in every stage.
    Exact is deliberately tiny early, then becomes important late.
    """
    stage = stage_from_t(t)
    kind = str(answer_kind or "")
    if kind in {"single_letter", "single_integer"}:
        if stage == "early":
            return StageCoefficients(0.52, 0.03, 0.25, 0.12, 0.08)
        if stage == "middle":
            return StageCoefficients(0.42, 0.03, 0.45, 0.07, 0.03)
        return StageCoefficients(0.30, 0.02, 0.63, 0.04, 0.01)
    if stage == "early":
        return StageCoefficients(action_align=0.72, skeleton_delta=0.08, exact_delta=0.03, clean_action=0.12, mask_action=0.05)
    if stage == "middle":
        return StageCoefficients(action_align=0.48, skeleton_delta=0.32, exact_delta=0.10, clean_action=0.08, mask_action=0.02)
    return StageCoefficients(action_align=0.35, skeleton_delta=0.15, exact_delta=0.45, clean_action=0.05, mask_action=0.00)


def token_kind(text: str) -> str:
    s = norm_text(text)
    if not s:
        return "empty"
    if s in BRACKET_LEFT:
        return "left_bracket"
    if s in BRACKET_RIGHT:
        return "right_bracket"
    if s in SEPARATORS:
        return "separator"
    if s in EQUALS:
        return "equals"
    if s in OPERATORS:
        return "operator"
    if s in UNITS:
        return "unit"
    if s in LATEX_ATOMS:
        return "latex_atom"
    if re.fullmatch(r"[A-Za-z]", s):
        return "letter"
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", s) or re.fullmatch(r"\.\d+", s):
        return "number"
    if re.search(r"\d", s) and re.fullmatch(r"[+\-\.0-9 ]+", s):
        return "number"
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", s):
        return "variable"
    if re.fullmatch(r"[\[\]\(\)\{\},=+\-*/^<>.:;]+", s):
        return "symbol"
    return "text"


def skeleton_group(kind: str) -> str:
    if kind == "left_bracket":
        return "left_bracket"
    if kind == "right_bracket":
        return "right_bracket"
    if kind == "number":
        return "value"
    if kind in {"letter", "variable"}:
        return "variable"
    if kind == "unit":
        return "unit"
    if kind in {"separator", "equals", "operator", "latex_atom"}:
        return kind
    return kind


def compatible_kind(cur_kind: str, gt_kind: str) -> bool:
    """Type compatibility for the same slot.

    Left and right brackets are intentionally not compatible with each other.
    This is the strict-position part: ')' in a left-bracket slot is wrong.
    """
    if cur_kind == gt_kind:
        return True
    if cur_kind == "number" and gt_kind == "number":
        return True
    if cur_kind in {"letter", "variable"} and gt_kind in {"letter", "variable"}:
        return True
    if cur_kind == "unit" and gt_kind == "unit":
        return True
    return False


def is_garbage_text(text: str) -> bool:
    s = str(text or "")
    ns = norm_text(s)
    if not ns:
        return False
    if ns in {"\\", "$$", "$", "```"}:
        return True
    # Backslash-closing bracket fragments are often malformed in short answer slots.
    if ns in {"\\)", "\\]", "\\}", "\\("}:
        return True
    if any(ch in ns for ch in ["�", "》", "）", "（", "“", "”"]):
        return True
    if len(ns) >= 8 and re.search(r"[A-Za-z]", ns) and " " in ns:
        return True
    if ns.lower() in {"the", "therefore", "answer", "because", "solution"}:
        return True
    return False


def is_mask_token(token_id: int, mask_id: int) -> bool:
    return int(token_id) == int(mask_id)


def semantic_match(cur_text: str, gt_text: str) -> bool:
    c = norm_text(cur_text)
    g = norm_text(gt_text)
    if not c or not g:
        return False
    if normalize_answer_text(c) == normalize_answer_text(g):
        return True
    if token_kind(c) == "unit" and token_kind(g) == "unit":
        return True
    return False


def decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return ""


def slot_importance(t: float, gt_text: str, answer_kind: str = "") -> float:
    stage = stage_from_t(t)
    gt_kind = token_kind(gt_text)
    group = skeleton_group(gt_kind)
    kind = str(answer_kind or "")
    if kind in {"single_letter", "single_integer"}:
        return 1.0
    if stage == "early":
        if group in {"left_bracket", "right_bracket", "separator", "equals", "operator", "unit"}:
            return 1.80
        if group in {"variable", "latex_atom"}:
            return 1.20
        if group == "value":
            return 0.50
        return 0.80
    if stage == "middle":
        if group in {"left_bracket", "right_bracket", "separator", "equals", "operator", "unit"}:
            return 1.35
        if group in {"variable", "latex_atom"}:
            return 1.10
        return 1.00
    if group == "value":
        return 1.20
    return 1.00


def compatible_score_by_stage(t: float, cur_kind: str, gt_kind: str, answer_kind: str = "") -> float:
    stage = stage_from_t(t)
    kind = str(answer_kind or "")
    if kind == "single_letter" and cur_kind in {"letter", "variable"} and gt_kind in {"letter", "variable"}:
        if stage == "early":
            return 0.10
        if stage == "middle":
            return -0.20
        return -0.60
    if stage == "early":
        # Position type is good, exact value can wait.
        if gt_kind in {"left_bracket", "right_bracket", "separator", "equals", "operator", "unit"}:
            return 0.72
        return 0.55
    if stage == "middle":
        if gt_kind in {"left_bracket", "right_bracket", "separator", "equals", "operator", "unit"}:
            return 0.45
        return 0.30
    # Late: same type but wrong exact is not enough anymore.
    if gt_kind in {"number", "letter", "variable"}:
        return -0.20
    return -0.05


def action_alignment_score(
    action_answer_index: int,
    token_id: int,
    gt_ids: Sequence[int],
    tokenizer,
    mask_id: int,
    t: float,
    answer_kind: str = "",
) -> Tuple[float, Dict[str, Any]]:
    """Direct reward for placing token_id into the selected answer slot.

    This is the main term.  It catches cases like slot2 -> '3' when slot2 should
    be comma.  A token appearing somewhere in GT is not rewarded if the selected
    slot expects a different type.
    """
    idx = int(action_answer_index)
    token_id = int(token_id)
    if idx < 0 or idx >= len(gt_ids):
        return -1.0, {"reason": "bad_action_index", "slot_importance": 1.0}

    gt_id = int(gt_ids[idx])
    tok_text = decode_token(tokenizer, token_id)
    gt_text = decode_token(tokenizer, gt_id)
    cur_kind = token_kind(tok_text)
    gt_kind = token_kind(gt_text)
    importance = slot_importance(t, gt_text, answer_kind)
    wrong_position_token = float((token_id in {int(x) for x in gt_ids}) and token_id != gt_id)

    if is_mask_token(token_id, mask_id):
        stage = stage_from_t(t)
        base = 0.04 if stage == "early" else (-0.15 if stage == "middle" else -0.55)
    elif is_garbage_text(tok_text):
        base = -1.0
    elif token_id == gt_id:
        base = 1.0
    elif semantic_match(tok_text, gt_text):
        base = 0.85
    elif compatible_kind(cur_kind, gt_kind):
        base = compatible_score_by_stage(t, cur_kind, gt_kind, answer_kind)
    else:
        # Exact GT token in the wrong type of slot is a major negative.
        if wrong_position_token:
            base = -0.78
        elif cur_kind in {"number", "letter", "variable"} and gt_kind not in {"number", "letter", "variable"}:
            base = -0.72
        elif gt_kind in {"number", "letter", "variable"} and cur_kind not in {"number", "letter", "variable"}:
            base = -0.60
        else:
            base = -0.50

    score = float(max(-1.0, min(1.0, base * importance)))
    return score, {
        "token_text": tok_text,
        "gt_text": gt_text,
        "token_kind": cur_kind,
        "gt_slot_kind": gt_kind,
        "slot_importance": float(importance),
        "wrong_position_token": wrong_position_token,
        "base_action_align": float(base),
        "action_align": float(score),
    }


def clean_action_score(token_id: int, tokenizer, mask_id: int, t: float) -> float:
    if is_mask_token(token_id, mask_id):
        return 0.0
    txt = decode_token(tokenizer, int(token_id))
    if is_garbage_text(txt):
        return -1.0
    ns = norm_text(txt)
    if len(ns) > 12 and re.search(r"[A-Za-z]", ns):
        return -0.35
    return 0.10


def mask_action_score(token_id: int, mask_id: int, t: float) -> float:
    if not is_mask_token(token_id, mask_id):
        return 0.0
    stage = stage_from_t(t)
    if stage == "early":
        return 0.05
    if stage == "middle":
        return -0.15
    return -0.50


def exact_score(cur_ids: Sequence[int], gt_ids: Sequence[int], mask_id: int) -> float:
    n = min(len(cur_ids), len(gt_ids))
    if n <= 0:
        return 0.0
    return float(sum(1 for c, g in zip(cur_ids[:n], gt_ids[:n]) if int(c) == int(g)) / n)


def skeleton_score(cur_ids: Sequence[int], gt_ids: Sequence[int], tokenizer, mask_id: int, answer_kind: str = "") -> float:
    n = min(len(cur_ids), len(gt_ids))
    if n <= 0:
        return 0.0
    hits = 0.0
    denom = 0.0
    for c, g in zip(cur_ids[:n], gt_ids[:n]):
        ci = int(c)
        gi = int(g)
        gt_text = decode_token(tokenizer, gi)
        cur_text = decode_token(tokenizer, ci) if ci != int(mask_id) else ""
        gt_kind = token_kind(gt_text)
        cur_kind = token_kind(cur_text) if ci != int(mask_id) else "mask"
        gt_group = skeleton_group(gt_kind)
        cur_group = skeleton_group(cur_kind)
        w = slot_importance(0.5, gt_text, answer_kind)
        denom += abs(w)
        if ci == int(mask_id):
            hits += 0.0
        elif ci == gi:
            hits += w
        elif semantic_match(cur_text, gt_text):
            hits += 0.85 * w
        elif compatible_kind(cur_kind, gt_kind) or cur_group == gt_group:
            # Skeleton can tolerate exact mismatch if slot type is correct.
            hits += 0.55 * w
        elif is_garbage_text(cur_text):
            hits -= 0.85 * w
        else:
            hits -= 0.35 * w
    return float(max(-1.0, min(1.0, hits / max(denom, 1e-8))))


def stage_state_score(
    cur_ids: Sequence[int],
    gt_ids: Sequence[int],
    tokenizer,
    mask_id: int,
    t: float,
    answer_kind: str = "",
) -> float:
    """A diagnostic full-state score, not the primary action reward."""
    coeff = stage_coefficients(t, answer_kind)
    skel = skeleton_score(cur_ids, gt_ids, tokenizer, mask_id, answer_kind)
    exact = exact_score(cur_ids, gt_ids, mask_id)
    # Approximate alignment as skeleton here; the primary action term is stricter.
    return float(max(-1.0, min(1.0, coeff.action_align * skel + coeff.exact_delta * exact)))


def step_alignment_reward(
    before_ids: Sequence[int],
    after_ids: Sequence[int],
    gt_ids: Sequence[int],
    tokenizer,
    mask_id: int,
    t: float,
    answer_kind: str = "",
    reward_clip: float = 1.0,
    action_answer_index: int | None = None,
    action_token_id: int | None = None,
) -> Tuple[float, Dict[str, Any]]:
    """Stage-aware transition reward.

    If action info is supplied, the largest term is direct position-type
    alignment for that action.  This is the recommended rollout mode.
    """
    coeff = stage_coefficients(t, answer_kind)
    skel_before = skeleton_score(before_ids, gt_ids, tokenizer, mask_id, answer_kind)
    skel_after = skeleton_score(after_ids, gt_ids, tokenizer, mask_id, answer_kind)
    exact_before = exact_score(before_ids, gt_ids, mask_id)
    exact_after = exact_score(after_ids, gt_ids, mask_id)
    skel_delta = float(skel_after - skel_before)
    exact_delta = float(exact_after - exact_before)

    if action_answer_index is None or action_token_id is None:
        # Fallback for older callers: state delta only.
        before_score = stage_state_score(before_ids, gt_ids, tokenizer, mask_id, t, answer_kind)
        after_score = stage_state_score(after_ids, gt_ids, tokenizer, mask_id, t, answer_kind)
        reward = float(after_score - before_score)
        info: Dict[str, Any] = {
            "stage": stage_from_t(t),
            "action_align": 0.0,
            "action_clean": 0.0,
            "action_mask": 0.0,
            "skeleton_before": skel_before,
            "skeleton_after": skel_after,
            "skeleton_delta": skel_delta,
            "exact_before": exact_before,
            "exact_after": exact_after,
            "exact_delta": exact_delta,
            "reward": reward,
        }
    else:
        action_align, action_info = action_alignment_score(
            int(action_answer_index), int(action_token_id), gt_ids, tokenizer, mask_id, t, answer_kind
        )
        clean = clean_action_score(int(action_token_id), tokenizer, mask_id, t)
        mask = mask_action_score(int(action_token_id), mask_id, t)
        reward = (
            coeff.action_align * action_align
            + coeff.skeleton_delta * skel_delta
            + coeff.exact_delta * exact_delta
            + coeff.clean_action * clean
            + coeff.mask_action * mask
        )
        info = {
            "stage": stage_from_t(t),
            "action_align": float(action_align),
            "action_clean": float(clean),
            "action_mask": float(mask),
            "skeleton_before": float(skel_before),
            "skeleton_after": float(skel_after),
            "skeleton_delta": float(skel_delta),
            "exact_before": float(exact_before),
            "exact_after": float(exact_after),
            "exact_delta": float(exact_delta),
            "reward": float(reward),
            **action_info,
        }

    c = abs(float(reward_clip)) if reward_clip is not None else 0.0
    if c > 0:
        reward = max(-c, min(c, float(reward)))
    info["reward"] = float(reward)
    return float(reward), info


def reward_to_go(rewards: Sequence[float], gamma: float = 0.95) -> List[float]:
    out: List[float] = []
    running = 0.0
    for r in reversed([float(x) for x in rewards]):
        running = float(r) + float(gamma) * running
        out.append(running)
    return list(reversed(out))


def normalize_advantages(values: Sequence[float], clip: float = 0.25, device=None, dtype=torch.float32) -> torch.Tensor:
    if not values:
        return torch.zeros(0, device=device, dtype=dtype)
    x = torch.tensor([float(v) for v in values], device=device, dtype=dtype)
    if x.numel() > 1:
        std = x.std(unbiased=False)
        if float(std.detach().item()) > 1e-8:
            x = (x - x.mean()) / (std + 1e-6)
        else:
            x = x - x.mean()
    if clip and clip > 0:
        x = x.clamp(-float(clip), float(clip))
    return x
