from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Tuple

try:
    from answer_specs import parse_answer_spec, final_answer_score, normalize_answer
except Exception:  # allow standalone import in quick tests
    parse_answer_spec = None
    final_answer_score = None
    def normalize_answer(x):
        return str(x or "").strip()

MASK_CHARS = {"□", "[MASK]", "<mask>", "<|mask|>"}
BAD_TEXT_RE = re.compile(r"(�|\\\\|\\[a-zA-Z]{2,}|<\|endoftext\|>|\n\n)")
NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")
INTERVAL_RE = re.compile(r"^([\(\[])([+-]?(?:\d+(?:\.\d+)?|\.\d+)),([+-]?(?:\d+(?:\.\d+)?|\.\d+))([\)\]])$")
EQUATION_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*=.+$")
UNIT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*(m|mm|cm|km|kg|g|s|ms|N|J|W|V|A|Hz|m/s|m/s\^2)$")


@dataclass(frozen=True)
class StageWeights:
    slot: float
    skeleton: float
    exact: float
    clean: float
    keep_mask: float


def stage_name(t: float, early_t: float = 0.70, late_t: float = 0.30) -> str:
    if t >= early_t:
        return "early"
    if t <= late_t:
        return "late"
    return "middle"


def weights_for_t(t: float) -> StageWeights:
    stage = stage_name(t)
    if stage == "early":
        # High-noise stage: do not force exact values.  Prefer safe slot decisions.
        return StageWeights(slot=0.55, skeleton=0.10, exact=0.05, clean=0.20, keep_mask=0.10)
    if stage == "middle":
        # Mid-stage: structure should emerge.
        return StageWeights(slot=0.25, skeleton=0.40, exact=0.20, clean=0.15, keep_mask=0.00)
    # Low-noise stage: final correction.
    return StageWeights(slot=0.10, skeleton=0.25, exact=0.55, clean=0.10, keep_mask=-0.10)


def compact(text: str) -> str:
    return re.sub(r"\s+", "", normalize_answer(str(text or "")))


def token_class(text: str) -> str:
    s = str(text or "")
    c = compact(s)
    if c in MASK_CHARS or c == "":
        return "mask"
    if c in {"(", "["}:
        return "left_bracket"
    if c in {")","]",
    }:
        return "right_bracket"
    if c == ",":
        return "comma"
    if c == "=":
        return "equal"
    if c in {"+", "-", "*", "/", "^"}:
        return "operator"
    if c == ".":
        return "dot"
    if c.isdigit() or NUMBER_RE.match(c):
        return "number"
    if c in {"m", "mm", "cm", "km", "kg", "g", "s", "ms", "N", "J", "W", "V", "A", "Hz", "m/s", "m/s^2"}:
        return "unit"
    if re.fullmatch(r"[A-Za-z]+", c):
        return "letter"
    if c.startswith("\\"):
        return "latex"
    return "other"


def slot_group(expected_token_text: str, answer_kind: str = "") -> str:
    cls = token_class(expected_token_text)
    if cls in {"left_bracket", "right_bracket"}:
        return "edge_symbol"
    if cls in {"comma", "equal"}:
        return "separator"
    if cls in {"operator", "dot"}:
        return "operator"
    if cls == "unit":
        return "unit"
    if cls == "number":
        return "value"
    if cls == "letter":
        # In equations, variable slots are structural; in option answers, exact letter is the value.
        return "variable" if answer_kind in {"equation", "symbolic_expression", "short_text"} else "value"
    return cls


def slot_readiness_weight(t: float, expected_token_text: str, answer_kind: str = "") -> float:
    """How much this slot should be learned at this diffusion time.

    This implements the exploration hypothesis:
    early: recover symbols / boundaries first;
    middle: align skeleton;
    late: recover exact numeric/content values.
    """
    stage = stage_name(t)
    group = slot_group(expected_token_text, answer_kind)
    if stage == "early":
        if group in {"edge_symbol", "separator", "operator", "unit"}:
            return 1.35
        if group in {"variable"}:
            return 1.00
        if group in {"value"}:
            return 0.45
        return 0.65
    if stage == "middle":
        if group in {"edge_symbol", "separator", "operator", "unit", "variable"}:
            return 1.10
        if group in {"value"}:
            return 0.90
        return 0.80
    # late
    return 1.00


def clean_score(answer_text: str) -> float:
    s = str(answer_text or "")
    if not s:
        return 0.0
    if BAD_TEXT_RE.search(s):
        return -1.0
    # Pure punctuation garbage.
    no_mask = s.replace("□", "").strip()
    if no_mask and not re.search(r"[A-Za-z0-9]", no_mask) and len(no_mask) >= 2:
        return -0.7
    # Bad mixed closing punctuation such as 3）5》.
    if re.search(r"[）》《；：，。！？]", s):
        return -0.8
    return 1.0


def slot_token_score(candidate_token_text: str, expected_token_text: str, t: float, answer_kind: str = "") -> float:
    cand = compact(candidate_token_text)
    exp = compact(expected_token_text)
    cand_cls = token_class(cand)
    exp_cls = token_class(exp)
    stage = stage_name(t)

    if cand == exp and exp:
        return 1.0

    if cand_cls == "mask":
        # Early: keeping mask is better than filling garbage. Late: mask is bad.
        if stage == "early":
            return 0.35
        if stage == "middle":
            return 0.05
        return -0.35

    # Same structural class is useful before exact correction.
    if cand_cls == exp_cls and exp_cls not in {"other", "latex"}:
        if stage == "early":
            # For numeric value slots, exact value should wait; any number is weakly okay.
            return 0.55 if exp_cls == "number" else 0.75
        if stage == "middle":
            return 0.65
        return 0.35

    # For interval endpoints, numbers are allowed in value slots but not in bracket/comma slots.
    if exp_cls == "number" and cand_cls in {"number", "dot", "operator"}:
        return 0.25 if stage != "late" else -0.10

    # Bad LaTeX / word fragments should be discouraged strongly in structured answers.
    if cand_cls in {"latex", "other"} and answer_kind in {"interval", "equation", "unit_decimal", "signed_decimal", "single_integer"}:
        return -0.75

    return -0.35


def _contains_in_order(s: str, chars: str) -> float:
    j = 0
    for ch in s:
        if j < len(chars) and ch == chars[j]:
            j += 1
    return j / max(1, len(chars))


def skeleton_score(answer_text: str, gt_answer: str) -> float:
    """Structure score for partial decoded answer states, in roughly [-1, 1]."""
    s = compact(answer_text).replace("□", "")
    gt = compact(gt_answer)
    if not s:
        return 0.0
    if s == gt:
        return 1.0

    # Interval: reward correct structural slots, not just token set.
    m = INTERVAL_RE.match(gt)
    if m:
        lb, left, right, rb = m.groups()
        score = 0.0
        # Edge symbols are more important than random numeric overlap.
        if len(s) >= 1:
            score += 0.18 if s[0] == lb else (-0.10 if s[0] in "([0123456789+-" else 0.0)
        if len(s) >= 1:
            score += 0.18 if s[-1] == rb else (-0.08 if s[-1] in ")]" else 0.0)
        if "," in s:
            score += 0.18
            l, r = s.split(",", 1)
            if left in l:
                score += 0.16
            if right in r:
                score += 0.16
        else:
            # weak ordered skeleton signal, e.g. "(3" before comma appears.
            score += 0.12 * _contains_in_order(s, lb + left)
            score += 0.12 * _contains_in_order(s, right + rb)
        if re.fullmatch(r"[\(\[][+-]?[0-9.]+,[+-]?[0-9.]+[\)\]]", s):
            score += 0.14
        return max(-0.4, min(1.0, score))

    # Unit decimal: number must come before unit; repeated number without unit is not enough.
    if UNIT_RE.match(gt):
        mg = UNIT_RE.match(gt)
        unit = mg.group(1) if mg else ""
        score = 0.0
        if re.search(r"[+-]?\d", s):
            score += 0.20
        if "." in gt and "." in s:
            score += 0.15
        if unit and unit in s:
            score += 0.30
            # Unit should be after the number, not before.
            if re.search(rf"\d.*{re.escape(unit)}$", s):
                score += 0.15
        if UNIT_RE.match(s):
            score += 0.20
        return max(-0.4, min(1.0, score))

    # Equation / symbolic: prioritize separators/operators in correct rough order.
    if "=" in gt:
        score = 0.0
        if "=" in s:
            score += 0.35
            gl, gr = gt.split("=", 1)
            sl, sr = s.split("=", 1)
            if gl and sl and gl[0] == sl[0]:
                score += 0.20
            common_rhs = len(set(gr) & set(sr)) / max(1, len(set(gr)))
            score += 0.30 * common_rhs
        common = len(set(gt) & set(s)) / max(1, len(set(gt)))
        score += 0.15 * common
        return max(-0.4, min(1.0, score))

    # Fallback: ordered character overlap.
    return max(0.0, min(1.0, _contains_in_order(s, gt)))


def exact_score(answer_text: str, gt_answer: str) -> float:
    if parse_answer_spec is not None and final_answer_score is not None:
        try:
            pred = parse_answer_spec(answer_text)
            gt = parse_answer_spec(gt_answer)
            return max(-1.0, min(1.0, float(final_answer_score(pred, gt))))
        except Exception:
            pass
    return 1.0 if compact(answer_text) == compact(gt_answer) else 0.0


def state_score(answer_text: str, gt_answer: str, t: float) -> float:
    w = weights_for_t(t)
    clean = clean_score(answer_text)
    skel = skeleton_score(answer_text, gt_answer)
    exact = exact_score(answer_text, gt_answer)
    # Slot score is unavailable for full text, so use skeleton as a proxy here.
    return float(w.skeleton * skel + w.exact * exact + w.clean * clean)


def transition_candidate_reward(
    before_answer: str,
    after_answer: str,
    gt_answer: str,
    t: float,
    candidate_token_text: str,
    expected_token_text: str,
    answer_kind: str = "",
    delta_weight: float = 0.65,
    absolute_weight: float = 0.35,
) -> Tuple[float, Dict[str, float | str]]:
    """Reward for one candidate local action.

    The important difference from final-answer reward:
    reward is attached to the transition x_t -> x_{t-dt}, so it can say
    whether this specific step improves the current answer state.
    """
    before = state_score(before_answer, gt_answer, t)
    after = state_score(after_answer, gt_answer, max(0.0, t - 1e-3))
    delta = after - before

    slot = slot_token_score(candidate_token_text, expected_token_text, t, answer_kind)
    ready = slot_readiness_weight(t, expected_token_text, answer_kind)
    w = weights_for_t(t)
    clean = clean_score(after_answer)
    skel = skeleton_score(after_answer, gt_answer)
    exact = exact_score(after_answer, gt_answer)

    absolute = (
        w.slot * ready * slot
        + w.skeleton * skel
        + w.exact * exact
        + w.clean * clean
    )
    reward = delta_weight * delta + absolute_weight * absolute
    reward = max(-1.0, min(1.0, float(reward)))
    return reward, {
        "stage": stage_name(t),
        "before_score": float(before),
        "after_score": float(after),
        "delta_score": float(delta),
        "slot_score": float(slot),
        "slot_readiness": float(ready),
        "skeleton_score": float(skel),
        "exact_score": float(exact),
        "clean_score": float(clean),
    }
