from __future__ import annotations

"""QRA-refinement reward for rollout-chain RL.

This reward is intentionally different from the pretrain-oriented
``slot_alignment_reward``.  The pretrain reward mainly teaches coarse slot and
skeleton structure.  This module assumes the QRA/SFT model already has useful
partial chains, so it tries to preserve good partial states while correcting
misplaced / shifted / malformed tokens.

Main design principles:
  * exact token in the correct slot is the strongest signal at every timestep;
  * a GT token in the wrong slot is only mildly penalized early, but strongly
    penalized late if it remains unresolved;
  * wrong non-GT values are penalized from the beginning;
  * flexible state alignment is token-aware: a BPE token such as "mx" is
    not scored as a clean "x" plus a free extra "m"; matched units from
    contaminated source tokens receive only partial credit;
  * control characters such as real newlines/tabs are strongly penalized before
    any normalization, so "\ny=2x+" cannot be rewarded as "y=2x+";
  * decimal/unit answers get an ordered numeric shaping term, e.g. 90 -> 90. ->
    90.4 is treated as a good partial path for GT 90.39, while .9939 is bad;
  * equation/symbolic answers allow small extra tokens but cap their score when
    they introduce internal unknown variables, e.g. y=2mx+1 is a mild error, not
    a near-exact answer.
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

    QRA-refinement is exact-first: exact token matches are valuable at every
    stage because late correction is empirically weak.  Early stage still keeps
    a large action-alignment term so the chain is guided before it freezes.
    """
    stage = stage_from_t(t)
    kind = str(answer_kind or "")
    if kind in {"single_letter", "single_integer", "letter", "integer"}:
        if stage == "early":
            return StageCoefficients(0.48, 0.02, 0.40, 0.07, 0.03)
        if stage == "middle":
            return StageCoefficients(0.34, 0.02, 0.56, 0.05, 0.01)
        return StageCoefficients(0.22, 0.01, 0.72, 0.04, 0.01)
    if stage == "early":
        return StageCoefficients(action_align=0.78, skeleton_delta=0.06, exact_delta=0.18, clean_action=0.10, mask_action=0.03)
    if stage == "middle":
        return StageCoefficients(action_align=0.55, skeleton_delta=0.18, exact_delta=0.30, clean_action=0.06, mask_action=0.02)
    return StageCoefficients(action_align=0.28, skeleton_delta=0.08, exact_delta=0.60, clean_action=0.04, mask_action=0.00)


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
    if kind in {"single_letter", "single_integer", "letter", "integer"}:
        return 1.0
    if stage == "early":
        if group in {"left_bracket", "right_bracket", "separator", "equals", "operator", "unit"}:
            return 1.80
        if group in {"variable", "latex_atom"}:
            return 1.20
        if group == "value":
            # QRA refinement: value slots are important from the beginning.
            # Exact value tokens should be protected before the chain freezes.
            return 1.25
        return 0.80
    if stage == "middle":
        if group in {"left_bracket", "right_bracket", "separator", "equals", "operator", "unit"}:
            return 1.35
        if group in {"variable", "latex_atom"}:
            return 1.10
        if group == "value":
            return 1.15
        return 1.00
    if group == "value":
        return 1.20
    return 1.00


def compatible_score_by_stage(t: float, cur_kind: str, gt_kind: str, answer_kind: str = "") -> float:
    stage = stage_from_t(t)
    kind = str(answer_kind or "")

    # QRA refinement: wrong non-GT numeric values should not be rewarded just
    # because the slot type is numeric.  Correct numeric values are handled by
    # the exact branch in action_alignment_score.  A GT token in the wrong slot
    # is handled separately as misplaced-token shaping.
    if gt_kind == "number" and cur_kind == "number":
        if stage == "early":
            return -0.55
        if stage == "middle":
            return -0.75
        return -1.00

    if kind in {"single_letter", "letter"} and cur_kind in {"letter", "variable"} and gt_kind in {"letter", "variable"}:
        if stage == "early":
            return -0.10
        if stage == "middle":
            return -0.35
        return -0.75

    if stage == "early":
        # For structural slots, correct type is useful early, but exact is still
        # preferred. For value slots, being the same broad type is not enough.
        if gt_kind in {"left_bracket", "right_bracket", "separator", "equals", "operator", "unit"}:
            return 0.55
        return -0.15
    if stage == "middle":
        if gt_kind in {"left_bracket", "right_bracket", "separator", "equals", "operator", "unit"}:
            return 0.25
        return -0.30
    # Late: same type but wrong exact should be corrected.
    if gt_kind in {"letter", "variable"}:
        return -0.55
    return -0.45


def misplaced_action_penalty_by_stage(t: float) -> float:
    """Penalty for selecting a GT token but putting it in the wrong slot.

    Early misplaced GT tokens are not treated as catastrophic: they at least
    contain useful GT information and may later be fixed by replacement.  By the
    late stage, however, unresolved misplacement is a serious error.
    """
    stage = stage_from_t(t)
    if stage == "early":
        return -0.35
    if stage == "middle":
        return -0.75
    return -1.10

def action_alignment_score(
    action_answer_index: int,
    token_id: int,
    gt_ids: Sequence[int],
    tokenizer,
    mask_id: int,
    t: float,
    answer_kind: str = "",
) -> Tuple[float, Dict[str, Any]]:
    """Direct token-slot reward for the selected BPE token.

    This is token-aware: if a single sampled token is "mx" and the GT slot is
    "x", it receives only partial credit and a contamination penalty.  We do not
    split it into a clean "m" slot plus a clean "x" slot.
    """
    idx = int(action_answer_index)
    token_id = int(token_id)
    if idx < 0 or idx >= len(gt_ids):
        return -1.0, {"reason": "bad_action_index", "slot_importance": 1.0}

    gt_id = int(gt_ids[idx])
    tok_text = decode_token(tokenizer, token_id)
    gt_text = decode_token(tokenizer, gt_id)
    gt_full_text = answer_text_from_ids(gt_ids, tokenizer, mask_id)
    cur_kind = token_kind(tok_text)
    gt_kind = token_kind(gt_text)
    importance = slot_importance(t, gt_text, answer_kind)
    wrong_position_token = float((token_id in {int(x) for x in gt_ids}) and token_id != gt_id)
    stage = stage_from_t(t)

    token_ctrl = control_penalty_text(tok_text)
    gt_units = units_from_text(gt_text)
    tok_units = _parse_token_units(tok_text, 0)
    # Does this BPE token contain the expected GT slot unit plus extra units?
    contains_gt_unit = False
    contaminated_token = False
    if gt_units and tok_units:
        for tu in tok_units:
            if unit_match_score(tu, gt_units[0]) > 0.5:
                contains_gt_unit = True
        raw_norm = norm_text(tok_text).replace(" ", "")
        gt_norm = norm_text(gt_text).replace(" ", "")
        contaminated_token = contains_gt_unit and bool(raw_norm and gt_norm and raw_norm != gt_norm)

    # Unknown variables in an action token are bad for equation-like answers,
    # but a token like "mx" containing a correct x is a mild error, not garbage.
    unk_action = unknown_variable_penalty(tok_text, gt_full_text, answer_kind)

    if is_mask_token(token_id, mask_id):
        base = 0.04 if stage == "early" else (-0.15 if stage == "middle" else -0.55)
    elif token_ctrl > 0:
        base = -1.20
    elif is_garbage_text(tok_text):
        base = -1.0
    elif token_id == gt_id:
        # Exact slot-token match is the highest-priority event in every stage.
        base = 1.45
        importance = max(float(importance), 1.15)
    elif semantic_match(tok_text, gt_text):
        base = 1.08
        importance = max(float(importance), 1.05)
    elif contaminated_token:
        # Example: GT slot is "x" but sampled BPE token is "mx".  It contains
        # useful GT information, but should not be treated as a clean correct
        # slot.  It becomes increasingly unacceptable late.
        if stage == "early":
            base = 0.08
        elif stage == "middle":
            base = -0.18
        else:
            base = -0.38
    elif compatible_kind(cur_kind, gt_kind):
        base = compatible_score_by_stage(t, cur_kind, gt_kind, answer_kind)
    else:
        if wrong_position_token:
            base = misplaced_action_penalty_by_stage(t)
        elif cur_kind in {"number", "letter", "variable"} and gt_kind not in {"number", "letter", "variable"}:
            base = -0.78
        elif gt_kind in {"number", "letter", "variable"} and cur_kind not in {"number", "letter", "variable"}:
            base = -0.70
        else:
            base = -0.55

    # Additional token-level penalties.  Keep them small for contaminated tokens
    # so y=2mx+1 is a mild/moderate error, but controls are strong errors.
    if token_ctrl > 0:
        base -= 0.35 * token_ctrl
    if unk_action > 0 and not contaminated_token:
        base -= (0.12 if stage == "early" else 0.25 if stage == "middle" else 0.40) * unk_action
    elif unk_action > 0 and contaminated_token:
        base -= (0.04 if stage == "early" else 0.10 if stage == "middle" else 0.18) * unk_action

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
        "token_control_penalty": float(token_ctrl),
        "token_unknown_variable_penalty": float(unk_action),
        "contaminated_token": float(1.0 if contaminated_token else 0.0),
    }

def clean_action_score(token_id: int, tokenizer, mask_id: int, t: float) -> float:
    if is_mask_token(token_id, mask_id):
        return 0.0
    txt = decode_token(tokenizer, int(token_id))
    ctrl = control_penalty_text(txt)
    if ctrl > 0:
        return float(max(-1.0, -0.65 - 0.25 * ctrl))
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




def answer_text_from_ids(ids: Sequence[int], tokenizer, mask_id: int) -> str:
    pieces: List[str] = []
    for tok in ids:
        ti = int(tok)
        if ti == int(mask_id) or ti < 0:
            continue
        pieces.append(decode_token(tokenizer, ti))
    return "".join(pieces)


def compact_math_text(text: str) -> str:
    s = normalize_answer_text(text)
    s = s.replace("□", "")
    s = s.replace("<|endoftext|>", "")
    return s


def _first_number_like(text: str) -> str:
    s = compact_math_text(text)
    m = re.search(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", s)
    return m.group(0) if m else ""


def _lcs_len(a: str, b: str) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for ca in a:
        prev = 0
        for j, cb in enumerate(b, start=1):
            old = dp[j]
            if ca == cb:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = old
    return dp[-1]


def decimal_order_score(answer_text: str, gt_text: str) -> float:
    """Decimal/unit numeric partial-state score in [-1, 1].

    This explicitly distinguishes good ordered partial states such as
    ``90``, ``90.``, ``90.4`` for GT ``90.39`` from bad shifted states such
    as ``.9939`` or ``8790.``.  It is used as state-potential shaping, not as
    a replacement for exact token match.
    """
    gt_num = _first_number_like(gt_text)
    cur_num = _first_number_like(answer_text)
    if not gt_num or "." not in gt_num:
        return 0.0
    if not cur_num:
        return 0.0

    gt_signless = gt_num[1:] if gt_num[:1] in "+-" else gt_num
    cur_signless = cur_num[1:] if cur_num[:1] in "+-" else cur_num
    gt_int = gt_signless.split(".", 1)[0]
    gt_digits = re.sub(r"\D", "", gt_signless)
    cur_digits = re.sub(r"\D", "", cur_signless)

    # Prefix over the full numeric string, including the decimal point.
    lcp = 0
    for a, b in zip(cur_signless, gt_signless):
        if a != b:
            break
        lcp += 1
    prefix_score = lcp / max(1, len(gt_signless))

    # Integer part should appear at the beginning, not shifted after a dot.
    if gt_int and cur_signless.startswith(gt_int):
        int_score = 1.0
    elif gt_int and cur_signless and gt_int.startswith(cur_signless.rstrip(".")):
        int_score = 0.55
    elif gt_int and gt_int in cur_signless:
        int_score = -0.10
    else:
        int_score = -0.35

    gt_dot = gt_signless.find(".")
    cur_dot = cur_signless.find(".")
    if gt_dot >= 0:
        if cur_dot == gt_dot:
            dot_score = 1.0
        elif cur_dot < 0:
            dot_score = 0.10 if prefix_score > 0 else -0.05
        else:
            dot_score = -0.75
    else:
        dot_score = 0.0

    ordered = _lcs_len(cur_digits, gt_digits) / max(1, len(gt_digits))
    valid = 1.0 if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", cur_num) else -0.30
    if gt_int and gt_signless.startswith(gt_int) and cur_signless.startswith("."):
        valid -= 0.70

    score = 0.40 * prefix_score + 0.25 * int_score + 0.20 * dot_score + 0.10 * ordered + 0.05 * valid
    return float(max(-1.0, min(1.0, score)))



def token_lcs_score(cur_ids: Sequence[int], gt_ids: Sequence[int], mask_id: int) -> float:
    """Flexible token-order score in [0, 1].

    Unlike exact_score, this tolerates shifts.  For example, if deleting one bad
    token makes the remaining equation tokens align, this score increases.  This
    is important because our action space has no insert/swap operation; a bad
    token can only be replaced (often by mask/empty) and then re-filled later.
    """
    cur = [int(x) for x in cur_ids if int(x) != int(mask_id) and int(x) >= 0]
    gt = [int(x) for x in gt_ids if int(x) != int(mask_id) and int(x) >= 0]
    if not gt:
        return 0.0
    if not cur:
        return 0.0
    dp = [0] * (len(gt) + 1)
    for c in cur:
        prev = 0
        for j, g in enumerate(gt, start=1):
            old = dp[j]
            if c == g:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = old
    return float(dp[-1] / max(1, len(gt)))


def text_lcs_score(answer_text: str, gt_text: str) -> float:
    cur = compact_math_text(answer_text)
    gt = compact_math_text(gt_text)
    if not gt or not cur:
        return 0.0
    return float(_lcs_len(cur, gt) / max(1, len(gt)))


def operator_alignment_score(answer_text: str, gt_text: str) -> float:
    """Flexible score for equations/inequalities/operators in [0, 1]."""
    cur = compact_math_text(answer_text)
    gt = compact_math_text(gt_text)
    if not gt:
        return 0.0
    ops = ["<=", ">=", "<", ">", "=", "+", "-", "*", "/", "^"]
    gt_ops = [op for op in ops if op in gt]
    if not gt_ops:
        return 0.0
    hits = 0.0
    for op in gt_ops:
        if op not in cur:
            continue
        gt_pos = gt.find(op) / max(1, len(gt) - 1)
        cur_pos = cur.find(op) / max(1, len(cur) - 1)
        # position-tolerant but rewards the operator landing in roughly the
        # same relative place, e.g. '=' in y=2x+1.
        hits += max(0.0, 1.0 - abs(gt_pos - cur_pos) * 2.0)
    return float(hits / max(1, len(gt_ops)))


def valid_compact_form_score(answer_text: str, gt_text: str, answer_kind: str = "") -> float:
    """Coarse validity score in [-1, 1]."""
    cur = compact_math_text(answer_text)
    gt = compact_math_text(gt_text)
    kind = str(answer_kind or "")
    if not cur:
        return 0.0
    if kind in {"decimal", "unit_decimal"} or re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+).*", gt):
        # If GT is not dot-leading, a dot-leading current answer is suspicious.
        if "." in gt and not gt.startswith(".") and cur.startswith("."):
            return -0.8
        if re.search(r"\d", cur) and cur.count(".") <= max(1, gt.count(".")):
            return 0.4
        return -0.4
    if kind in {"equation", "inequality", "symbolic"} or any(op in gt for op in ["=", "<", ">", "/", "+", "-"]):
        # Penalize repeated operators like '..', '==', or operators at the wrong
        # extreme unless the GT has that shape.
        if re.search(r"(\.\.|==|<<|>>)", cur):
            return -0.6
        if any(op in gt and op in cur for op in ["=", "<", ">", "/", "+", "-"]):
            return 0.4
        return -0.2
    return 0.0




# ---------------------- token-aware decoded alignment ----------------------

CONTROL_CHARS = {"\n", "\r", "\t"}
CONTROL_MARKERS = {"<|endoftext|>", "<eos>", "</s>"}


def has_control_text(text: str) -> bool:
    s = str(text or "")
    return any(ch in s for ch in CONTROL_CHARS) or any(m in s for m in CONTROL_MARKERS)


def control_penalty_text(text: str) -> float:
    """Penalty in [0, 1+] for raw control characters / EOS-like tokens.

    Important: this is computed before strip/normalization.  Therefore a raw
    answer such as "\ny=2x+" is not silently converted to "y=2x+".
    """
    s = str(text or "")
    if not s:
        return 0.0
    hits = 0
    for ch in CONTROL_CHARS:
        hits += s.count(ch)
    for marker in CONTROL_MARKERS:
        hits += s.count(marker)
    # Also catch common bad natural-language fragments in short answer slots.
    low = norm_text(s).lower()
    phrase_bad = 1 if any(w in low for w in ["bottom line", " line", "therefore", "answer:", "endoftext"]) else 0
    return float(min(1.5, 0.35 * hits + 0.55 * phrase_bad))


def compact_keep_control(text: str) -> str:
    """Compact math text while preserving control characters as visible markers."""
    s = str(text or "")
    s = s.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    s = s.replace("□", "")
    s = s.replace("<|endoftext|>", "<eot>")
    s = s.replace("Ġ", " ").replace("▁", " ").replace("\u00a0", " ")
    s = s.replace("\\left", "").replace("\\right", "")
    return re.sub(r"\s+", "", s)


def _parse_token_units(token_text: str, source_index: int) -> List[Dict[str, Any]]:
    """Split a decoded BPE token into math units while preserving source token.

    This prevents scoring a single token such as "mx" as if it were two clean
    independent slots.  The units can match GT units, but the source token is
    still marked as contaminated when it contains extra unmatched units.
    """
    raw = str(token_text or "")
    units: List[Dict[str, Any]] = []
    # Preserve controls as explicit units.
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch in CONTROL_CHARS:
            marker = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}.get(ch, repr(ch))
            units.append({"text": marker, "kind": "control", "source": source_index, "raw": raw})
            i += 1
            continue
        # Skip ordinary spaces, but do not skip newlines/tabs above.
        if ch.isspace():
            i += 1
            continue
        # Multi-character comparators.
        if raw.startswith("<=", i) or raw.startswith(">=", i):
            units.append({"text": raw[i:i+2], "kind": "operator", "source": source_index, "raw": raw})
            i += 2
            continue
        # Signed/unsigned numbers, including decimals.
        m = re.match(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", raw[i:])
        if m:
            txt = m.group(0)
            units.append({"text": txt, "kind": "number", "source": source_index, "raw": raw})
            i += len(txt)
            continue
        # Latex commands as units.
        m = re.match(r"\\[A-Za-z]+", raw[i:])
        if m:
            txt = m.group(0)
            units.append({"text": txt, "kind": "latex", "source": source_index, "raw": raw})
            i += len(txt)
            continue
        if ch.isalpha():
            # Each variable letter is a separate semantic unit, but source keeps
            # whether it came from one BPE token such as "mx".
            units.append({"text": ch, "kind": "variable", "source": source_index, "raw": raw})
            i += 1
            continue
        if ch in "()[]{}":
            units.append({"text": ch, "kind": "bracket", "source": source_index, "raw": raw})
            i += 1
            continue
        if ch in ",;":
            units.append({"text": ch, "kind": "separator", "source": source_index, "raw": raw})
            i += 1
            continue
        if ch in "=+*/^<>.-:":
            # A decimal point inside a number is captured by the number regex.
            kind = "equals" if ch == "=" else ("operator" if ch in "+*/^<>-" else "separator")
            units.append({"text": ch, "kind": kind, "source": source_index, "raw": raw})
            i += 1
            continue
        units.append({"text": ch, "kind": "other", "source": source_index, "raw": raw})
        i += 1
    if not units and raw:
        units.append({"text": norm_text(raw), "kind": token_kind(raw), "source": source_index, "raw": raw})
    return units


def units_from_ids(ids: Sequence[int], tokenizer, mask_id: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for src, tok in enumerate(ids):
        ti = int(tok)
        if ti == int(mask_id) or ti < 0:
            continue
        out.extend(_parse_token_units(decode_token(tokenizer, ti), src))
    return out


def units_from_text(text: str) -> List[Dict[str, Any]]:
    return _parse_token_units(str(text or ""), 0)


def unit_match_score(pred_u: Dict[str, Any], gt_u: Dict[str, Any]) -> float:
    pt, gt = str(pred_u.get("text", "")), str(gt_u.get("text", ""))
    pk, gk = str(pred_u.get("kind", "")), str(gt_u.get("kind", ""))
    if pt == gt:
        return 1.0
    # Numeric partial containment should be weak, not full.  For example token
    # "90" is not a clean match for GT token "9" unless the GT unit is actually
    # "90".  We mostly rely on decimal_order_score for numeric partials.
    if pk == gk == "number" and (pt in gt or gt in pt):
        return 0.45
    if pk == gk and pk in {"variable", "operator", "equals", "bracket", "separator"}:
        return 0.15
    return -1e9


def token_aware_unit_lcs_score(cur_ids: Sequence[int], gt_ids: Sequence[int], tokenizer, mask_id: int) -> Tuple[float, float, float]:
    """Return (match_score, contaminated_match_frac, control_frac).

    The match_score is an order-preserving unit alignment normalized by GT unit
    count.  Matched units from source tokens containing extra units get reduced
    credit.  Thus a BPE token "mx" matched to GT unit "x" receives partial
    credit and contributes to contamination.
    """
    pred_units = units_from_ids(cur_ids, tokenizer, mask_id)
    gt_units = units_from_ids(gt_ids, tokenizer, mask_id)
    if not gt_units:
        return 0.0, 0.0, 0.0
    if not pred_units:
        return 0.0, 0.0, 0.0

    source_counts: Dict[int, int] = {}
    for u in pred_units:
        source_counts[int(u.get("source", -1))] = source_counts.get(int(u.get("source", -1)), 0) + 1

    n, m = len(pred_units), len(gt_units)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    take = [[False] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = max(dp[i - 1][j], dp[i][j - 1])
            ms = unit_match_score(pred_units[i - 1], gt_units[j - 1])
            if ms > -1e8:
                src = int(pred_units[i - 1].get("source", -1))
                raw_norm = norm_text(str(pred_units[i - 1].get("raw", ""))).replace(" ", "")
                unit_norm = norm_text(str(pred_units[i - 1].get("text", ""))).replace(" ", "")
                contaminated = source_counts.get(src, 0) > 1 or (raw_norm and unit_norm and raw_norm != unit_norm)
                credit = ms * (0.62 if contaminated else 1.0)
                cand = dp[i - 1][j - 1] + credit
                if cand > best:
                    best = cand
                    take[i][j] = True
            dp[i][j] = best

    # Backtrack matched sources to compute contamination fraction.
    i, j = n, m
    matched = 0
    contaminated_hits = 0
    while i > 0 and j > 0:
        if take[i][j] and dp[i][j] > max(dp[i - 1][j], dp[i][j - 1]) - 1e-9:
            u = pred_units[i - 1]
            src = int(u.get("source", -1))
            raw_norm = norm_text(str(u.get("raw", ""))).replace(" ", "")
            unit_norm = norm_text(str(u.get("text", ""))).replace(" ", "")
            contaminated = source_counts.get(src, 0) > 1 or (raw_norm and unit_norm and raw_norm != unit_norm)
            matched += 1
            contaminated_hits += 1 if contaminated else 0
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    controls = sum(1 for u in pred_units if u.get("kind") == "control")
    return (
        float(max(0.0, min(1.0, dp[n][m] / max(1, len(gt_units))))),
        float(contaminated_hits / max(1, matched)),
        float(controls / max(1, len(pred_units))),
    )


def levenshtein_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, start=1):
            old = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if ca == cb else 1))
            prev = old
    dist = dp[-1]
    return float(max(0.0, 1.0 - dist / max(1, max(len(a), len(b)))))


def prefix_partial_score(answer_text: str, gt_text: str) -> float:
    cur = compact_keep_control(answer_text)
    gt = compact_keep_control(gt_text)
    if not cur or not gt:
        return 0.0
    # If control marker is present, do not let a stripped prefix look good.
    if "\\n" in cur or "\\t" in cur or "\\r" in cur:
        return 0.0
    lcp = 0
    for a, b in zip(cur, gt):
        if a != b:
            break
        lcp += 1
    return float(lcp / max(1, len(gt)))


def _letters(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]", compact_keep_control(text))


def unknown_variable_penalty(answer_text: str, gt_text: str, answer_kind: str = "") -> float:
    kind = str(answer_kind or "")
    if kind not in {"equation", "inequality", "symbolic"} and not any(op in str(gt_text) for op in ["=", "<", ">", "+", "-"]):
        return 0.0
    gt_vars = set(_letters(gt_text))
    if not gt_vars:
        return 0.0
    pred_vars = _letters(answer_text)
    extras = [v for v in pred_vars if v not in gt_vars]
    return float(min(1.0, len(extras) / max(1, len(gt_vars))))


def internal_extra_penalty(answer_text: str, gt_text: str) -> float:
    """Penalty for extra characters inside an otherwise ordered answer.

    Extra suffix is less severe than an internal insertion.  This handles
    y=2mx+1: the GT is almost present as a subsequence, but the internal 'm'
    should prevent a near-exact score.
    """
    cur = compact_keep_control(answer_text)
    gt = compact_keep_control(gt_text)
    if not cur or not gt:
        return 0.0
    # Controls are handled separately.
    cur_no_ctrl = cur.replace("\\n", "").replace("\\t", "").replace("\\r", "")
    if not cur_no_ctrl or not gt:
        return 0.0
    # Align greedily to identify unmatched characters in cur.
    j = 0
    unmatched_positions: List[int] = []
    matched_positions: List[int] = []
    for i, ch in enumerate(cur_no_ctrl):
        if j < len(gt) and ch == gt[j]:
            matched_positions.append(i)
            j += 1
        else:
            unmatched_positions.append(i)
    if not unmatched_positions:
        return 0.0
    if j < max(1, len(gt)) * 0.5:
        # If almost nothing matches, this is not a small internal-extra case.
        return min(1.0, len(unmatched_positions) / max(1, len(cur_no_ctrl)))
    first_m = min(matched_positions) if matched_positions else 0
    last_m = max(matched_positions) if matched_positions else -1
    internal = [p for p in unmatched_positions if first_m < p < last_m]
    suffix = [p for p in unmatched_positions if p > last_m]
    prefix = [p for p in unmatched_positions if p < first_m]
    # Internal extra is most damaging; suffix is mild.
    raw = 0.18 * len(prefix) + 0.55 * len(internal) + 0.22 * len(suffix)
    return float(min(1.0, raw / max(1, len(gt))))


def symbol_misplacement_penalty(answer_text: str, gt_text: str) -> float:
    cur = compact_keep_control(answer_text)
    gt = compact_keep_control(gt_text)
    if not cur or not gt:
        return 0.0
    # Do not compare controls as normal chars.
    if "\\n" in cur or "\\t" in cur or "\\r" in cur:
        # controls have their own stronger penalty, but they also make symbol
        # positions unreliable.
        cur_cmp = cur.replace("\\n", "").replace("\\t", "").replace("\\r", "")
    else:
        cur_cmp = cur
    symbols = ["<=", ">=", "=", "<", ">", "+", ",", ".", "(", ")", "[", "]"]
    penalties = []
    for sym in symbols:
        if sym not in gt:
            continue
        if sym not in cur_cmp:
            penalties.append(1.0)
            continue
        gt_pos = gt.find(sym) / max(1, len(gt) - 1)
        cur_pos = cur_cmp.find(sym) / max(1, len(cur_cmp) - 1)
        penalties.append(min(1.0, abs(gt_pos - cur_pos) * 2.5))
    return float(sum(penalties) / max(1, len(penalties)))


def token_contamination_penalty(cur_ids: Sequence[int], gt_ids: Sequence[int], tokenizer, mask_id: int) -> float:
    _, contaminated, control_frac = token_aware_unit_lcs_score(cur_ids, gt_ids, tokenizer, mask_id)
    return float(max(contaminated, control_frac))


def state_score_cap(answer_text: str, gt_text: str, answer_kind: str, t: float) -> float:
    """Maximum allowed score for known bad phenomena.

    Caps prevent a pure LCS score from giving y=2mx+1 or \ny=2x+ near-exact
    credit.  The cap is staged: late chain is stricter.
    """
    stage = stage_from_t(t)
    cap = 1.0
    ctrl = control_penalty_text(answer_text)
    if ctrl > 0:
        cap = min(cap, 0.45 if stage != "late" else 0.38)
    unk = unknown_variable_penalty(answer_text, gt_text, answer_kind)
    extra_internal = internal_extra_penalty(answer_text, gt_text)
    sym_mis = symbol_misplacement_penalty(answer_text, gt_text)
    # Unknown internal variable, e.g. y=2mx+1. Mild error but not exact.
    if unk > 0 and extra_internal > 0:
        cap = min(cap, 0.82 if stage != "late" else 0.76)
    elif unk > 0:
        cap = min(cap, 0.88 if stage != "late" else 0.82)
    if sym_mis > 0.45:
        cap = min(cap, 0.58 if stage != "late" else 0.50)
    return float(cap)

def flexible_state_score(
    cur_ids: Sequence[int],
    gt_ids: Sequence[int],
    tokenizer,
    mask_id: int,
    t: float,
    answer_kind: str = "",
) -> float:
    """Token-aware global state quality used as potential-based shaping.

    This term asks whether the decoded answer is closer to GT without treating a
    single contaminated BPE token such as "mx" as a clean independent "x".  It
    also strongly penalizes raw control characters such as newlines.
    """
    cur_text = answer_text_from_ids(cur_ids, tokenizer, mask_id)
    gt_text = answer_text_from_ids(gt_ids, tokenizer, mask_id)
    exact = exact_score(cur_ids, gt_ids, mask_id)
    skel = skeleton_score(cur_ids, gt_ids, tokenizer, mask_id, answer_kind)
    unit_lcs, contaminated_match, control_frac = token_aware_unit_lcs_score(cur_ids, gt_ids, tokenizer, mask_id)
    txt_lcs = text_lcs_score(cur_text, gt_text)
    edit_sim = levenshtein_similarity(compact_keep_control(cur_text), compact_keep_control(gt_text))
    prefix = prefix_partial_score(cur_text, gt_text)
    op_align = operator_alignment_score(cur_text, gt_text)
    numeric = decimal_order_score(cur_text, gt_text)
    valid = valid_compact_form_score(cur_text, gt_text, answer_kind)
    mis = misplaced_gt_token_penalty(cur_ids, gt_ids, mask_id)
    ctrl = control_penalty_text(cur_text)
    unk_var = unknown_variable_penalty(cur_text, gt_text, answer_kind)
    internal_extra = internal_extra_penalty(cur_text, gt_text)
    sym_mis = symbol_misplacement_penalty(cur_text, gt_text)
    stage = stage_from_t(t)

    if stage == "early":
        weights = {
            # Exact is already the largest positive state term early.  Prefix
            # and unit alignment help useful partial chains, but cannot outrank
            # correct slot-token matches.
            "exact": 0.34, "unit": 0.18, "edit": 0.08, "prefix": 0.16,
            "skel": 0.11, "struct": 0.10, "valid": 0.03,
            "mis": 0.10, "ctrl": 0.36, "unk": 0.10, "internal": 0.12, "contam": 0.10, "sym": 0.14,
        }
    elif stage == "middle":
        weights = {
            "exact": 0.45, "unit": 0.16, "edit": 0.08, "prefix": 0.08,
            "skel": 0.10, "struct": 0.09, "valid": 0.04,
            "mis": 0.22, "ctrl": 0.50, "unk": 0.20, "internal": 0.24, "contam": 0.18, "sym": 0.28,
        }
    else:
        weights = {
            "exact": 0.62, "unit": 0.10, "edit": 0.06, "prefix": 0.04,
            "skel": 0.07, "struct": 0.08, "valid": 0.03,
            "mis": 0.34, "ctrl": 0.68, "unk": 0.34, "internal": 0.38, "contam": 0.28, "sym": 0.42,
        }

    struct = max(numeric, op_align)
    positive = (
        weights["exact"] * exact
        + weights["unit"] * unit_lcs
        + weights["edit"] * edit_sim
        + weights["prefix"] * prefix
        + weights["skel"] * skel
        + weights["struct"] * struct
        + weights["valid"] * valid
    )
    penalty = (
        weights["mis"] * mis
        + weights["ctrl"] * ctrl
        + weights["unk"] * unk_var
        + weights["internal"] * internal_extra
        + weights["contam"] * contaminated_match
        + weights["sym"] * sym_mis
    )
    score = positive - penalty
    cap = state_score_cap(cur_text, gt_text, answer_kind, t)
    score = min(score, cap)
    return float(max(-1.0, min(1.0, score)))

def is_erasure_action(token_id: int, tokenizer, mask_id: int) -> bool:
    if is_mask_token(int(token_id), int(mask_id)):
        return True
    txt = decode_token(tokenizer, int(token_id))
    ns = norm_text(txt)
    return ns in {"", "<|endoftext|>", "</s>", "<eos>"}

def misplaced_gt_token_penalty(cur_ids: Sequence[int], gt_ids: Sequence[int], mask_id: int) -> float:
    """Fraction of filled slots containing a GT token in the wrong position."""
    n = min(len(cur_ids), len(gt_ids))
    if n <= 0:
        return 0.0
    gt_set = {int(x) for x in gt_ids[:n]}
    wrong = 0
    filled = 0
    for i, (c, g) in enumerate(zip(cur_ids[:n], gt_ids[:n])):
        ci = int(c)
        if ci == int(mask_id) or ci < 0:
            continue
        filled += 1
        if ci != int(g) and ci in gt_set:
            wrong += 1
    return float(wrong / max(1, n))


def extra_weights_by_stage(t: float) -> Tuple[float, float, float, float]:
    stage = stage_from_t(t)
    if stage == "early":
        # flexible_state_weight, numeric_order_weight, misplacement_weight, erasure_bonus_weight
        # Keep erasure conservative: the model rarely fixes late, so deleting a
        # correct or near-correct token should not be over-rewarded.
        return 0.28, 0.22, 0.22, 0.16
    if stage == "middle":
        return 0.42, 0.24, 0.45, 0.22
    return 0.62, 0.16, 0.65, 0.25

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

    gt_text_full = answer_text_from_ids(gt_ids, tokenizer, mask_id)
    before_text_full = answer_text_from_ids(before_ids, tokenizer, mask_id)
    after_text_full = answer_text_from_ids(after_ids, tokenizer, mask_id)
    numeric_before = decimal_order_score(before_text_full, gt_text_full)
    numeric_after = decimal_order_score(after_text_full, gt_text_full)
    numeric_delta = float(numeric_after - numeric_before)
    misplace_before = misplaced_gt_token_penalty(before_ids, gt_ids, mask_id)
    misplace_after = misplaced_gt_token_penalty(after_ids, gt_ids, mask_id)
    misplace_delta = float(misplace_before - misplace_after)
    flexible_before = flexible_state_score(before_ids, gt_ids, tokenizer, mask_id, t, answer_kind)
    flexible_after = flexible_state_score(after_ids, gt_ids, tokenizer, mask_id, t, answer_kind)
    flexible_delta = float(flexible_after - flexible_before)
    ctrl_before = control_penalty_text(before_text_full)
    ctrl_after = control_penalty_text(after_text_full)
    ctrl_delta = float(ctrl_before - ctrl_after)
    internal_extra_before = internal_extra_penalty(before_text_full, gt_text_full)
    internal_extra_after = internal_extra_penalty(after_text_full, gt_text_full)
    internal_extra_delta = float(internal_extra_before - internal_extra_after)
    unknown_var_before = unknown_variable_penalty(before_text_full, gt_text_full, answer_kind)
    unknown_var_after = unknown_variable_penalty(after_text_full, gt_text_full, answer_kind)
    unknown_var_delta = float(unknown_var_before - unknown_var_after)
    state_w, numeric_w, misplace_w, erase_w = extra_weights_by_stage(t)

    if action_answer_index is None or action_token_id is None:
        # Fallback for older callers: state-potential delta only.
        reward = float(state_w * flexible_delta + numeric_w * numeric_delta + misplace_w * misplace_delta)
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
            "numeric_before": numeric_before,
            "numeric_after": numeric_after,
            "numeric_delta": numeric_delta,
            "misplace_before": misplace_before,
            "misplace_after": misplace_after,
            "misplace_delta": misplace_delta,
            "flexible_before": flexible_before,
            "flexible_after": flexible_after,
            "flexible_delta": flexible_delta,
            "control_before": ctrl_before,
            "control_after": ctrl_after,
            "control_delta": ctrl_delta,
            "internal_extra_before": internal_extra_before,
            "internal_extra_after": internal_extra_after,
            "internal_extra_delta": internal_extra_delta,
            "unknown_var_before": unknown_var_before,
            "unknown_var_after": unknown_var_after,
            "unknown_var_delta": unknown_var_delta,
            "state_weight": state_w,
            "numeric_weight": numeric_w,
            "misplace_weight": misplace_w,
            "erase_weight": erase_w,
            "erase_bonus": 0.0,
            "reward": reward,
        }
    else:
        action_align, action_info = action_alignment_score(
            int(action_answer_index), int(action_token_id), gt_ids, tokenizer, mask_id, t, answer_kind
        )
        clean = clean_action_score(int(action_token_id), tokenizer, mask_id, t)
        mask = mask_action_score(int(action_token_id), mask_id, t)
        erase_bonus = 0.0
        if is_erasure_action(int(action_token_id), tokenizer, mask_id):
            # Conservative delete-like correction.  Erasure is only rewarded when
            # it removes a demonstrably bad phenomenon (misplacement, internal
            # extra variable, unknown variable, or control token) and does not
            # damage exact correctness.  This avoids teaching the model to shorten
            # already-good QRA chains.
            erase_signal = max(
                0.0,
                0.35 * max(0.0, flexible_delta)
                + 0.75 * max(0.0, misplace_delta)
                + 0.55 * max(0.0, internal_extra_delta)
                + 0.55 * max(0.0, unknown_var_delta)
                + 0.90 * max(0.0, ctrl_delta),
            )
            if exact_delta < -1e-9:
                erase_signal *= 0.15
            if stage_from_t(t) == "late" and exact_delta < -1e-9:
                erase_signal = 0.0
            erase_bonus = float(erase_w * erase_signal)
            if erase_bonus > 0:
                mask = max(float(mask), 0.03)

        reward = (
            coeff.action_align * action_align
            + coeff.skeleton_delta * skel_delta
            + coeff.exact_delta * exact_delta
            + coeff.clean_action * clean
            + coeff.mask_action * mask
            + state_w * flexible_delta
            + numeric_w * numeric_delta
            + misplace_w * misplace_delta
            + erase_bonus
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
            "numeric_before": float(numeric_before),
            "numeric_after": float(numeric_after),
            "numeric_delta": float(numeric_delta),
            "misplace_before": float(misplace_before),
            "misplace_after": float(misplace_after),
            "misplace_delta": float(misplace_delta),
            "flexible_before": float(flexible_before),
            "flexible_after": float(flexible_after),
            "flexible_delta": float(flexible_delta),
            "control_before": float(ctrl_before),
            "control_after": float(ctrl_after),
            "control_delta": float(ctrl_delta),
            "internal_extra_before": float(internal_extra_before),
            "internal_extra_after": float(internal_extra_after),
            "internal_extra_delta": float(internal_extra_delta),
            "unknown_var_before": float(unknown_var_before),
            "unknown_var_after": float(unknown_var_after),
            "unknown_var_delta": float(unknown_var_delta),
            "state_weight": float(state_w),
            "numeric_weight": float(numeric_w),
            "misplace_weight": float(misplace_w),
            "erase_weight": float(erase_w),
            "erase_bonus": float(erase_bonus),
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
