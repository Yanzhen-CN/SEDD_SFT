from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Optional

ANSWER_RE = re.compile(r"(?is)(?:^|\n)\s*answer\s*:\s*(.*)$")
REASON_RE = re.compile(r"(?is)(?:^|\n)\s*reasoning\s*:\s*(.*?)(?=(?:\n\s*answer\s*:)|$)")
LETTER_RE = re.compile(r"^[A-Za-z]$")
INTEGER_RE = re.compile(r"^[+-]?\d+$")
DECIMAL_RE = re.compile(r"^[+-]?(?:\d+\.\d+|\.\d+)$")
NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")
INTERVAL_RE = re.compile(
    r"^\s*([\(\[])\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*,\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*([\)\]])\s*$"
)

NOISE_MARKERS = ["<|endoftext|>", "<pad>", "[MASK]", "□", "�"]


@dataclass
class AnswerSpec:
    raw: str
    canonical: str
    type: str
    components: Dict[str, str] = field(default_factory=dict)

    @property
    def structured(self) -> bool:
        return self.type in {"signed_decimal", "interval"}

    @property
    def single(self) -> bool:
        return self.type in {"single_letter", "single_integer"}


def strip_noise(text: str) -> str:
    s = str(text or "")
    for marker in NOISE_MARKERS:
        s = s.replace(marker, "")
    # Common GPT-2 artifact after decoding invalid / special ids.
    s = s.replace("Ġ", " ")
    return s


def extract_answer_section(text: str) -> str:
    raw = str(text or "")
    m = ANSWER_RE.search(raw)
    if m:
        raw = m.group(1)
    # Keep only the first non-empty answer line; QRA answer should be concise.
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    raw = lines[0] if lines else raw.strip()
    return normalize_answer(raw)


def extract_reasoning_section(text: str) -> str:
    raw = str(text or "")
    m = REASON_RE.search(raw)
    if m:
        return strip_noise(m.group(1)).strip()
    # If there is no explicit anchor, use everything before Answer: as weak reasoning.
    raw = re.split(r"(?is)\n\s*answer\s*:", raw)[0]
    raw = re.sub(r"(?is)^\s*reasoning\s*:\s*", "", raw).strip()
    return strip_noise(raw)


def normalize_answer(text: str) -> str:
    s = strip_noise(text).strip()
    s = re.sub(r"(?is)^\s*answer\s*:\s*", "", s).strip()
    # Remove common trailing sentence punctuation, but never remove interval brackets.
    s = s.strip().strip("` ")
    if len(s) > 1 and s[-1] in ".;":
        s = s[:-1].strip()
    # Remove LaTeX boxing if present.
    box = re.match(r"^\\boxed\{(.+)\}$", s)
    if box:
        s = box.group(1).strip()
    # Normalize spaces inside compact mathematical answers.
    s = re.sub(r"\s+", " ", s)
    return s


def _canonical_number(s: str) -> str:
    s = normalize_answer(s)
    if s.startswith("+"):
        s = s[1:]
    # Keep integer spelling for integer answers.
    if INTEGER_RE.match(s):
        try:
            return str(int(s))
        except Exception:
            return s
    try:
        val = float(s)
        if math.isfinite(val):
            # Compact float canonicalization while preserving decimal nature.
            out = ("%.12g" % val)
            return out
    except Exception:
        pass
    return s


def parse_answer_spec(text: str) -> AnswerSpec:
    raw = normalize_answer(text)
    compact = raw.replace(" ", "")

    m = INTERVAL_RE.match(compact)
    if m:
        lb, left, right, rb = m.groups()
        left_c = _canonical_number(left)
        right_c = _canonical_number(right)
        canonical = f"{lb}{left_c},{right_c}{rb}"
        return AnswerSpec(
            raw=raw,
            canonical=canonical,
            type="interval",
            components={
                "left_bracket": lb,
                "left_value": left_c,
                "comma": ",",
                "right_value": right_c,
                "right_bracket": rb,
            },
        )

    if LETTER_RE.match(compact):
        return AnswerSpec(raw=raw, canonical=compact.upper(), type="single_letter", components={"letter": compact.upper()})

    if INTEGER_RE.match(compact):
        can = _canonical_number(compact)
        sign = "-" if can.startswith("-") else ""
        digits = can[1:] if sign else can
        return AnswerSpec(raw=raw, canonical=can, type="single_integer", components={"sign": sign, "digits": digits})

    if DECIMAL_RE.match(compact):
        sign = "-" if compact.startswith("-") else ("+" if compact.startswith("+") else "")
        body = compact[1:] if sign else compact
        int_part, frac_part = body.split(".", 1)
        if int_part == "":
            int_part = "0"
        canonical = _canonical_number(compact)
        return AnswerSpec(
            raw=raw,
            canonical=canonical,
            type="signed_decimal",
            components={"sign": sign, "int_part": str(int(int_part)) if int_part.isdigit() else int_part, "dot": ".", "frac_part": frac_part},
        )

    return AnswerSpec(raw=raw, canonical=raw.strip().lower(), type="short_text", components={"text": raw.strip().lower()})


def same_answer_type(pred: AnswerSpec, gt: AnswerSpec) -> bool:
    if pred.type == gt.type:
        return True
    # Allow numeric equivalence between integer and decimal, but score format separately.
    if pred.type in {"single_integer", "signed_decimal"} and gt.type in {"single_integer", "signed_decimal"}:
        return True
    return False


def exact_or_numeric_match(pred: AnswerSpec, gt: AnswerSpec, tol: float = 1e-9) -> bool:
    if pred.canonical == gt.canonical:
        return True
    if pred.type in {"single_integer", "signed_decimal"} and gt.type in {"single_integer", "signed_decimal"}:
        try:
            return abs(float(pred.canonical) - float(gt.canonical)) <= tol
        except Exception:
            return False
    return False


def longest_common_prefix(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def _safe_float(text: str) -> Optional[float]:
    try:
        val = float(text)
        if math.isfinite(val):
            return val
    except Exception:
        return None
    return None


def answer_potential(answer_text: str, gt: AnswerSpec) -> float:
    """Partial-state potential Phi(x) in [0, 1].

    This is intentionally tolerant to incomplete/noisy decoded states. It is used
    only for local reward shaping, not for final exact-match reporting.
    """
    s_raw = normalize_answer(answer_text)
    s = s_raw.replace(" ", "")
    if not s:
        return 0.0

    pred = parse_answer_spec(s_raw)
    if exact_or_numeric_match(pred, gt):
        return 1.0

    if gt.type == "interval":
        c = gt.components
        score = 0.0
        # Component-level potential. A wrong component can lower the potential
        # because it blocks exact recovery at that position.
        if s.startswith(c["left_bracket"]):
            score += 0.15
        elif s and s[0] in "([":
            score -= 0.08
        if s.endswith(c["right_bracket"]):
            score += 0.15
        elif s and s[-1] in ")]":
            score -= 0.08
        if "," in s:
            score += 0.10
        # Value matching before/after comma when possible; otherwise use contains
        # as a weak signal for very noisy intermediate strings.
        if "," in s:
            left_side, right_side = s.split(",", 1)
            if c["left_value"] in left_side:
                score += 0.20
            if c["right_value"] in right_side:
                score += 0.20
        else:
            if c["left_value"] in s:
                score += 0.12
            if c["right_value"] in s:
                score += 0.12
        if INTERVAL_RE.match(s):
            score += 0.20 if same_answer_type(pred, gt) else -0.15
        return float(max(-0.25, min(1.0, score)))

    if gt.type == "signed_decimal":
        c = gt.components
        score = 0.0
        # Type/format potential.
        if DECIMAL_RE.match(s):
            score += 0.15
        elif NUMBER_RE.match(s):
            score += 0.05
        elif LETTER_RE.match(s):
            score -= 0.25

        expected_sign = c.get("sign", "")
        if expected_sign:
            if s.startswith(expected_sign):
                score += 0.15
            elif s and s[0] in "+-":
                score -= 0.08
        else:
            if not s.startswith(("+", "-")):
                score += 0.10
            else:
                score -= 0.05

        body = s[1:] if s.startswith(("+", "-")) else s
        int_part = body.split(".", 1)[0] if body else ""
        gt_int = c.get("int_part", "")
        if int_part == gt_int:
            score += 0.20
        elif gt_int.startswith(int_part) and int_part:
            score += 0.10

        if "." in body:
            score += 0.10
            frac = body.split(".", 1)[1]
            gt_frac = c.get("frac_part", "")
            if gt_frac:
                lcp = longest_common_prefix(frac, gt_frac)
                score += 0.25 * (lcp / max(1, len(gt_frac)))
        val_pred = _safe_float(s)
        val_gt = _safe_float(gt.canonical)
        if val_pred is not None and val_gt is not None:
            scale = max(1.0, abs(val_gt))
            closeness = max(0.0, 1.0 - abs(val_pred - val_gt) / scale)
            score += 0.15 * closeness
        return float(max(-0.35, min(1.0, score)))

    # For single answers, local potential is deliberately coarse.
    if gt.type == "single_letter":
        if s.upper() == gt.canonical:
            return 1.0
        return 0.2 if LETTER_RE.match(s) else -0.5
    if gt.type == "single_integer":
        pred = parse_answer_spec(s_raw)
        if exact_or_numeric_match(pred, gt):
            return 1.0
        if same_answer_type(pred, gt):
            return 0.2
        return -0.5
    return 1.0 if pred.canonical == gt.canonical else 0.0


def final_answer_score(pred: AnswerSpec, gt: AnswerSpec) -> float:
    """Terminal answer score in [-1, 1].

    Design principle:
    - exact / numerically equivalent: strong positive
    - same type but wrong content: not catastrophic; it means the format is learned
    - wrong type: very bad
    """
    if exact_or_numeric_match(pred, gt):
        return 1.0
    if not same_answer_type(pred, gt):
        return -1.0
    if gt.type in {"single_letter", "single_integer"}:
        return 0.2
    if gt.structured:
        return max(0.15, answer_potential(pred.raw, gt) * 0.8)
    return 0.0
