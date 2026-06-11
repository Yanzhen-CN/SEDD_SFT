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
UNIT_DECIMAL_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*([a-zA-Z]+(?:/[a-zA-Z]+(?:\^?\d+)?)?|[a-zA-Z]+\^?\d+)\s*$"
)
EQUATION_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*\s*=\s*.+$")
SYMBOLIC_CHARS_RE = re.compile(r"^[A-Za-z0-9_+\-*/^=().,\[\]{}\\\s]+$")

NOISE_MARKERS = ["<|endoftext|>", "<pad>", "[MASK]", "□", "�"]


@dataclass
class AnswerSpec:
    raw: str
    canonical: str
    type: str
    components: Dict[str, str] = field(default_factory=dict)

    @property
    def structured(self) -> bool:
        return self.type in {"signed_decimal", "interval", "unit_decimal", "equation", "symbolic_expression"}

    @property
    def single(self) -> bool:
        return self.type in {"single_letter", "single_integer"}


def strip_noise(text: str) -> str:
    s = str(text or "")
    for marker in NOISE_MARKERS:
        s = s.replace(marker, "")
    s = s.replace("Ġ", " ")
    return s


def _strip_wrapping_math(s: str) -> str:
    s = s.strip()
    for left, right in [("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")]:
        if s.startswith(left) and s.endswith(right) and len(s) >= len(left) + len(right):
            s = s[len(left):-len(right)].strip()
    return s


def normalize_answer(text: str) -> str:
    s = strip_noise(text).strip()
    s = re.sub(r"(?is)^\s*answer\s*:\s*", "", s).strip()
    s = re.sub(
        r"(?is)^\s*(?:final\s+answer|final\s+solution|the\s+answer|the\s+solution|our\s+answer|answer|solution)\s*(?:is|=|:)?\s*",
        "",
        s,
    ).strip()
    s = s.strip().strip("` ")
    s = _strip_wrapping_math(s)
    box = re.match(r"^\\boxed\{(.+)\}$", s, flags=re.S)
    if box:
        s = box.group(1).strip()
    if len(s) > 1 and s[-1] in ".;":
        # Keep decimal point for numbers like "1." only if this is the whole answer.
        if not re.fullmatch(r"[+-]?\d+\.", s):
            s = s[:-1].strip()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_answer_section(text: str) -> str:
    raw = str(text or "")
    m = ANSWER_RE.search(raw)
    if m:
        raw = m.group(1)
    phrase = re.search(
        r"(?is)(?:final\s+answer|final\s+solution|the\s+answer|the\s+solution|our\s+answer)\s*(?:is|=|:)?\s*(.+)$",
        raw,
    )
    if phrase:
        raw = phrase.group(1).strip()
    lines = []
    for ln in raw.strip().splitlines():
        line = ln.strip()
        if not line:
            continue
        if line in {"$$", "$", r"\[", r"\]"}:
            continue
        if re.fullmatch(r"\\(?:begin|end)\{[^}]+\}", line):
            continue
        lines.append(line)
    raw = lines[0] if lines else raw.strip()
    return normalize_answer(raw)


def extract_reasoning_section(text: str) -> str:
    raw = str(text or "")
    m = REASON_RE.search(raw)
    if m:
        return strip_noise(m.group(1)).strip()
    raw = re.split(r"(?is)\n\s*answer\s*:", raw)[0]
    raw = re.sub(r"(?is)^\s*reasoning\s*:\s*", "", raw).strip()
    return strip_noise(raw)


def compact_math(s: str) -> str:
    s = normalize_answer(s)
    s = s.replace(" ", "")
    s = s.replace("×", "*").replace("−", "-")
    s = s.replace("\\cdot", "*")
    s = s.replace("\\times", "*")
    s = s.replace("\\sqrt", "sqrt")
    s = s.replace("\\pi", "pi")
    return s


def _canonical_number(s: str) -> str:
    s = normalize_answer(s).replace(" ", "")
    if s.startswith("+"):
        s = s[1:]
    if INTEGER_RE.match(s):
        try:
            return str(int(s))
        except Exception:
            return s
    try:
        val = float(s)
        if math.isfinite(val):
            return "%.12g" % val
    except Exception:
        pass
    return s


def _unit_norm(unit: str) -> str:
    u = str(unit or "").strip().replace(" ", "")
    u = u.replace("²", "^2").replace("³", "^3")
    return u


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
            components={"left_bracket": lb, "left_value": left_c, "comma": ",", "right_value": right_c, "right_bracket": rb},
        )

    m = UNIT_DECIMAL_RE.match(raw)
    if m:
        value, unit = m.groups()
        value_c = _canonical_number(value)
        unit_c = _unit_norm(unit)
        return AnswerSpec(
            raw=raw,
            canonical=f"{value_c} {unit_c}",
            type="unit_decimal",
            components={"value": value_c, "unit": unit_c},
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

    if EQUATION_RE.match(raw) and len(raw) <= 80:
        lhs, rhs = raw.split("=", 1)
        lhs_c = compact_math(lhs)
        rhs_c = compact_math(rhs)
        return AnswerSpec(
            raw=raw,
            canonical=f"{lhs_c}={rhs_c}",
            type="equation",
            components={"lhs": lhs_c, "equals": "=", "rhs": rhs_c},
        )

    if 1 <= len(raw) <= 80 and SYMBOLIC_CHARS_RE.match(raw) and re.search(r"[A-Za-z]", raw) and re.search(r"[+\-*/^()\\]", raw):
        return AnswerSpec(raw=raw, canonical=compact_math(raw), type="symbolic_expression", components={"expr": compact_math(raw)})

    return AnswerSpec(raw=raw, canonical=raw.strip().lower(), type="short_text", components={"text": raw.strip().lower()})


def same_answer_type(pred: AnswerSpec, gt: AnswerSpec) -> bool:
    if pred.type == gt.type:
        return True
    if pred.type in {"single_integer", "signed_decimal"} and gt.type in {"single_integer", "signed_decimal"}:
        return True
    # If the target is a numeric value with a unit, a plain number has partial value but wrong unit.
    if gt.type == "unit_decimal" and pred.type in {"single_integer", "signed_decimal"}:
        return True
    return False


def _safe_float(text: str) -> Optional[float]:
    try:
        val = float(str(text).replace("+", ""))
        if math.isfinite(val):
            return val
    except Exception:
        return None
    return None


def exact_or_numeric_match(pred: AnswerSpec, gt: AnswerSpec, tol: float = 1e-9) -> bool:
    if pred.canonical == gt.canonical:
        return True
    if pred.type in {"single_integer", "signed_decimal"} and gt.type in {"single_integer", "signed_decimal"}:
        vp, vg = _safe_float(pred.canonical), _safe_float(gt.canonical)
        return vp is not None and vg is not None and abs(vp - vg) <= tol
    if pred.type == "unit_decimal" and gt.type == "unit_decimal":
        if pred.components.get("unit") != gt.components.get("unit"):
            return False
        vp, vg = _safe_float(pred.components.get("value", "")), _safe_float(gt.components.get("value", ""))
        return vp is not None and vg is not None and abs(vp - vg) <= tol
    return False


def longest_common_prefix(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def _char_overlap_score(a: str, b: str) -> float:
    a = compact_math(a)
    b = compact_math(b)
    if not a or not b:
        return 0.0
    lcp = longest_common_prefix(a, b) / max(1, len(b))
    common = sum(1 for ch in set(a) if ch in set(b)) / max(1, len(set(b)))
    return max(lcp, 0.5 * common)


def answer_potential(answer_text: str, gt: AnswerSpec) -> float:
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
        vp, vg = _safe_float(s), _safe_float(gt.canonical)
        if vp is not None and vg is not None:
            scale = max(1.0, abs(vg))
            score += 0.15 * max(0.0, 1.0 - abs(vp - vg) / scale)
        return float(max(-0.35, min(1.0, score)))

    if gt.type == "unit_decimal":
        score = 0.0
        if pred.type == "unit_decimal":
            score += 0.20
            if pred.components.get("unit") == gt.components.get("unit"):
                score += 0.35
            elif pred.components.get("unit"):
                score -= 0.10
            vp = _safe_float(pred.components.get("value", ""))
        else:
            vp = _safe_float(s)
        vg = _safe_float(gt.components.get("value", ""))
        if vp is not None and vg is not None:
            scale = max(1.0, abs(vg))
            score += 0.35 * max(0.0, 1.0 - abs(vp - vg) / scale)
        if gt.components.get("unit", "") in compact_math(s_raw):
            score += 0.10
        return float(max(-0.35, min(1.0, score)))

    if gt.type == "equation":
        if pred.type == "equation":
            score = 0.20
            if pred.components.get("lhs") == gt.components.get("lhs"):
                score += 0.25
            if "=" in s_raw:
                score += 0.15
            score += 0.40 * _char_overlap_score(pred.components.get("rhs", ""), gt.components.get("rhs", ""))
            return float(max(-0.2, min(1.0, score)))
        return float(0.6 * _char_overlap_score(s_raw, gt.canonical) - 0.2)

    if gt.type == "symbolic_expression":
        return float(max(-0.2, min(1.0, _char_overlap_score(s_raw, gt.canonical))))

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
    if exact_or_numeric_match(pred, gt):
        return 1.0
    if not same_answer_type(pred, gt):
        if gt.type in {"equation", "symbolic_expression"}:
            return max(-0.5, answer_potential(pred.raw, gt) * 0.8)
        return -1.0
    if gt.type in {"single_letter", "single_integer"}:
        return 0.2
    if gt.structured:
        return max(0.10, answer_potential(pred.raw, gt) * 0.8)
    return 0.0
