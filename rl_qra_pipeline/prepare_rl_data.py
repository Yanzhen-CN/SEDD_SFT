from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SHORT_MAX_CHARS = 128
SHORT_MAX_TOKENS_APPROX = 32

QUESTION_KEYS = ["question", "problem", "prompt", "input", "query"]
REASONING_KEYS = [
    "deepseek_thinking_trajectory",
    "gemini_thinking_trajectory",
    "reasoning",
    "rationale",
    "thinking",
    "cot",
    "chain_of_thought",
]
SOLUTION_KEYS = ["solution", "answer", "final_answer", "target", "output"]

BAD_LONG_MARKERS = ["\\sum", "\\int", "\\prod", "\\lim", "\\begin", "\\end", "aligned", "equation", "cases"]

FINAL_TEMPLATES = [
    " Therefore, the final answer is {answer}.",
    " Thus, the final solution is {answer}.",
    " Finally, the answer is {answer}.",
    " Hence, the answer is {answer}.",
]

FINAL_MARKER_RE = re.compile(
    r"(?is)(?:"
    r"final\s+answer\s*(?:is|=|:)?|"
    r"final\s+solution\s*(?:is|=|:)?|"
    r"the\s+answer\s*(?:is|=|:)?|"
    r"the\s+solution\s*(?:is|=|:)?|"
    r"answer\s*(?:is|=|:)|"
    r"solution\s*(?:is|=|:)|"
    r"therefore[,\s]+(?:the\s+)?(?:final\s+)?(?:answer|solution)\s*(?:is|=|:)?|"
    r"thus[,\s]+(?:the\s+)?(?:final\s+)?(?:answer|solution)\s*(?:is|=|:)?|"
    r"hence[,\s]+(?:the\s+)?(?:final\s+)?(?:answer|solution)\s*(?:is|=|:)?|"
    r"finally[,\s]+(?:the\s+)?(?:answer|solution)\s*(?:is|=|:)?|"
    r"our\s+answer\s*(?:is|=|:)?"
    r")"
)

UNIT_RE = r"(?:m|mm|cm|km|kg|g|s|ms|N|J|W|V|A|Hz|m/s|m/s\^2|m\\/s|m\\/s\^2|\\mathrm\{\s*(?:m|mm|cm|kg|g|s|N|J|W|V|A|Hz)\s*\})"
NUM_RE = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"

# Priority matters: if a window contains r=...=1.903 m, we prefer the final unit decimal over the entire long equation.
CANDIDATE_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    (
        "interval",
        re.compile(rf"[\(\[]\s*{NUM_RE}\s*,\s*{NUM_RE}\s*[\)\]]"),
    ),
    (
        "inequality",
        re.compile(rf"[A-Za-z][A-Za-z0-9_]*\s*(?:<=|>=|<|>)\s*{NUM_RE}"),
    ),
    (
        "unit_decimal",
        re.compile(rf"{NUM_RE}\s*{UNIT_RE}"),
    ),
    (
        "equation",
        re.compile(r"[A-Za-z][A-Za-z0-9_]*\s*=\s*[A-Za-z0-9_+\-*/^().\\]+"),
    ),
    (
        "symbolic_expression",
        re.compile(r"(?:\\?sqrt\s*\(?\s*\d+\s*\)?|\d+\s*sqrt\s*\(?\s*\d+\s*\)?|[A-Za-z0-9_]+\s*\^\s*\d+(?:\s*[+\-*/]\s*[A-Za-z0-9_]+\s*\^?\s*\d*)*|\d+\s*/\s*\d+|pi\s*\*\s*r\s*\^\s*2|v\s*\^\s*2\s*=\s*u\s*\^\s*2\s*\+\s*2as)"),
    ),
    (
        "signed_decimal_or_number",
        re.compile(NUM_RE),
    ),
    (
        "option",
        re.compile(r"(?<![A-Za-z])(?:[A-E]|[a-e])(?![A-Za-z])"),
    ),
]


def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        for line in text.splitlines():
            if line.strip():
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
        return rows
    obj = json.loads(text)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("data", "train", "valid", "validation", "test", "rows", "examples"):
            value = obj.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    return []


def load_split(input_dir: Path, split: str) -> List[Dict[str, Any]]:
    names = [split]
    if split == "val":
        names += ["valid", "validation"]
    candidates: List[Path] = []
    for name in names:
        candidates += [
            input_dir / f"{name}.jsonl",
            input_dir / f"{name}.json",
            input_dir / name / "data.jsonl",
            input_dir / name / "data.json",
        ]
    for path in candidates:
        rows = read_json_or_jsonl(path)
        if rows:
            return rows
    return []


def pick_first(row: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_answer_text(text: str) -> str:
    text = str(text or "").strip()
    text = text.replace("\\,", " ")
    text = re.sub(r"\\mathrm\{\s*([^{}]+?)\s*\}", r"\1", text)
    text = re.sub(r"\\text\{\s*([^{}]+?)\s*\}", r"\1", text)
    text = text.replace("~", " ")
    text = normalize_spaces(text)
    text = text.strip("` ")
    # Strip surrounding math delimiters only when they wrap the whole answer.
    for left, right in [("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")]:
        if text.startswith(left) and text.endswith(right) and len(text) > len(left) + len(right):
            text = text[len(left): -len(right)].strip()
    boxed = re.fullmatch(r"\\boxed\{(.+)\}", text, flags=re.S)
    if boxed:
        text = boxed.group(1).strip()
    if len(text) > 1 and text[-1] in ".;，。":
        text = text[:-1].strip()
    return normalize_spaces(text)


def strip_short_answer_wrappers(raw: str) -> str:
    text = str(raw or "").strip()
    marker = re.search(r"(?is)####\s*(.+)$", text)
    if marker:
        text = marker.group(1).strip()
    text = re.sub(
        r"(?is)^\s*(?:final\s+answer|final\s+solution|the\s+answer|the\s+solution|answer|solution|our\s+answer)\s*(?:is|=|:)?\s*",
        "",
        text,
    ).strip()
    return normalize_answer_text(text)


def has_long_derivation_shape(raw: str) -> bool:
    raw = str(raw or "").strip()
    if not raw:
        return False
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) > 2:
        return True
    if "$$" in raw and "\n" in raw:
        return True
    if len(raw) > SHORT_MAX_CHARS * 2 and any(marker in raw for marker in BAD_LONG_MARKERS):
        return True
    if len(raw) > 120 and raw.count("=") >= 2 and any(x in raw for x in ["\\frac", "\\sqrt", "\\sin", "\\cos", "\\left", "\\right"]):
        return True
    return False


def answer_kind(answer: str) -> str:
    compact = answer.replace(" ", "")
    if re.fullmatch(r"^[A-Za-z]$", compact):
        return "option"
    if re.fullmatch(rf"^{NUM_RE}$", compact):
        return "number"
    if re.fullmatch(rf"^[\(\[]\s*{NUM_RE}\s*,\s*{NUM_RE}\s*[\)\]]$", answer):
        return "interval"
    if re.fullmatch(rf"^{NUM_RE}\s*{UNIT_RE}$", answer):
        return "unit_decimal"
    if re.fullmatch(rf"^[A-Za-z][A-Za-z0-9_]*\s*(?:<=|>=|<|>)\s*{NUM_RE}$", compact):
        return "inequality"
    if re.fullmatch(r"^[A-Za-z][A-Za-z0-9_]*\s*=\s*[A-Za-z0-9_+\-*/^().\\]+$", compact):
        return "equation"
    if re.fullmatch(r"^[A-Za-z0-9_+\-*/^=().,\[\]{}\\]+$", compact):
        return "symbolic_expression"
    return "short_text"


def is_supported_short_answer(answer: str) -> Tuple[bool, str]:
    answer = normalize_answer_text(answer)
    if not answer:
        return False, "empty_after_clean"
    if answer in {"$", "$$", r"\[", r"\]", r"\(", r"\)"}:
        return False, "math_delimiter_only"
    if "\n" in answer:
        return False, "newline_after_clean"
    if len(answer) > SHORT_MAX_CHARS:
        return False, "too_long"
    if len(answer.split()) > SHORT_MAX_TOKENS_APPROX:
        return False, "too_many_words"
    if any(marker in answer for marker in BAD_LONG_MARKERS):
        return False, "complex_latex"

    kind = answer_kind(answer)
    if kind in {"option", "number", "interval", "inequality", "unit_decimal", "equation", "symbolic_expression"}:
        return True, kind
    return False, "unsupported_short_form"


def looks_like_short_answer(raw: str) -> Tuple[bool, str, str]:
    raw = str(raw or "").strip()
    if not raw:
        return False, "", "empty_solution"
    if has_long_derivation_shape(raw):
        return False, "", "long_derivation_or_display_math"
    cleaned = strip_short_answer_wrappers(raw)
    ok, reason_or_kind = is_supported_short_answer(cleaned)
    if ok:
        return True, cleaned, "direct_" + reason_or_kind
    return False, "", reason_or_kind


def extract_candidates_from_window(window: str) -> List[Tuple[str, str]]:
    window = str(window or "")
    # Stop at the next likely sentence if the window is prose. Do not stop at ')' or ']' because intervals need them.
    window = window[:260]
    candidates: List[Tuple[str, str]] = []
    for kind, pattern in CANDIDATE_PATTERNS:
        for m in pattern.finditer(window):
            raw = m.group(0)
            candidate = normalize_answer_text(raw)
            ok, reason_or_kind = is_supported_short_answer(candidate)
            if ok:
                candidates.append((candidate, reason_or_kind))
    # Deduplicate while preserving order.
    seen = set()
    deduped: List[Tuple[str, str]] = []
    for candidate, kind in candidates:
        key = candidate.lower().replace(" ", "")
        if key not in seen:
            seen.add(key)
            deduped.append((candidate, kind))
    return deduped


def extract_final_answer(text: str) -> Tuple[bool, str, str]:
    text = str(text or "")
    if not text.strip():
        return False, "", "no_text"

    matches = list(FINAL_MARKER_RE.finditer(text))
    if matches:
        # Use the last final-answer style marker. This avoids grabbing intermediate numbers.
        for match in reversed(matches):
            window = text[match.end(): match.end() + 320]
            candidates = extract_candidates_from_window(window)
            if candidates:
                # Prefer the last candidate in the marker window for chained equations like r=...=1.903 m.
                answer, kind = candidates[-1]
                return True, answer, "marker_" + kind
        return False, "", "marker_found_but_no_supported_answer"

    # Fallback: if the whole text ends with a supported short answer pattern, keep it.
    tail = text[-360:]
    candidates = extract_candidates_from_window(tail)
    if candidates:
        answer, kind = candidates[-1]
        return True, answer, "tail_" + kind
    return False, "", "no_final_marker"


def answer_in_reasoning(reasoning: str, answer: str) -> bool:
    reasoning_norm = re.sub(r"\s+", "", str(reasoning or "").lower())
    answer_norm = re.sub(r"\s+", "", str(answer or "").lower())
    if not answer_norm:
        return False
    if answer_norm in reasoning_norm:
        return True
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", answer_norm):
        return re.search(rf"(?<!\d){re.escape(answer_norm)}(?!\d)", reasoning_norm) is not None
    return False


def make_sample(row: Dict[str, Any], split: str, idx: int, answer: str, source_tag: str, synthetic: bool = False) -> Dict[str, Any]:
    question = pick_first(row, QUESTION_KEYS)
    reasoning = pick_first(row, REASONING_KEYS)
    if not question:
        question = str(row.get("text", "")).strip()
    if not reasoning:
        reasoning = "First, we analyze the problem step by step."
    if not answer_in_reasoning(reasoning, answer):
        template = FINAL_TEMPLATES[idx % len(FINAL_TEMPLATES)]
        reasoning = reasoning.rstrip() + template.format(answer=answer)
    sid = row.get("id") or row.get("uid") or f"{split}_{idx:06d}"
    if synthetic:
        sid = f"synthetic_{sid}"
    kind = answer_kind(answer)
    return {
        "id": f"s1f_rl_{split}_{sid}",
        "source": source_tag,
        "answer_kind": kind,
        "question": question,
        "reasoning": reasoning,
        "answer": answer,
        "solution": answer,
        "segments": {
            "question": {"text": f"Question:\n{question.strip()}\n\n", "train": False},
            "reasoning": {"text": f"Reasoning:\n{reasoning.strip()}\n", "train": False},
            "answer_anchor": {"text": "\nAnswer:\n", "train": False},
            "answer": {"text": str(answer).strip(), "train": True},
        },
    }



# Synthetic hard cases are generated parametrically.  Unlike the previous
# paraphrase version, we do NOT create several questions for the same answer.
# Each generated synthetic example has a unique final answer; only the wording
# style is randomized across examples of the same type.

SYNTHETIC_FINAL_TEMPLATES = [
    "Therefore, the final answer is {answer}.",
    "Thus, the final solution is {answer}.",
    "Finally, the answer is {answer}.",
    "Hence, the answer should be {answer}.",
    "So the solution is {answer}.",
    "We conclude that the answer is {answer}.",
    "This gives the final solution: {answer}.",
    "The required answer is {answer}.",
]

REASONING_OPENERS = [
    "To solve the problem, first translate the condition into the target answer form.",
    "First, identify the boundary values and whether each boundary is included.",
    "We start by isolating the requested quantity from the given relation.",
    "The key step is to keep the final response in a compact mathematical form.",
    "First, simplify the expression and then write only the final result.",
]


def fmt_num(value: float | int) -> str:
    if isinstance(value, int) or abs(float(value) - round(float(value))) < 1e-9:
        return str(int(round(float(value))))
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return text if text != "-0" else "0"


def make_interval_case(rng: random.Random, idx: int) -> Dict[str, str]:
    a = rng.randint(-12, 7)
    b = rng.randint(a + 1, a + 12)
    left_closed = rng.choice([True, False])
    right_closed = rng.choice([True, False])
    left = "[" if left_closed else "("
    right = "]" if right_closed else ")"
    answer = f"{left}{a},{b}{right}"

    lower_symbol = ">=" if left_closed else ">"
    upper_symbol = "<=" if right_closed else "<"
    lower_phrase = "included" if left_closed else "excluded"
    upper_phrase = "included" if right_closed else "excluded"

    question_templates = [
        f"Convert the condition {a} { '<=' if left_closed else '<' } x { '<=' if right_closed else '<' } {b} into interval notation.",
        f"Write the interval for all x such that x {lower_symbol} {a} and x {upper_symbol} {b}.",
        f"Give the interval with left endpoint {a} {lower_phrase} and right endpoint {b} {upper_phrase}.",
        f"A solution set runs from {a} to {b}; the left endpoint is {lower_phrase} and the right endpoint is {upper_phrase}. Write it as an interval.",
    ]
    reasoning_templates = [
        f"First, the left endpoint is {a} and it is {lower_phrase}. The right endpoint is {b} and it is {upper_phrase}.",
        f"First, translate the left inequality into {'a bracket' if left_closed else 'a parenthesis'} and the right inequality into {'a bracket' if right_closed else 'a parenthesis'}.",
        f"The lower bound is {a} and the upper bound is {b}. The endpoint symbols follow directly from whether equality is allowed.",
        f"To solve, keep the two bounds in increasing order and choose the endpoint marks from the inequality signs.",
    ]
    return {
        "id": f"param_interval_{idx:04d}",
        "kind": "interval",
        "question": rng.choice(question_templates),
        "reasoning": rng.choice(reasoning_templates) + " " + rng.choice(SYNTHETIC_FINAL_TEMPLATES).format(answer=answer),
        "solution": answer,
    }


def make_inequality_case(rng: random.Random, idx: int) -> Dict[str, str]:
    var = rng.choice(["x", "y", "t", "r"])
    c = rng.randint(-15, 15)
    op = rng.choice([">", ">=", "<", "<="])
    answer = f"{var}{op}{c}"
    phrase = {
        ">": "greater than",
        ">=": "greater than or equal to",
        "<": "less than",
        "<=": "less than or equal to",
    }[op]
    question_templates = [
        f"Write the final inequality if {var} is {phrase} {c}.",
        f"Convert the verbal condition '{var} is {phrase} {c}' into symbolic form.",
        f"Give the compact answer for the condition that {var} must be {phrase} {c}.",
        f"State the solution as a one-sided inequality for variable {var} with threshold {c}.",
    ]
    reasoning_templates = [
        f"First, keep the variable {var} on the left side and place the threshold {c} on the right side.",
        f"To solve, choose the inequality symbol that matches the phrase '{phrase}'.",
        f"The condition is one-sided, so the answer should be written as a single symbolic inequality.",
    ]
    return {
        "id": f"param_inequality_{idx:04d}",
        "kind": "inequality",
        "question": rng.choice(question_templates),
        "reasoning": rng.choice(reasoning_templates) + " " + rng.choice(SYNTHETIC_FINAL_TEMPLATES).format(answer=answer),
        "solution": answer,
    }


def make_linear_equation_case(rng: random.Random, idx: int) -> Dict[str, str]:
    var = rng.choice(["x", "y", "a", "t"])
    value = rng.randint(-12, 18)
    coef = rng.choice([2, 3, 4, 5, -2, -3])
    bias = rng.randint(-10, 10)
    rhs = coef * value + bias
    answer = f"{var}={value}"
    question_templates = [
        f"Solve {coef}{var}+{bias}={rhs}.",
        f"Find {var} from the linear equation {coef}{var}+{bias}={rhs}.",
        f"What is the value of {var} if {coef}{var}+{bias} equals {rhs}?",
        f"Isolate {var} in {coef}{var}+{bias}={rhs}.",
    ]
    reasoning_templates = [
        f"First, subtract {bias} from both sides to get {coef}{var}={coef * value}. Then divide by {coef}.",
        f"To solve, move the constant term to the right side and then divide by the coefficient of {var}.",
        f"First isolate the variable term. The resulting value for {var} is {value}.",
    ]
    return {
        "id": f"param_equation_linear_{idx:04d}",
        "kind": "equation",
        "question": rng.choice(question_templates),
        "reasoning": rng.choice(reasoning_templates) + " " + rng.choice(SYNTHETIC_FINAL_TEMPLATES).format(answer=answer),
        "solution": answer,
    }


def make_formula_equation_case(rng: random.Random, idx: int) -> Dict[str, str]:
    templates = [
        ("Solve F=ma for acceleration a.", "a=F/m", "First, divide both sides by m so that acceleration is isolated."),
        ("Write Newton's second law in compact form.", "F=ma", "First, force equals mass times acceleration in Newton's second law."),
        ("Write the circle area formula using A and r.", "A=pi*r^2", "First, circle area is pi times radius squared."),
        ("Write the slope-intercept line with slope {m} and intercept {b}.", None, None),
        ("Solve v=d/t for distance d.", "d=v*t", "First, multiply both sides by t to isolate distance."),
        ("Solve p=mv for velocity v.", "v=p/m", "First, divide momentum by mass to isolate velocity."),
    ]
    q, ans, reason = rng.choice(templates)
    if ans is None:
        m = rng.choice([-3, -2, -1, 1, 2, 3, 4, 5])
        b = rng.randint(-8, 8)
        sign = "+" if b >= 0 else ""
        ans = f"y={m}x{sign}{b}"
        q = q.format(m=m, b=b)
        reason = "First, use y=mx+b. Substitute the given slope and intercept into this form."
    return {
        "id": f"param_equation_formula_{idx:04d}",
        "kind": "equation",
        "question": q,
        "reasoning": reason + " " + rng.choice(SYNTHETIC_FINAL_TEMPLATES).format(answer=ans),
        "solution": ans,
    }


def make_symbolic_case(rng: random.Random, idx: int) -> Dict[str, str]:
    family = rng.choice(["sqrt", "square", "fraction", "monomial", "kinematics", "factor"])
    if family == "sqrt":
        outside = rng.randint(2, 9)
        inside = rng.choice([2, 3, 5, 6, 7, 10, 11, 13])
        n = outside * outside * inside
        ans = f"{outside}sqrt({inside})"
        question = rng.choice([
            f"Simplify sqrt({n}).",
            f"Write sqrt({n}) in simplified radical form.",
            f"Take the largest square factor out of sqrt({n}).",
        ])
        reasoning = f"First, factor {n} as {outside * outside} times {inside}. Then sqrt({outside * outside})={outside}."
    elif family == "square":
        k = rng.randint(1, 8)
        ans = f"x^2+{2*k}x+{k*k}"
        question = rng.choice([
            f"Expand (x+{k})^2.",
            f"Multiply out (x+{k})(x+{k}).",
            f"Use the square formula to expand (x+{k})^2.",
        ])
        reasoning = f"First, use (a+b)^2=a^2+2ab+b^2 with b={k}."
    elif family == "fraction":
        den = rng.choice([4, 5, 6, 7, 8, 9, 10, 12])
        num = rng.randint(1, den - 1)
        # Keep simple proper fraction answers.
        ans = f"{num}/{den}"
        mult = rng.randint(2, 6)
        question = rng.choice([
            f"Simplify {num*mult}/{den*mult}.",
            f"Reduce the fraction {num*mult}/{den*mult} to lowest terms.",
            f"Write {num*mult} divided by {den*mult} as a simplified fraction.",
        ])
        reasoning = f"First, divide numerator and denominator by the common factor {mult}."
    elif family == "monomial":
        coef = rng.randint(2, 9)
        power = rng.randint(2, 4)
        ans = f"{coef}x^{power}"
        question = rng.choice([
            f"Simplify {coef} times " + " times ".join(["x"] * power) + ".",
            f"Write the product {coef}*" + "*".join(["x"] * power) + " in exponent form.",
            f"Combine repeated x factors in {coef}*" + "*".join(["x"] * power) + ".",
        ])
        reasoning = f"First, combine the {power} repeated x factors into x^{power} and keep the coefficient {coef}."
    elif family == "kinematics":
        ans = rng.choice(["v^2=u^2+2as", "s=u*t+0.5*a*t^2", "v=u+a*t"])
        question = rng.choice([
            "Write a compact constant-acceleration kinematics formula.",
            "Give a symbolic expression used in one-dimensional motion.",
            "State the requested kinematics relation in short form.",
        ])
        reasoning = "First, choose the standard relation matching the requested variables and keep it in compact symbolic form."
    else:
        k = rng.randint(1, 8)
        ans = f"(x+{k})^2"
        question = rng.choice([
            f"Factor x^2+{2*k}x+{k*k}.",
            f"Write x^2+{2*k}x+{k*k} as a perfect square.",
            f"Factor the quadratic x^2+{2*k}x+{k*k}.",
        ])
        reasoning = f"First, recognize the perfect square pattern with b={k}."
    return {
        "id": f"param_symbolic_{idx:04d}",
        "kind": "symbolic_expression",
        "question": question,
        "reasoning": reasoning + " " + rng.choice(SYNTHETIC_FINAL_TEMPLATES).format(answer=ans),
        "solution": ans,
    }


def make_unit_decimal_case(rng: random.Random, idx: int) -> Dict[str, str]:
    unit = rng.choice(["m", "mm", "cm", "kg", "g", "s", "ms", "N", "J", "m/s", "m/s^2"])
    value = rng.choice([
        round(rng.uniform(-12, 12), 2),
        round(rng.uniform(0.05, 5), 3),
        rng.choice([-9.8, -2.50, 0.125, 1.903, 3.14]),
    ])
    ans = f"{fmt_num(value)} {unit}"
    question_templates = [
        f"Report the measured value {fmt_num(value)} using the unit {unit}.",
        f"Write the final numerical result as a signed decimal with unit {unit}: {fmt_num(value)}.",
        f"Give the compact answer for a quantity equal to {fmt_num(value)} {unit}.",
        f"A calculation gives {fmt_num(value)} in units of {unit}. Write the final answer with unit.",
    ]
    reasoning_templates = [
        f"First, keep the decimal value {fmt_num(value)} unchanged and attach the unit {unit}.",
        f"To solve, preserve the sign and decimal places, then write the unit symbol {unit} after the number.",
        f"The result is already in the requested unit, so only the compact value with unit is needed.",
    ]
    return {
        "id": f"param_unit_decimal_{idx:04d}",
        "kind": "unit_decimal",
        "question": rng.choice(question_templates),
        "reasoning": rng.choice(reasoning_templates) + " " + rng.choice(SYNTHETIC_FINAL_TEMPLATES).format(answer=ans),
        "solution": ans,
    }


def make_number_case(rng: random.Random, idx: int) -> Dict[str, str]:
    a = round(rng.uniform(-15, 15), 2)
    b = round(rng.uniform(-8, 8), 2)
    value = round(a + b, 2)
    ans = fmt_num(value)
    question_templates = [
        f"Compute {fmt_num(a)} + {fmt_num(b)}.",
        f"Evaluate the signed decimal expression {fmt_num(a)}+{fmt_num(b)}.",
        f"Add {fmt_num(a)} and {fmt_num(b)} and give only the result.",
    ]
    reasoning_templates = [
        f"First, combine the two signed decimals. The arithmetic result is {ans}.",
        f"To solve, add the values while preserving the correct sign.",
        f"First align the decimal places and perform the addition.",
    ]
    return {
        "id": f"param_number_{idx:04d}",
        "kind": "number",
        "question": rng.choice(question_templates),
        "reasoning": rng.choice(reasoning_templates) + " " + rng.choice(SYNTHETIC_FINAL_TEMPLATES).format(answer=ans),
        "solution": ans,
    }


SYNTHETIC_GENERATORS = [
    ("interval", make_interval_case),
    ("inequality", make_inequality_case),
    ("equation", make_linear_equation_case),
    ("equation", make_formula_equation_case),
    ("symbolic_expression", make_symbolic_case),
    ("unit_decimal", make_unit_decimal_case),
    ("number", make_number_case),
]


def make_extra_cases(seed: int = 42, synthetic_count: int = 240) -> List[Dict[str, str]]:
    """Generate parameterized synthetic cases with unique answers.

    synthetic_count controls the total number of generated synthetic examples.
    The generator first samples a problem type, then samples parameters such as
    interval endpoints, inequality threshold, equation coefficients, expression
    constants, or unit decimal values.  It then constructs the reasoning
    backwards so that every example ends with a final-answer style marker.
    """
    rng = random.Random(seed)
    generated: List[Dict[str, str]] = []
    seen_answers: set[str] = set()
    attempts = 0
    max_attempts = max(2000, synthetic_count * 40)

    # Slightly emphasize the hard forms that motivated S1F_RL.
    weights = {
        "interval": 0.24,
        "inequality": 0.12,
        "equation": 0.20,
        "symbolic_expression": 0.20,
        "unit_decimal": 0.18,
        "number": 0.06,
    }
    names = [name for name, _ in SYNTHETIC_GENERATORS]
    probs = [weights.get(name, 0.1) for name in names]

    while len(generated) < synthetic_count and attempts < max_attempts:
        attempts += 1
        gen_index = rng.choices(range(len(SYNTHETIC_GENERATORS)), weights=probs, k=1)[0]
        kind, gen = SYNTHETIC_GENERATORS[gen_index]
        row = gen(rng, len(generated))
        answer = normalize_answer_text(row.get("solution", ""))
        ok, reason_or_kind = is_supported_short_answer(answer)
        if not ok:
            continue
        key = answer.lower().replace(" ", "")
        if key in seen_answers:
            continue
        seen_answers.add(key)
        row["solution"] = answer
        row["kind"] = row.get("kind", reason_or_kind)
        generated.append(row)

    rng.shuffle(generated)
    return generated



def allocate_synthetic_quotas(
    real_counts: Dict[str, int],
    synthetic_ratio: float,
    min_synthetic_per_split: int,
    max_synthetic_fraction: float,
) -> Dict[str, int]:
    """Allocate synthetic examples after seeing real extracted counts.

    synthetic_ratio is relative to the real samples in the same split. For example,
    0.30 means roughly 30 synthetic samples per 100 real samples. The actual
    fraction in the final dataset is ratio / (1 + ratio).

    min_synthetic_per_split prevents tiny val/test splits such as 4 examples.
    max_synthetic_fraction avoids synthetic examples dominating a split when the
    real count is very small.
    """
    quotas: Dict[str, int] = {}
    for split in ["train", "val", "test"]:
        real_n = int(real_counts.get(split, 0))
        base = int(round(real_n * synthetic_ratio))
        quota = max(min_synthetic_per_split, base)

        # If there are no real examples at all in a split, still add a small
        # diagnostic set, but do not create a huge synthetic-only split.
        if real_n <= 0:
            quota = min_synthetic_per_split

        if max_synthetic_fraction > 0 and real_n > 0:
            # quota / (real_n + quota) <= max_synthetic_fraction
            max_quota = int(max_synthetic_fraction * real_n / max(1e-8, 1.0 - max_synthetic_fraction))
            max_quota = max(min_synthetic_per_split, max_quota)
            quota = min(quota, max_quota)

        quotas[split] = max(0, quota)
    return quotas


def split_extra_cases_by_quota(seed: int, quotas: Dict[str, int]) -> Dict[str, List[Dict[str, Any]]]:
    total_needed = sum(max(0, int(v)) for v in quotas.values())
    cases = make_extra_cases(seed=seed, synthetic_count=total_needed)

    out: Dict[str, List[Dict[str, Any]]] = {}
    cursor = 0
    for split in ["train", "val", "test"]:
        n = max(0, int(quotas.get(split, 0)))
        out[split] = cases[cursor: cursor + n]
        cursor += n
    return out


def build_real_split(rows: List[Dict[str, Any]], split: str, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Build only real S1K-derived examples, before adding synthetic data."""
    out: List[Dict[str, Any]] = []
    stats: Dict[str, int] = {
        "total": len(rows),
        "kept_direct_solution": 0,
        "kept_extracted_final": 0,
        "synthetic_added": 0,
    }

    for idx, row in enumerate(rows):
        solution = pick_first(row, SOLUTION_KEYS)
        ok, answer, reason = looks_like_short_answer(solution)
        if ok:
            out.append(make_sample(row, split, idx, answer, source_tag="s1f_rl_direct_solution", synthetic=False))
            stats["kept_direct_solution"] += 1
            stats[reason] = stats.get(reason, 0) + 1
            continue

        reasoning = pick_first(row, REASONING_KEYS)
        search_text = "\n".join([reasoning, solution])
        ok2, extracted, extract_reason = extract_final_answer(search_text)
        if ok2:
            out.append(make_sample(row, split, idx, extracted, source_tag="s1f_rl_extracted_final", synthetic=False))
            stats["kept_extracted_final"] += 1
            stats[extract_reason] = stats.get(extract_reason, 0) + 1
            continue

        stats[f"drop_{reason}"] = stats.get(f"drop_{reason}", 0) + 1
        stats[f"extract_fail_{extract_reason}"] = stats.get(f"extract_fail_{extract_reason}", 0) + 1

    stats["kept_real_total"] = stats["kept_direct_solution"] + stats["kept_extracted_final"]
    stats["kept_total"] = stats["kept_real_total"]
    return out, stats


def add_synthetic_to_split(
    out: List[Dict[str, Any]],
    stats: Dict[str, int],
    split: str,
    seed: int,
    add_synthetic: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    existing_answers = {
        normalize_answer_text(str(item.get("answer", ""))).lower().replace(" ", "")
        for item in out
    }
    start = len(out)
    for j, row in enumerate(add_synthetic):
        ok, answer, reason = looks_like_short_answer(row.get("solution", ""))
        if not ok:
            stats[f"synthetic_drop_{reason}"] = stats.get(f"synthetic_drop_{reason}", 0) + 1
            continue
        answer_key = normalize_answer_text(answer).lower().replace(" ", "")
        if answer_key in existing_answers:
            stats["synthetic_drop_duplicate_answer"] = stats.get("synthetic_drop_duplicate_answer", 0) + 1
            continue
        existing_answers.add(answer_key)
        out.append(make_sample(row, split, start + j, answer, source_tag="s1f_rl_synthetic_parametric", synthetic=True))
        stats["synthetic_added"] += 1

    rng = random.Random(seed + {"train": 11, "val": 17, "test": 23}.get(split, 0))
    rng.shuffle(out)
    stats["kept_total"] = stats.get("kept_real_total", 0) + stats["synthetic_added"]
    return out, stats


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default=str(repo_dir / "data" / "S1K"), help="Original S1K split directory, normally repo_root/data/S1K.")
    parser.add_argument("--output-dir", type=str, default=str(script_dir / "data" / "S1F_RL"), help="Output directory. Default: rl_qra_pipeline/data/S1F_RL.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--synthetic-ratio",
        type=float,
        default=0.30,
        help="Synthetic/real ratio per split after real S1K extraction. 0.30 means synthetic is about 23 percent of the final split.",
    )
    parser.add_argument(
        "--min-synthetic-per-split",
        type=int,
        default=8,
        help="Minimum synthetic examples added to each split, useful when val/test real counts are tiny.",
    )
    parser.add_argument(
        "--max-synthetic-fraction",
        type=float,
        default=0.40,
        help="Maximum synthetic fraction in each final split. Use 0 to disable. Default 0.40 prevents synthetic domination.",
    )
    parser.add_argument(
        "--synthetic-count",
        type=int,
        default=None,
        help="Optional override for total synthetic count. If set, allocation is 8:1:1 with per-split minimum still applied.",
    )
    parser.add_argument(
        "--synthetic-variants-per-case",
        type=int,
        default=None,
        help="Deprecated alias. If provided, it is treated as --synthetic-count for compatibility.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # First pass: extract only real S1K-derived examples. This lets us decide
    # how many synthetic hard cases to add based on the actual real data size.
    real_built: Dict[str, List[Dict[str, Any]]] = {}
    split_stats: Dict[str, Dict[str, int]] = {}
    real_counts: Dict[str, int] = {}

    for split in ["train", "val", "test"]:
        rows = load_split(input_dir, split)
        built, stats = build_real_split(rows=rows, split=split, seed=args.seed)
        real_built[split] = built
        split_stats[split] = stats
        real_counts[split] = len(built)

    synthetic_count_override = args.synthetic_count
    if synthetic_count_override is None and args.synthetic_variants_per_case is not None:
        synthetic_count_override = args.synthetic_variants_per_case

    if synthetic_count_override is not None:
        total = max(0, int(synthetic_count_override))
        quotas = {
            "train": int(round(total * 0.8)),
            "val": int(round(total * 0.1)),
        }
        quotas["test"] = max(0, total - quotas["train"] - quotas["val"])
        for split in ["train", "val", "test"]:
            quotas[split] = max(args.min_synthetic_per_split, quotas[split]) if total > 0 else 0
    else:
        quotas = allocate_synthetic_quotas(
            real_counts=real_counts,
            synthetic_ratio=args.synthetic_ratio,
            min_synthetic_per_split=args.min_synthetic_per_split,
            max_synthetic_fraction=args.max_synthetic_fraction,
        )

    extra_by_split = split_extra_cases_by_quota(args.seed, quotas)

    summary: Dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "version": "s1f_rl_v5_final_marker_extraction_plus_ratio_controlled_parametric_synthetic",
        "real_counts_before_synthetic": real_counts,
        "synthetic_ratio": args.synthetic_ratio,
        "min_synthetic_per_split": args.min_synthetic_per_split,
        "max_synthetic_fraction": args.max_synthetic_fraction,
        "synthetic_count_override": synthetic_count_override,
        "synthetic_quotas": quotas,
        "synthetic_total_assigned": sum(len(v) for v in extra_by_split.values()),
        "synthetic_policy": "extract real S1K samples first, then add parametric hard cases per split according to real-count ratio; no repeated synthetic answer",
        "synthetic_types": {
            "interval": "parametric",
            "inequality": "parametric",
            "equation": "parametric",
            "symbolic_expression": "parametric",
            "unit_decimal_or_signed_decimal": "parametric",
        },
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        built, stats = add_synthetic_to_split(
            out=real_built[split],
            stats=split_stats[split],
            split=split,
            seed=args.seed,
            add_synthetic=extra_by_split.get(split, []),
        )
        write_jsonl(output_dir / f"{split}.jsonl", built)
        synthetic_fraction = (stats["synthetic_added"] / max(1, len(built))) if built else 0.0
        summary["splits"][split] = {
            **stats,
            "written": len(built),
            "synthetic_quota": quotas.get(split, 0),
            "synthetic_assigned": len(extra_by_split.get(split, [])),
            "synthetic_fraction_final": synthetic_fraction,
        }
        print(
            f"[{split}] total={stats['total']} "
            f"real={stats.get('kept_real_total', 0)} "
            f"direct={stats['kept_direct_solution']} "
            f"extracted={stats['kept_extracted_final']} "
            f"synthetic={stats['synthetic_added']} "
            f"written={len(built)} "
            f"synthetic_fraction={synthetic_fraction:.3f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "build_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
