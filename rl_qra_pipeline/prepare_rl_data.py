from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


QUESTION_KEYS = ["question", "problem", "prompt", "input", "query"]
REASONING_KEYS = [
    "reasoning",
    "rationale",
    "thinking",
    "cot",
    "chain_of_thought",
    "deepseek_thinking_trajectory",
    "gemini_thinking_trajectory",
]
SOLUTION_KEYS = ["solution", "answer", "final_answer", "target", "output"]

VALID_UNITS = [
    "m/s^2", "m/s", "mm", "cm", "km", "kg", "ms", "Hz", "m", "g", "s", "N", "J", "W", "V", "A",
]
UNIT_PATTERN = "(?:" + "|".join(re.escape(u) for u in sorted(VALID_UNITS, key=len, reverse=True)) + ")"
NUM_PATTERN = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
VAR_PATTERN = r"[A-Za-z][A-Za-z0-9_]*"

FINAL_MARKERS = [
    r"therefore,?\s+the\s+final\s+answer\s+is",
    r"thus,?\s+the\s+final\s+answer\s+is",
    r"hence,?\s+the\s+final\s+answer\s+is",
    r"finally,?\s+the\s+final\s+answer\s+is",
    r"therefore,?\s+the\s+final\s+solution\s+is",
    r"thus,?\s+the\s+final\s+solution\s+is",
    r"hence,?\s+the\s+final\s+solution\s+is",
    r"finally,?\s+the\s+final\s+solution\s+is",
    r"the\s+final\s+answer\s+is",
    r"final\s+answer\s+is",
    r"final\s+answer\s*[:=]",
    r"the\s+answer\s+should\s+be",
    r"the\s+answer\s+is",
    r"answer\s*[:=]",
    r"the\s+solution\s+is",
    r"solution\s*[:=]",
    r"so\s+the\s+solution\s+is",
    r"so\s+the\s+answer\s+is",
    r"our\s+answer\s+is",
]
FINAL_MARKER_RE = re.compile(r"(?is)(?:" + "|".join(FINAL_MARKERS) + r")\s*")

BAD_DELIMITER_ONLY = {
    "", "$", "$$", r"\(", r"\)", r"\[", r"\]", "[", "]", "(", ")",
    ".", ",", ":", ";", "=", "-", "+", "*", "/", "\\", "|",
}

KEEP_REASON_FINAL_TEMPLATES = [
    "Therefore, the final answer is {answer}.",
    "Thus, the final solution is {answer}.",
    "So the solution is {answer}.",
    "The answer should be {answer}.",
    "Finally, the answer is {answer}.",
]


# -------------------------- basic IO --------------------------

def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        return rows
    obj = json.loads(text)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ["data", "rows", "examples", "train", "val", "valid", "validation", "test"]:
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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# -------------------------- field extraction --------------------------

def pick_first(row: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def get_segment_text(row: Dict[str, Any], wanted: List[str]) -> str:
    segs = row.get("segments")
    wanted_lower = {w.lower() for w in wanted}

    if isinstance(segs, dict):
        for key, value in segs.items():
            key_l = str(key).lower()
            if key_l in wanted_lower or any(w in key_l for w in wanted_lower):
                if isinstance(value, dict):
                    text = value.get("text", "")
                else:
                    text = value
                if str(text).strip():
                    return str(text).strip()

        # Fallback: answer segment is often the only segment with train=True.
        if "answer" in wanted_lower:
            for key, value in segs.items():
                if isinstance(value, dict) and value.get("train") is True:
                    text = value.get("text", "")
                    if str(text).strip():
                        return str(text).strip()

    if isinstance(segs, list):
        for item in segs:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("key") or item.get("type") or "").lower()
            if name in wanted_lower or any(w in name for w in wanted_lower):
                text = item.get("text", "")
                if str(text).strip():
                    return str(text).strip()
        if "answer" in wanted_lower:
            for item in segs:
                if isinstance(item, dict) and item.get("train") is True:
                    text = item.get("text", "")
                    if str(text).strip():
                        return str(text).strip()

    return ""


def get_question(row: Dict[str, Any]) -> str:
    text = get_segment_text(row, ["question", "prompt", "input"])
    if text:
        return strip_label_prefix(text, ["Question", "Problem", "Prompt", "Input"])
    return strip_label_prefix(pick_first(row, QUESTION_KEYS), ["Question", "Problem", "Prompt", "Input"])


def get_reasoning(row: Dict[str, Any]) -> str:
    text = get_segment_text(row, ["reasoning", "rationale", "thinking", "cot"])
    if text:
        return strip_label_prefix(text, ["Reasoning", "Rationale", "Thinking", "CoT"])
    return strip_label_prefix(pick_first(row, REASONING_KEYS), ["Reasoning", "Rationale", "Thinking", "CoT"])


def get_answer_source_text(row: Dict[str, Any]) -> str:
    text = get_segment_text(row, ["answer", "target", "solution"])
    if text:
        return text
    return pick_first(row, SOLUTION_KEYS)


def strip_label_prefix(text: str, labels: List[str]) -> str:
    out = str(text or "").strip()
    for label in labels:
        out = re.sub(rf"(?is)^\s*{re.escape(label)}\s*[:：]?\s*", "", out).strip()
    return out


# -------------------------- answer cleaning and typing --------------------------

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_latex_units(text: str) -> str:
    out = str(text or "")
    # \mathrm{~m}, \mathrm{m/s^2}, \text{ kg } -> m, kg
    out = re.sub(r"\\(?:mathrm|text)\s*\{\s*~?\s*([^{}]+?)\s*\}", lambda m: " " + m.group(1).strip(), out)
    out = out.replace("\\,", " ").replace("\\;", " ").replace("~", " ")
    out = out.replace("−", "-")
    return out


def remove_math_wrappers(text: str) -> str:
    out = str(text or "").strip()
    # remove code ticks first
    out = out.strip("` ")
    changed = True
    while changed:
        changed = False
        wrappers = [("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")]
        for left, right in wrappers:
            if out.startswith(left) and out.endswith(right) and len(out) > len(left) + len(right):
                out = out[len(left): -len(right)].strip()
                changed = True
    boxed = re.fullmatch(r"\\boxed\s*\{(.+)\}", out, flags=re.S)
    if boxed:
        out = boxed.group(1).strip()
    return out


def clean_candidate_answer(raw: str) -> str:
    text = normalize_latex_units(str(raw or "").strip())
    text = remove_math_wrappers(text)
    text = strip_label_prefix(text, [
        "Final answer", "Final solution", "The final answer", "The answer", "Answer", "Solution", "Our answer",
    ])
    text = normalize_spaces(text)
    # Remove trailing sentence punctuation, but keep interval bracket and decimals.
    while len(text) > 1 and text[-1] in ".;，。":
        text = text[:-1].strip()
    # Normalize spaces around common expression operators, except keep unit space.
    text = re.sub(r"\s*([=<>+*/^,()\[\]])\s*", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Put back one space between number and unit if it became glued.
    text = re.sub(rf"^({NUM_PATTERN})({UNIT_PATTERN})$", r"\1 \2", text)
    return text


def is_pure_symbol(answer: str) -> bool:
    a = str(answer or "").strip()
    if a in BAD_DELIMITER_ONLY:
        return True
    # No letter/digit at all means pure punctuation/symbol noise.
    if not re.search(r"[A-Za-z0-9]", a):
        return True
    return False


def balanced_brackets(s: str) -> bool:
    pairs = {')': '(', ']': '[', '}': '{'}
    stack: List[str] = []
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return not stack


def classify_answer(answer: str) -> Tuple[bool, str, str]:
    """Return (is_valid, kind, reason)."""
    a = clean_candidate_answer(answer)
    compact = a.replace(" ", "")

    if not a:
        return False, "invalid", "empty"
    if is_pure_symbol(a):
        return False, "invalid", "pure_symbol"
    if len(a) > 96:
        return False, "invalid", "too_long"
    if "\n" in a:
        return False, "invalid", "newline"
    if not balanced_brackets(compact):
        return False, "invalid", "unbalanced_brackets"
    if compact.count("$$") or compact in BAD_DELIMITER_ONLY:
        return False, "invalid", "math_delimiter_noise"

    # option letter
    if re.fullmatch(r"[A-Ea-e]", compact):
        return True, "option", "valid"

    # integer / decimal
    if re.fullmatch(NUM_PATTERN, compact):
        return True, "number", "valid"

    # unit decimal / unit number
    if re.fullmatch(rf"{NUM_PATTERN}{UNIT_PATTERN}", compact):
        return True, "unit_number", "valid"

    # interval: (a,b], [a,b), etc. Require left < right.
    m = re.fullmatch(rf"([\(\[])(?:\s*)({NUM_PATTERN})(?:\s*),(?:\s*)({NUM_PATTERN})(?:\s*)([\)\]])", a)
    if m:
        left = float(m.group(2))
        right = float(m.group(3))
        if left < right:
            return True, "interval", "valid"
        return False, "invalid", "bad_interval_order"

    # inequality: x>3, y<=-2
    if re.fullmatch(rf"{VAR_PATTERN}(?:<=|>=|<|>){NUM_PATTERN}", compact):
        return True, "inequality", "valid"

    # equation: exactly one equality, non-empty sides, no x==3
    if compact.count("=") == 1:
        lhs, rhs = compact.split("=", 1)
        if lhs and rhs and re.fullmatch(VAR_PATTERN, lhs):
            if re.fullmatch(r"[A-Za-z0-9_+\-*/^().\\]+", rhs):
                return True, "equation", "valid"
        return False, "invalid", "bad_equation"

    # fraction
    if re.fullmatch(rf"{NUM_PATTERN}/{NUM_PATTERN}", compact):
        return True, "fraction", "valid"

    # symbolic expression with variables/numbers and operators.
    allowed_symbolic = re.fullmatch(r"[A-Za-z0-9_+\-*/^().\\]+", compact) is not None
    has_content = re.search(r"[A-Za-z]", compact) and re.search(r"[0-9]", compact)
    has_operator = re.search(r"[+\-*/^()]", compact) is not None
    if allowed_symbolic and has_content and has_operator:
        return True, "symbolic_expression", "valid"

    # Very short alphanumeric text is allowed, but long phrases are not useful for this RL dataset.
    if re.fullmatch(r"[A-Za-z0-9_\-]+", compact) and len(compact) <= 24:
        return True, "short_text", "valid"

    return False, "invalid", "unsupported_form"


def normalize_for_match(text: str) -> str:
    out = normalize_latex_units(text).lower()
    out = re.sub(r"\\boxed\s*\{([^{}]+)\}", r"\1", out)
    out = re.sub(r"[\s$`{}]", "", out)
    out = out.replace("−", "-")
    return out


def answer_matches_reasoning(reasoning: str, answer: str) -> bool:
    if not reasoning.strip() or not answer.strip():
        return False
    rn = normalize_for_match(reasoning)
    an = normalize_for_match(answer)
    if an and an in rn:
        return True
    extracted = extract_final_answer(reasoning)
    if extracted:
        return normalize_for_match(extracted) == an
    return False


# -------------------------- final answer extraction --------------------------

def candidate_patterns() -> List[re.Pattern[str]]:
    # Ordered from structured to generic. Use raw text after normalizing latex units.
    patterns = [
        rf"[\(\[]\s*{NUM_PATTERN}\s*,\s*{NUM_PATTERN}\s*[\)\]]",              # interval
        rf"{VAR_PATTERN}\s*(?:<=|>=|<|>)\s*{NUM_PATTERN}",                        # inequality
        rf"{VAR_PATTERN}\s*=\s*[A-Za-z0-9_+\-*/^().\\]+",                         # equation
        rf"{NUM_PATTERN}\s*{UNIT_PATTERN}",                                       # unit number
        rf"{NUM_PATTERN}\s*/\s*{NUM_PATTERN}",                                    # fraction
        r"[A-Za-z0-9_+\-*/^().\\]*sqrt\s*\(?\s*\d+\s*\)?[A-Za-z0-9_+\-*/^().\\]*", # sqrt expr
        r"[A-Za-z][A-Za-z0-9_]*\^\d+(?:[+\-]\d*[A-Za-z][A-Za-z0-9_]*(?:\^\d+)?)*(?:[+\-]\d+)?", # polynomial-ish
        rf"{NUM_PATTERN}",                                                        # number
        r"\b[A-Ea-e]\b",                                                          # option
    ]
    return [re.compile(p, re.I) for p in patterns]


CANDIDATE_RE_LIST = candidate_patterns()


def extract_final_answer(text: str) -> str:
    raw = normalize_latex_units(str(text or ""))
    if not raw.strip():
        return ""

    matches = list(FINAL_MARKER_RE.finditer(raw))
    windows: List[str] = []
    if matches:
        for m in matches[-3:]:
            tail = raw[m.end():]
            # Do not stop immediately at decimal point; take a conservative window.
            tail = tail[:240]
            windows.append(tail)
    else:
        # Fallback: last 240 chars, useful for solution fields with no explicit marker.
        windows.append(raw[-240:])

    best = ""
    for window in reversed(windows):
        # Remove display math wrappers but keep contents.
        w = window.replace("$$", " ").replace("$", " ")
        w = re.sub(r"\\\[|\\\]|\\\(|\\\)", " ", w)
        for cre in CANDIDATE_RE_LIST:
            candidates = list(cre.finditer(w))
            if not candidates:
                continue
            # The answer is usually the last structured token in the final window.
            for match in reversed(candidates):
                cand = clean_candidate_answer(match.group(0))
                ok, _, _ = classify_answer(cand)
                if ok:
                    return cand
                if not best:
                    best = cand
    return ""


def choose_answer_for_row(row: Dict[str, Any]) -> Tuple[bool, str, str, str]:
    """Return (ok, cleaned_answer, source, reason)."""
    answer_text = get_answer_source_text(row)
    direct = clean_candidate_answer(answer_text)
    ok, kind, reason = classify_answer(direct)
    if ok:
        return True, direct, "direct_answer", kind

    reasoning = get_reasoning(row)
    extracted = extract_final_answer(reasoning)
    if extracted:
        ok2, kind2, reason2 = classify_answer(extracted)
        if ok2:
            return True, extracted, "extracted_from_reasoning", kind2

    # Last resort: if solution/answer was long but has a final marker or a final numeric/unit answer.
    extracted2 = extract_final_answer(answer_text)
    if extracted2:
        ok3, kind3, reason3 = classify_answer(extracted2)
        if ok3:
            return True, extracted2, "extracted_from_solution", kind3

    return False, "", "drop", reason


# -------------------------- sample construction --------------------------

def make_segments(question: str, reasoning: str, answer: str) -> Dict[str, Any]:
    return {
        "question": {"text": f"Question:\n{question.strip()}\n\n", "train": False},
        "reasoning": {"text": f"Reasoning:\n{reasoning.strip()}\n", "train": False},
        "answer_anchor": {"text": "\nAnswer:\n", "train": False},
        "answer": {"text": answer.strip(), "train": True},
    }


def make_real_sample(row: Dict[str, Any], split: str, idx: int, answer: str, answer_source: str, answer_kind: str) -> Dict[str, Any]:
    question = get_question(row)
    reasoning = get_reasoning(row)
    rid = row.get("id") or row.get("uid") or row.get("sample_id") or f"{split}_{idx:06d}"
    return {
        "id": f"s1f_rl_{split}_{rid}",
        "source": "s1f_rl_filtered_qra",
        "answer_source": answer_source,
        "answer_kind": answer_kind,
        "original_id": rid,
        "question": question,
        "reasoning": reasoning,
        "answer": answer,
        "solution": answer,
        "segments": make_segments(question, reasoning, answer),
    }


def make_synthetic_sample(split: str, idx: int, case: Dict[str, str]) -> Dict[str, Any]:
    answer = clean_candidate_answer(case["answer"])
    return {
        "id": f"s1f_rl_{split}_synthetic_{idx:06d}",
        "source": "s1f_rl_synthetic_hard",
        "answer_source": "parametric_synthetic",
        "answer_kind": case["kind"],
        "question": case["question"],
        "reasoning": case["reasoning"],
        "answer": answer,
        "solution": answer,
        "segments": make_segments(case["question"], case["reasoning"], answer),
    }


# -------------------------- synthetic generation --------------------------

def final_sentence(rng: random.Random, answer: str) -> str:
    templates = [
        "Therefore, the final answer is {answer}.",
        "Thus, the final solution is {answer}.",
        "So the solution is {answer}.",
        "The answer should be {answer}.",
        "Finally, the answer is {answer}.",
        "Hence, the answer is {answer}.",
    ]
    return rng.choice(templates).format(answer=answer)


def gen_interval_case(rng: random.Random) -> Dict[str, str]:
    a = rng.randint(-8, 5)
    b = rng.randint(a + 1, a + 12)
    left = rng.choice(["(", "["])
    right = rng.choice([")", "]"])
    answer = f"{left}{a},{b}{right}"
    lower_op = ">" if left == "(" else ">="
    upper_op = "<" if right == ")" else "<="
    var = rng.choice(["x", "t", "r", "y"])
    question_templates = [
        f"Write the interval notation for all {var} satisfying {a} {lower_op.replace('>', '<')} {var} {upper_op} {b}.",
        f"Express the solution set {var}{lower_op}{a} and {var}{upper_op}{b} as an interval.",
        f"Convert the compound inequality {a} {lower_op.replace('>', '<')} {var} {upper_op} {b} into interval notation.",
    ]
    reasoning_templates = [
        f"To solve, identify the two endpoints {a} and {b}. The left endpoint is {'included' if left == '[' else 'excluded'}, and the right endpoint is {'included' if right == ']' else 'excluded'}. {final_sentence(rng, answer)}",
        f"First, translate each inequality sign into an endpoint bracket. The lower bound gives {left}, and the upper bound gives {right}. {final_sentence(rng, answer)}",
        f"We keep all values between {a} and {b}. The endpoint convention determines the brackets. {final_sentence(rng, answer)}",
    ]
    return {"kind": "interval", "question": rng.choice(question_templates), "reasoning": rng.choice(reasoning_templates), "answer": answer}


def gen_inequality_case(rng: random.Random) -> Dict[str, str]:
    var = rng.choice(["x", "y", "t", "r"])
    c = rng.randint(-9, 9)
    op = rng.choice([">", ">=", "<", "<="])
    answer = f"{var}{op}{c}"
    question_templates = [
        f"Give a short inequality answer describing all values of {var} that are {op} {c}.",
        f"Write the solution condition using {var} and the boundary {c}.",
        f"State the final inequality when the allowed values satisfy {var} {op} {c}.",
    ]
    reasoning_templates = [
        f"To solve, isolate the variable and keep the comparison direction. {final_sentence(rng, answer)}",
        f"First, identify the boundary value {c}. Then write the allowed side using the same inequality sign. {final_sentence(rng, answer)}",
        f"The condition is already in one-variable inequality form, so we report it directly. {final_sentence(rng, answer)}",
    ]
    return {"kind": "inequality", "question": rng.choice(question_templates), "reasoning": rng.choice(reasoning_templates), "answer": answer}


def gen_equation_case(rng: random.Random) -> Dict[str, str]:
    mode = rng.choice(["linear", "line", "formula"])
    if mode == "linear":
        var = rng.choice(["x", "y", "a", "t"])
        val = rng.randint(-9, 15)
        coef = rng.choice([2, 3, 4, 5])
        bias = rng.randint(-6, 6)
        rhs = coef * val + bias
        answer = f"{var}={val}"
        question = f"Solve {coef}{var}+{bias}={rhs}."
        reasoning = rng.choice([
            f"To solve, subtract {bias} from both sides and divide by {coef}. {final_sentence(rng, answer)}",
            f"First isolate the variable term, then divide by the coefficient. {final_sentence(rng, answer)}",
            f"We rearrange the linear equation until the variable is alone. {final_sentence(rng, answer)}",
        ])
    elif mode == "line":
        m = rng.choice([-3, -2, -1, 1, 2, 3, 4])
        b = rng.randint(-8, 8)
        sign = "+" if b >= 0 else ""
        answer = f"y={m}x{sign}{b}" if b != 0 else f"y={m}x"
        question = f"Write the line with slope {m} and intercept {b}."
        reasoning = rng.choice([
            f"First, use y=mx+b. Substitute m={m} and b={b}. {final_sentence(rng, answer)}",
            f"The slope-intercept form is y=mx+b, so plug in the given values. {final_sentence(rng, answer)}",
        ])
    else:
        formulas = [
            ("Newton's second law solved for acceleration", "a=F/m", "Start from F=ma and divide by m."),
            ("distance from speed and time", "d=v*t", "Use distance equals speed times time."),
            ("force from mass and acceleration", "F=ma", "Use Newton's second law directly."),
            ("circle area formula", "A=pi*r^2", "The area of a circle is pi times radius squared."),
        ]
        desc, answer, lead = rng.choice(formulas)
        question = f"Write the formula for {desc}."
        reasoning = f"To solve, recall the standard relationship. {lead} {final_sentence(rng, answer)}"
    return {"kind": "equation", "question": question, "reasoning": reasoning, "answer": answer}


def gen_symbolic_case(rng: random.Random) -> Dict[str, str]:
    mode = rng.choice(["sqrt", "square", "fraction", "physics", "factor"])
    if mode == "sqrt":
        n = rng.choice([2, 3, 5, 6, 7])
        k = rng.randint(2, 8)
        answer = f"{k}sqrt({n})"
        question = f"Simplify sqrt({k*k*n})."
        reasoning = rng.choice([
            f"First, split {k*k*n} into {k*k} times {n}. The square root of {k*k} is {k}. {final_sentence(rng, answer)}",
            f"To solve, factor out the largest perfect square. {final_sentence(rng, answer)}",
        ])
    elif mode == "square":
        c = rng.randint(1, 6)
        answer = f"x^2+{2*c}x+{c*c}"
        question = f"Expand (x+{c})^2."
        reasoning = f"First, use (a+b)^2=a^2+2ab+b^2. Substitute b={c}. {final_sentence(rng, answer)}"
    elif mode == "fraction":
        den = rng.choice([4, 5, 6, 7, 8, 9])
        num = rng.randint(1, den - 1)
        g = math.gcd(num, den)
        num2, den2 = num // g, den // g
        scale = rng.choice([2, 3, 4])
        answer = f"{num2}/{den2}"
        question = f"Simplify {num2*scale}/{den2*scale}."
        reasoning = f"To solve, divide numerator and denominator by their common factor {scale}. {final_sentence(rng, answer)}"
    elif mode == "physics":
        answer = rng.choice(["v^2=u^2+2as", "p=mv", "E=mc^2"])
        question = "Write the requested physics relation in symbolic form."
        reasoning = f"First, recall the standard symbolic relationship and keep all variables in formula form. {final_sentence(rng, answer)}"
    else:
        c = rng.randint(1, 6)
        answer = f"(x+{c})^2"
        question = f"Factor x^2+{2*c}x+{c*c}."
        reasoning = f"First, recognize the perfect-square pattern x^2+2cx+c^2. Here c={c}. {final_sentence(rng, answer)}"
    return {"kind": "symbolic_expression", "question": question, "reasoning": reasoning, "answer": answer}


def gen_unit_case(rng: random.Random) -> Dict[str, str]:
    unit = rng.choice(VALID_UNITS)
    value = rng.choice([round(rng.uniform(-9.5, 9.5), 2), round(rng.uniform(0.05, 3.0), 3)])
    # Keep human-readable trailing zeros sometimes for mm/m cases.
    if unit in {"mm", "m", "cm"}:
        number = f"{value:.2f}" if rng.random() < 0.4 else str(value)
    else:
        number = str(value)
    answer = f"{number} {unit}"
    question_templates = [
        f"Report the measured value {number} using the unit {unit}.",
        f"Give the final numeric result with unit {unit}: {number}.",
        f"Write the signed decimal value {number} together with the unit {unit}.",
    ]
    reasoning_templates = [
        f"To solve, keep the sign and decimal value, then attach the required unit. {final_sentence(rng, answer)}",
        f"First, preserve the numeric precision. Then include the unit exactly once. {final_sentence(rng, answer)}",
        f"The result should be written as a short decimal followed by its unit. {final_sentence(rng, answer)}",
    ]
    return {"kind": "unit_number", "question": rng.choice(question_templates), "reasoning": rng.choice(reasoning_templates), "answer": answer}


def generate_one_case(rng: random.Random) -> Dict[str, str]:
    gens = [gen_interval_case, gen_inequality_case, gen_equation_case, gen_symbolic_case, gen_unit_case]
    weights = [0.24, 0.12, 0.24, 0.24, 0.16]
    return rng.choices(gens, weights=weights, k=1)[0](rng)


def generate_synthetic_cases(count: int, seed: int, seen_answers: set[str]) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    out: List[Dict[str, str]] = []
    attempts = 0
    while len(out) < count and attempts < count * 80 + 500:
        attempts += 1
        case = generate_one_case(rng)
        ans = clean_candidate_answer(case["answer"])
        ok, kind, _ = classify_answer(ans)
        if not ok:
            continue
        key = normalize_for_match(ans)
        if key in seen_answers:
            continue
        seen_answers.add(key)
        case["answer"] = ans
        case["kind"] = kind
        out.append(case)
    return out


# -------------------------- split construction --------------------------

def filter_real_split(
    rows: List[Dict[str, Any]],
    split: str,
    require_reasoning_match: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], set[str]]:
    out: List[Dict[str, Any]] = []
    stats: Dict[str, int] = {
        "raw_total": len(rows),
        "real_kept": 0,
    }
    seen_answers: set[str] = set()

    for idx, row in enumerate(rows):
        ok, answer, answer_source, kind_or_reason = choose_answer_for_row(row)
        if not ok:
            key = f"drop_{kind_or_reason}"
            stats[key] = stats.get(key, 0) + 1
            continue

        reasoning = get_reasoning(row)
        if require_reasoning_match and not answer_matches_reasoning(reasoning, answer):
            stats["drop_reasoning_answer_mismatch"] = stats.get("drop_reasoning_answer_mismatch", 0) + 1
            continue

        valid, kind, reason = classify_answer(answer)
        if not valid:
            stats[f"drop_{reason}"] = stats.get(f"drop_{reason}", 0) + 1
            continue

        sample = make_real_sample(row, split, idx, answer, answer_source, kind)
        out.append(sample)
        stats["real_kept"] += 1
        stats[f"kept_{kind}"] = stats.get(f"kept_{kind}", 0) + 1
        stats[f"kept_{answer_source}"] = stats.get(f"kept_{answer_source}", 0) + 1
        seen_answers.add(normalize_for_match(answer))

    return out, stats, seen_answers


def compute_synthetic_quota(
    real_count: int,
    split: str,
    synthetic_ratio: float,
    min_train: int,
    min_eval: int,
    max_synthetic_fraction: float,
) -> int:
    min_n = min_train if split == "train" else min_eval
    base = max(min_n, int(round(real_count * synthetic_ratio)))
    if max_synthetic_fraction > 0 and real_count > 0:
        # synth / (real + synth) <= max_fraction
        max_n = int((max_synthetic_fraction * real_count) / max(1e-8, (1.0 - max_synthetic_fraction)))
        base = min(base, max(min_n, max_n))
    return max(0, base)


def build_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    rng = random.Random(args.seed)

    report: Dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "base": "filtered_qra_plus_parametric_synthetic",
        "require_reasoning_match": bool(args.require_reasoning_match),
        "synthetic_ratio": args.synthetic_ratio,
        "min_synthetic_train": args.min_synthetic_train,
        "min_synthetic_eval": args.min_synthetic_eval,
        "max_synthetic_fraction": args.max_synthetic_fraction,
        "splits": {},
    }

    global_seen: set[str] = set()
    split_rows: Dict[str, List[Dict[str, Any]]] = {}

    for split in ["train", "val", "test"]:
        raw_rows = load_split(input_dir, split)
        real_rows, stats, seen = filter_real_split(
            raw_rows,
            split=split,
            require_reasoning_match=bool(args.require_reasoning_match),
        )
        global_seen.update(seen)

        quota = compute_synthetic_quota(
            real_count=len(real_rows),
            split=split,
            synthetic_ratio=float(args.synthetic_ratio),
            min_train=int(args.min_synthetic_train),
            min_eval=int(args.min_synthetic_eval),
            max_synthetic_fraction=float(args.max_synthetic_fraction),
        )
        synth_cases = generate_synthetic_cases(
            count=quota,
            seed=args.seed + {"train": 101, "val": 202, "test": 303}[split],
            seen_answers=global_seen,
        )
        synth_rows = [make_synthetic_sample(split, i, c) for i, c in enumerate(synth_cases)]
        all_rows = real_rows + synth_rows
        rng.shuffle(all_rows)
        split_rows[split] = all_rows

        final_count = len(all_rows)
        synth_count = len(synth_rows)
        report["splits"][split] = {
            **stats,
            "synthetic_quota": quota,
            "synthetic_added": synth_count,
            "written": final_count,
            "synthetic_fraction_final": (synth_count / final_count) if final_count else 0.0,
        }

    # Do not move real samples across splits. This just reports the achieved ratio.
    total_written = sum(len(v) for v in split_rows.values())
    report["total_written"] = total_written
    if total_written:
        report["final_split_ratio"] = {
            split: len(split_rows[split]) / total_written for split in ["train", "val", "test"]
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in split_rows.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)
    (output_dir / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return report


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(script_dir / "data" / "QRA"),
        help="Input QRA data directory. Default: rl_qra_pipeline/data/QRA.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(script_dir / "data" / "S1F_RL"),
        help="Output directory. Default: rl_qra_pipeline/data/S1F_RL.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-ratio", type=float, default=0.25)
    parser.add_argument("--min-synthetic-train", type=int, default=20)
    parser.add_argument("--min-synthetic-eval", type=int, default=8)
    parser.add_argument("--max-synthetic-fraction", type=float, default=0.35)
    parser.add_argument(
        "--require-reasoning-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop real QRA rows whose answer cannot be matched in reasoning. Default: true.",
    )
    args = parser.parse_args()

    report = build_dataset(args)
    print(f"Wrote {report['output_dir']}")
    for split, info in report["splits"].items():
        print(
            f"[{split}] raw={info['raw_total']} real_kept={info['real_kept']} "
            f"synthetic={info['synthetic_added']} written={info['written']} "
            f"synthetic_frac={info['synthetic_fraction_final']:.3f}"
        )
    if "final_split_ratio" in report:
        print("final_split_ratio:", json.dumps(report["final_split_ratio"], ensure_ascii=False))


if __name__ == "__main__":
    main()
