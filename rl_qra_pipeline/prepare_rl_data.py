from __future__ import annotations

import argparse
import copy
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
ANSWER_KEYS = ["answer", "solution", "final_answer", "target", "output", "reference_completion"]

VALID_UNITS = [
    "m/s^2", "m/s", "mm", "cm", "km", "kg", "ms", "Hz",
    "m", "g", "s", "N", "J", "W", "V", "A",
]
UNIT_PATTERN = "(?:" + "|".join(re.escape(u) for u in sorted(VALID_UNITS, key=len, reverse=True)) + ")"
NUM_PATTERN = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
VAR_PATTERN = r"[A-Za-z][A-Za-z0-9_]*"

PURE_SYMBOLS = {
    "", "$", "$$", r"\(", r"\)", r"\[", r"\]", "[", "]", "(", ")",
    ".", ",", ":", ";", "=", "-", "+", "*", "/", "\\", "|", "<", ">",
}

BAD_CHARS_RE = re.compile(r"[�]|[）》《【】『』「」]|>>|<<")
FULLWIDTH_TRANS = str.maketrans({
    "−": "-", "–": "-", "—": "-",
    "，": ",", "。": ".", "：": ":", "；": ";",
    "（": "(", "）": ")", "［": "[", "］": "]",
})

FINAL_MARKERS = [
    r"therefore,?\s+the\s+final\s+answer\s+is",
    r"thus,?\s+the\s+final\s+answer\s+is",
    r"hence,?\s+the\s+final\s+answer\s+is",
    r"finally,?\s+the\s+final\s+answer\s+is",
    r"therefore,?\s+the\s+final\s+solution\s+is",
    r"thus,?\s+the\s+final\s+solution\s+is",
    r"hence,?\s+the\s+final\s+solution\s+is",
    r"finally,?\s+the\s+final\s+solution\s+is",
    r"so\s+the\s+solution\s+is",
    r"so\s+the\s+answer\s+is",
    r"the\s+answer\s+should\s+be",
    r"the\s+final\s+answer\s+is",
    r"final\s+answer\s*[:=]?",
    r"the\s+answer\s*[:=]?",
    r"answer\s*[:=]",
    r"the\s+solution\s*[:=]?",
    r"solution\s*[:=]",
    r"our\s+answer\s+is",
]
FINAL_MARKER_RE = re.compile(r"(?is)(?:" + "|".join(FINAL_MARKERS) + r")\s*")

FINAL_SENTENCES = [
    "Therefore, the final answer is {answer}.",
    "Thus, the final solution is {answer}.",
    "So the solution is {answer}.",
    "The answer should be {answer}.",
    "Finally, the answer is {answer}.",
    "Hence, the answer is {answer}.",
]


# ----------------------------- IO -----------------------------

def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {path}:{line_no}: {exc}") from exc
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


def split_candidates(input_dir: Path, split: str) -> List[Path]:
    names = [split]
    if split == "val":
        names += ["valid", "validation"]
    out: List[Path] = []
    for name in names:
        out += [
            input_dir / f"{name}.jsonl",
            input_dir / f"{name}.json",
            input_dir / name / "data.jsonl",
            input_dir / name / "data.json",
        ]
    return out


def load_split(input_dir: Path, split: str) -> Tuple[List[Dict[str, Any]], Optional[Path]]:
    for path in split_candidates(input_dir, split):
        rows = read_json_or_jsonl(path)
        if rows:
            return rows, path
    return [], None


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ----------------------------- QRA row parsing -----------------------------

def pick_first(row: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def strip_label_prefix(text: str, labels: List[str]) -> str:
    out = str(text or "").strip()
    for label in labels:
        out = re.sub(rf"(?is)^\s*{re.escape(label)}\s*[:：]?\s*", "", out).strip()
    return out


def get_segment_text(row: Dict[str, Any], wanted: List[str]) -> str:
    wanted_lower = {w.lower() for w in wanted}
    segs = row.get("segments")

    if isinstance(segs, dict):
        for key, value in segs.items():
            key_l = str(key).lower()
            if key_l in wanted_lower or any(w in key_l for w in wanted_lower):
                text = value.get("text", "") if isinstance(value, dict) else value
                if str(text).strip():
                    return str(text).strip()
        if "answer" in wanted_lower:
            for value in segs.values():
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
    text = get_segment_text(row, ["question", "problem", "prompt", "input"])
    if not text:
        text = pick_first(row, QUESTION_KEYS)
    return strip_label_prefix(text, ["Question", "Problem", "Prompt", "Input"])


def get_reasoning(row: Dict[str, Any]) -> str:
    text = get_segment_text(row, ["reasoning", "rationale", "thinking", "cot"])
    if not text:
        text = pick_first(row, REASONING_KEYS)
    return strip_label_prefix(text, ["Reasoning", "Rationale", "Thinking", "CoT"])


def get_answer_text(row: Dict[str, Any]) -> str:
    text = get_segment_text(row, ["answer", "target", "solution"])
    if text:
        return text
    return pick_first(row, ANSWER_KEYS)


# ----------------------------- answer cleaning / validity -----------------------------

def normalize_latex_units(text: str) -> str:
    out = str(text or "").translate(FULLWIDTH_TRANS)
    out = re.sub(r"\\(?:mathrm|text)\s*\{\s*~?\s*([^{}]+?)\s*\}", lambda m: " " + m.group(1).strip(), out)
    out = out.replace("\\,", " ").replace("\\;", " ").replace("~", " ")
    out = out.replace("\\left", "").replace("\\right", "")
    out = out.replace("\\cdot", "*").replace("\\times", "*")
    out = out.replace("\\pi", "pi")
    return out


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def remove_math_wrappers(text: str) -> str:
    out = str(text or "").strip().strip("` ")
    changed = True
    while changed:
        changed = False
        for left, right in [("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")]:
            if out.startswith(left) and out.endswith(right) and len(out) > len(left) + len(right):
                out = out[len(left): -len(right)].strip()
                changed = True
    boxed = re.fullmatch(r"\\boxed\s*\{(.+)\}", out, flags=re.S)
    if boxed:
        out = boxed.group(1).strip()
    return out


def clean_answer(raw: str) -> str:
    text = normalize_latex_units(str(raw or "").strip())
    text = remove_math_wrappers(text)
    text = strip_label_prefix(text, [
        "Final answer", "Final solution", "The final answer", "The final solution",
        "The answer", "The solution", "Answer", "Solution", "Our answer",
    ])
    text = normalize_spaces(text)
    while len(text) > 1 and text[-1] in ".;，。":
        text = text[:-1].strip()
    text = re.sub(r"\s*([=<>+*/^,()\[\]])\s*", r"\1", text)
    text = normalize_spaces(text)
    text = re.sub(rf"^({NUM_PATTERN})({UNIT_PATTERN})$", r"\1 \2", text)
    return text


def has_bad_shape(answer: str) -> bool:
    a = str(answer or "").strip()
    if BAD_CHARS_RE.search(a):
        return True
    if any(token in a for token in ["\\begin", "\\end", "aligned", "cases"]):
        return True
    if a.count("$") >= 2 and len(a) <= 8:
        return True
    return False


def is_pure_symbol(answer: str) -> bool:
    a = str(answer or "").strip()
    if a in PURE_SYMBOLS:
        return True
    return re.search(r"[A-Za-z0-9]", a) is None


def balanced_brackets(s: str) -> bool:
    pairs = {')': '(', ']': '[', '}': '{'}
    stack: List[str] = []
    for ch in s:
        if ch in "([{" :
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return not stack


def classify_answer(answer: str, allow_short_text: bool = False) -> Tuple[bool, str, str]:
    a = clean_answer(answer)
    compact = a.replace(" ", "")

    if not a:
        return False, "invalid", "empty"
    if len(a) > 128:
        return False, "invalid", "too_long"
    if "\n" in a:
        return False, "invalid", "newline"
    if has_bad_shape(a):
        return False, "invalid", "bad_symbol_chars"
    if is_pure_symbol(a):
        return False, "invalid", "pure_symbol"
    if not balanced_brackets(compact):
        return False, "invalid", "unbalanced_brackets"
    if compact.count("==") > 0 or compact.count("=>") > 0:
        return False, "invalid", "bad_equation_symbol"

    if re.fullmatch(r"[A-Ea-e]", compact):
        return True, "option", "valid"

    if re.fullmatch(NUM_PATTERN, compact):
        return True, "number", "valid"

    if re.fullmatch(rf"{NUM_PATTERN}{UNIT_PATTERN}", compact):
        return True, "unit_number", "valid"

    m = re.fullmatch(rf"([\(\[]){NUM_PATTERN},{NUM_PATTERN}([\)\]])", compact)
    if m:
        nums = re.findall(NUM_PATTERN, compact)
        if len(nums) == 2 and float(nums[0]) < float(nums[1]):
            return True, "interval", "valid"
        return False, "invalid", "bad_interval_order"

    if re.fullmatch(rf"{VAR_PATTERN}(?:<=|>=|<|>){NUM_PATTERN}", compact):
        return True, "inequality", "valid"

    if compact.count("=") == 1:
        lhs, rhs = compact.split("=", 1)
        if lhs and rhs and re.fullmatch(VAR_PATTERN, lhs) and re.fullmatch(r"[A-Za-z0-9_+\-*/^().\\]+", rhs):
            return True, "equation", "valid"
        return False, "invalid", "bad_equation"

    if re.fullmatch(rf"{NUM_PATTERN}/{NUM_PATTERN}", compact):
        return True, "fraction", "valid"

    allowed_expr = re.fullmatch(r"[A-Za-z0-9_+\-*/^().\\]+", compact) is not None
    has_letter = re.search(r"[A-Za-z]", compact) is not None
    has_digit = re.search(r"\d", compact) is not None
    has_operator = re.search(r"[+\-*/^()]", compact) is not None
    if allowed_expr and has_letter and has_digit and has_operator:
        return True, "symbolic_expression", "valid"

    if allow_short_text and re.fullmatch(r"[A-Za-z0-9_\-]+", compact) and len(compact) <= 24:
        return True, "short_text", "valid"

    return False, "invalid", "unsupported_form"


CANDIDATE_RE_LIST = [
    re.compile(rf"[\(\[]\s*{NUM_PATTERN}\s*,\s*{NUM_PATTERN}\s*[\)\]]"),
    re.compile(rf"{VAR_PATTERN}\s*(?:<=|>=|<|>)\s*{NUM_PATTERN}"),
    re.compile(rf"{VAR_PATTERN}\s*=\s*[A-Za-z0-9_+\-*/^().\\]+"),
    re.compile(rf"{NUM_PATTERN}\s*{UNIT_PATTERN}"),
    re.compile(rf"{NUM_PATTERN}\s*/\s*{NUM_PATTERN}"),
    re.compile(r"[A-Za-z0-9_+\-*/^().\\]*sqrt\s*\(?\s*\d+\s*\)?[A-Za-z0-9_+\-*/^().\\]*", re.I),
    re.compile(r"[A-Za-z][A-Za-z0-9_]*\^\d+(?:[+\-]\d*[A-Za-z][A-Za-z0-9_]*(?:\^\d+)?)*(?:[+\-]\d+)?"),
    re.compile(NUM_PATTERN),
    re.compile(r"\b[A-Ea-e]\b"),
]


def extract_final_answer(text: str, allow_short_text: bool = False) -> str:
    raw = normalize_latex_units(str(text or ""))
    if not raw.strip():
        return ""

    windows: List[str] = []
    marker_matches = list(FINAL_MARKER_RE.finditer(raw))
    if marker_matches:
        for match in marker_matches[-3:]:
            windows.append(raw[match.end(): match.end() + 300])
    else:
        windows.append(raw[-300:])

    for window in reversed(windows):
        w = window.replace("$$", " ").replace("$", " ")
        w = re.sub(r"\\\[|\\\]|\\\(|\\\)", " ", w)
        for cre in CANDIDATE_RE_LIST:
            matches = list(cre.finditer(w))
            for m in reversed(matches):
                cand = clean_answer(m.group(0))
                ok, _, _ = classify_answer(cand, allow_short_text=allow_short_text)
                if ok:
                    return cand
    return ""


def choose_answer(row: Dict[str, Any], allow_short_text: bool = False) -> Tuple[bool, str, str, str]:
    answer_text = get_answer_text(row)
    direct = clean_answer(answer_text)
    ok, kind, reason = classify_answer(direct, allow_short_text=allow_short_text)
    if ok:
        return True, direct, "direct_answer", kind

    extracted_from_answer = extract_final_answer(answer_text, allow_short_text=allow_short_text)
    if extracted_from_answer:
        ok2, kind2, _ = classify_answer(extracted_from_answer, allow_short_text=allow_short_text)
        if ok2:
            return True, extracted_from_answer, "extracted_from_answer", kind2

    reasoning = get_reasoning(row)
    extracted_from_reasoning = extract_final_answer(reasoning, allow_short_text=allow_short_text)
    if extracted_from_reasoning:
        ok3, kind3, _ = classify_answer(extracted_from_reasoning, allow_short_text=allow_short_text)
        if ok3:
            return True, extracted_from_reasoning, "extracted_from_reasoning", kind3

    return False, "", "drop", reason


def normalize_for_match(text: str) -> str:
    out = normalize_latex_units(text).lower()
    out = re.sub(r"\\boxed\s*\{([^{}]+)\}", r"\1", out)
    out = re.sub(r"[\s$`{}]", "", out)
    return out


def answer_matches_reasoning(reasoning: str, answer: str) -> bool:
    if not reasoning.strip() or not answer.strip():
        return False
    rn = normalize_for_match(reasoning)
    an = normalize_for_match(answer)
    if an and an in rn:
        return True
    extracted = extract_final_answer(reasoning, allow_short_text=True)
    return bool(extracted and normalize_for_match(extracted) == an)


# ----------------------------- row rewriting -----------------------------

def make_segments(question: str, reasoning: str, answer: str) -> Dict[str, Any]:
    return {
        "question": {"text": f"Question:\n{question.strip()}\n\n", "train": False},
        "reasoning": {"text": f"Reasoning:\n{reasoning.strip()}\n", "train": False},
        "answer_anchor": {"text": "\nAnswer:\n", "train": False},
        "answer": {"text": answer.strip(), "train": True},
    }


def update_answer_segment(row: Dict[str, Any], answer: str) -> Dict[str, Any]:
    new = copy.deepcopy(row)
    new["answer"] = answer
    new["solution"] = answer

    segs = new.get("segments")
    updated = False
    if isinstance(segs, dict):
        for key, value in segs.items():
            key_l = str(key).lower()
            if "answer" in key_l and "anchor" not in key_l:
                if isinstance(value, dict):
                    value["text"] = answer
                    value["train"] = True
                else:
                    segs[key] = answer
                updated = True
                break
        if not updated:
            for key, value in segs.items():
                if isinstance(value, dict) and value.get("train") is True:
                    value["text"] = answer
                    updated = True
                    break
        if not updated:
            segs["answer"] = {"text": answer, "train": True}
            updated = True
    elif isinstance(segs, list):
        for item in segs:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("key") or item.get("type") or "").lower()
            if "answer" in name or item.get("train") is True:
                item["text"] = answer
                item["train"] = True
                updated = True
                break
        if not updated:
            segs.append({"name": "answer", "text": answer, "train": True})
            updated = True

    if not updated:
        q = get_question(new)
        r = get_reasoning(new)
        new["segments"] = make_segments(q, r, answer)

    return new


def make_real_sample(row: Dict[str, Any], split: str, idx: int, answer: str, source: str, kind: str) -> Dict[str, Any]:
    new = update_answer_segment(row, answer)
    rid = row.get("id") or row.get("uid") or row.get("sample_id") or f"{split}_{idx:06d}"
    new["id"] = f"s1f_rl_{split}_{rid}"
    new["source"] = "s1f_rl_filtered_qra"
    new["original_source"] = row.get("source", "QRA")
    new["original_id"] = rid
    new["answer_source"] = source
    new["answer_kind"] = kind
    if not new.get("question"):
        new["question"] = get_question(row)
    if not new.get("reasoning"):
        new["reasoning"] = get_reasoning(row)
    return new


def make_synthetic_sample(split: str, idx: int, case: Dict[str, str]) -> Dict[str, Any]:
    answer = clean_answer(case["answer"])
    question = case["question"]
    reasoning = case["reasoning"]
    return {
        "id": f"s1f_rl_{split}_synthetic_{idx:06d}",
        "source": "s1f_rl_synthetic_hard",
        "answer_source": "parametric_synthetic",
        "answer_kind": case["kind"],
        "question": question,
        "reasoning": reasoning,
        "answer": answer,
        "solution": answer,
        "segments": make_segments(question, reasoning, answer),
    }


# ----------------------------- synthetic generation -----------------------------

def final_sentence(rng: random.Random, answer: str) -> str:
    return rng.choice(FINAL_SENTENCES).format(answer=answer)


def gen_interval_case(rng: random.Random) -> Dict[str, str]:
    a = rng.randint(-12, 8)
    b = rng.randint(a + 1, a + 15)
    left = rng.choice(["(", "["])
    right = rng.choice([")", "]"])
    answer = f"{left}{a},{b}{right}"
    var = rng.choice(["x", "t", "r", "y"])
    lower_symbol = "<" if left == "(" else "<="
    upper_symbol = "<" if right == ")" else "<="
    questions = [
        f"Write the interval notation for {a} {lower_symbol} {var} {upper_symbol} {b}.",
        f"Express all {var} between {a} and {b} using interval notation, with endpoint inclusion as specified.",
        f"Convert the compound inequality {a} {lower_symbol} {var} {upper_symbol} {b} into interval notation.",
    ]
    reasonings = [
        f"To solve, identify endpoints {a} and {b}. The left endpoint is {'included' if left == '[' else 'excluded'} and the right endpoint is {'included' if right == ']' else 'excluded'}. {final_sentence(rng, answer)}",
        f"First, translate each boundary sign into the correct bracket. Then place the lower endpoint before the upper endpoint. {final_sentence(rng, answer)}",
        f"We keep the values from {a} to {b}; the bracket type records whether each endpoint is allowed. {final_sentence(rng, answer)}",
    ]
    return {"kind": "interval", "question": rng.choice(questions), "reasoning": rng.choice(reasonings), "answer": answer}


def gen_inequality_case(rng: random.Random) -> Dict[str, str]:
    var = rng.choice(["x", "y", "t", "r"])
    c = rng.randint(-12, 12)
    op = rng.choice([">", ">=", "<", "<="])
    answer = f"{var}{op}{c}"
    questions = [
        f"State the short inequality answer for values of {var} satisfying {var} {op} {c}.",
        f"Write the solution condition using variable {var} and boundary {c}.",
        f"Give the final inequality when the allowed side is {op} {c}.",
    ]
    reasonings = [
        f"To solve, isolate the variable and preserve the comparison direction. {final_sentence(rng, answer)}",
        f"First, identify the boundary value {c}. Then write the permitted side with the correct inequality sign. {final_sentence(rng, answer)}",
        f"The result is already a one-variable inequality, so it can be reported directly. {final_sentence(rng, answer)}",
    ]
    return {"kind": "inequality", "question": rng.choice(questions), "reasoning": rng.choice(reasonings), "answer": answer}


def gen_equation_case(rng: random.Random) -> Dict[str, str]:
    mode = rng.choice(["linear", "line", "formula"])
    if mode == "linear":
        var = rng.choice(["x", "y", "a", "t"])
        val = rng.randint(-10, 18)
        coef = rng.choice([2, 3, 4, 5, 6])
        bias = rng.randint(-8, 8)
        rhs = coef * val + bias
        sign = "+" if bias >= 0 else ""
        answer = f"{var}={val}"
        question = f"Solve {coef}{var}{sign}{bias}={rhs}."
        reasoning = rng.choice([
            f"To solve, move the constant term first, then divide by {coef}. {final_sentence(rng, answer)}",
            f"First isolate the variable term. After division by the coefficient, we get the variable value. {final_sentence(rng, answer)}",
            f"We rearrange the linear equation until {var} is alone. {final_sentence(rng, answer)}",
        ])
    elif mode == "line":
        m = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5])
        b = rng.randint(-9, 9)
        answer = f"y={m}x" if b == 0 else f"y={m}x{'+' if b > 0 else ''}{b}"
        question = f"Write the line with slope {m} and y-intercept {b}."
        reasoning = rng.choice([
            f"First, use slope-intercept form y=mx+b. Substitute m={m} and b={b}. {final_sentence(rng, answer)}",
            f"The slope gives the coefficient of x and the intercept gives the constant term. {final_sentence(rng, answer)}",
        ])
    else:
        desc, answer, lead = rng.choice([
            ("Newton's second law solved for acceleration", "a=F/m", "Start with F=ma and divide by m."),
            ("distance from speed and time", "d=v*t", "Distance equals speed times time."),
            ("force from mass and acceleration", "F=ma", "Newton's second law gives force as mass times acceleration."),
            ("circle area", "A=pi*r^2", "The area of a circle is pi times radius squared."),
            ("momentum", "p=mv", "Momentum equals mass times velocity."),
        ])
        question = f"Write the formula for {desc}."
        reasoning = f"To solve, recall the standard symbolic relationship. {lead} {final_sentence(rng, answer)}"
    return {"kind": "equation", "question": question, "reasoning": reasoning, "answer": answer}


def gen_symbolic_case(rng: random.Random) -> Dict[str, str]:
    mode = rng.choice(["sqrt", "square", "fraction", "physics", "factor"])
    if mode == "sqrt":
        n = rng.choice([2, 3, 5, 6, 7])
        k = rng.randint(2, 9)
        answer = f"{k}sqrt({n})"
        question = f"Simplify sqrt({k*k*n})."
        reasoning = rng.choice([
            f"First, factor {k*k*n} as {k*k} times {n}. The square root of {k*k} is {k}. {final_sentence(rng, answer)}",
            f"To solve, factor out the largest perfect square. {final_sentence(rng, answer)}",
        ])
    elif mode == "square":
        c = rng.randint(1, 8)
        answer = f"x^2+{2*c}x+{c*c}"
        question = f"Expand (x+{c})^2."
        reasoning = f"First, use (a+b)^2=a^2+2ab+b^2. Substitute b={c}. {final_sentence(rng, answer)}"
    elif mode == "fraction":
        den = rng.choice([4, 5, 6, 7, 8, 9, 10, 12])
        num = rng.randint(1, den - 1)
        g = math.gcd(num, den)
        num_s, den_s = num // g, den // g
        scale = rng.choice([2, 3, 4, 5])
        answer = f"{num_s}/{den_s}"
        question = f"Simplify {num_s*scale}/{den_s*scale}."
        reasoning = f"To solve, divide the numerator and denominator by their common factor {scale}. {final_sentence(rng, answer)}"
    elif mode == "physics":
        answer = rng.choice(["v^2=u^2+2as", "E=mc^2", "p=mv"])
        question = "Write the requested physics relation in symbolic form."
        reasoning = f"First, recall the standard relationship and keep the variables symbolic. {final_sentence(rng, answer)}"
    else:
        c = rng.randint(1, 8)
        answer = f"(x+{c})^2"
        question = f"Factor x^2+{2*c}x+{c*c}."
        reasoning = f"First, recognize the perfect-square form x^2+2cx+c^2. Here c={c}. {final_sentence(rng, answer)}"
    return {"kind": "symbolic_expression", "question": question, "reasoning": reasoning, "answer": answer}


def gen_unit_case(rng: random.Random) -> Dict[str, str]:
    unit = rng.choice(["m", "mm", "cm", "kg", "g", "s", "ms", "N", "J", "m/s", "m/s^2"])
    if unit == "m/s^2" and rng.random() < 0.4:
        number = "-9.8"
    else:
        value = rng.choice([round(rng.uniform(-9.5, 9.5), 2), round(rng.uniform(0.05, 3.0), 3)])
        number = f"{value:.2f}" if unit in {"mm", "m", "cm"} and rng.random() < 0.45 else str(value)
    answer = f"{number} {unit}"
    questions = [
        f"Report the measured value {number} using the unit {unit}.",
        f"Give the final numeric result with unit {unit}: {number}.",
        f"Write the signed decimal value {number} together with the unit {unit}.",
    ]
    reasonings = [
        f"To solve, keep the sign and decimal value, then attach the required unit. {final_sentence(rng, answer)}",
        f"First, preserve the numeric precision. Then include the unit exactly once. {final_sentence(rng, answer)}",
        f"The result should be written as a short decimal followed by its unit. {final_sentence(rng, answer)}",
    ]
    return {"kind": "unit_number", "question": rng.choice(questions), "reasoning": rng.choice(reasonings), "answer": answer}


def generate_one_case(rng: random.Random) -> Dict[str, str]:
    gens = [gen_interval_case, gen_inequality_case, gen_equation_case, gen_symbolic_case, gen_unit_case]
    # Main hard cases: interval, equation, symbolic, unit decimal; keep inequality because x>c is useful.
    weights = [0.28, 0.15, 0.22, 0.20, 0.15]
    return rng.choices(gens, weights=weights, k=1)[0](rng)


def generate_synthetic_cases(count: int, seed: int, seen_answers: set[str]) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    out: List[Dict[str, str]] = []
    attempts = 0
    while len(out) < count and attempts < count * 120 + 1000:
        attempts += 1
        case = generate_one_case(rng)
        ans = clean_answer(case["answer"])
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
    if len(out) < count:
        print(f"[warn] only generated {len(out)}/{count} synthetic cases; uniqueness constraints may be too strict.")
    return out


# ----------------------------- build -----------------------------

def filter_qra_split(
    rows: List[Dict[str, Any]],
    split: str,
    require_reasoning_match: bool,
    allow_short_text: bool,
    max_drop_examples: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], set[str]]:
    out: List[Dict[str, Any]] = []
    seen_answers: set[str] = set()
    stats: Dict[str, Any] = {
        "raw_total": len(rows),
        "real_kept": 0,
        "drop_examples": [],
    }

    for idx, row in enumerate(rows):
        ok, answer, source, kind_or_reason = choose_answer(row, allow_short_text=allow_short_text)
        if not ok:
            key = f"drop_{kind_or_reason}"
            stats[key] = stats.get(key, 0) + 1
            if len(stats["drop_examples"]) < max_drop_examples:
                stats["drop_examples"].append({
                    "idx": idx,
                    "reason": key,
                    "raw_answer": get_answer_text(row)[:180],
                })
            continue

        if require_reasoning_match and not answer_matches_reasoning(get_reasoning(row), answer):
            stats["drop_reasoning_answer_mismatch"] = stats.get("drop_reasoning_answer_mismatch", 0) + 1
            if len(stats["drop_examples"]) < max_drop_examples:
                stats["drop_examples"].append({
                    "idx": idx,
                    "reason": "drop_reasoning_answer_mismatch",
                    "answer": answer,
                    "reasoning_tail": get_reasoning(row)[-180:],
                })
            continue

        valid, kind, reason = classify_answer(answer, allow_short_text=allow_short_text)
        if not valid:
            key = f"drop_{reason}"
            stats[key] = stats.get(key, 0) + 1
            if len(stats["drop_examples"]) < max_drop_examples:
                stats["drop_examples"].append({"idx": idx, "reason": key, "answer": answer})
            continue

        out.append(make_real_sample(row, split, idx, answer, source, kind))
        stats["real_kept"] += 1
        stats[f"kept_{kind}"] = stats.get(f"kept_{kind}", 0) + 1
        stats[f"kept_{source}"] = stats.get(f"kept_{source}", 0) + 1
        seen_answers.add(normalize_for_match(answer))

    return out, stats, seen_answers


def compute_synth_quotas(real_total: int, synthetic_final_fraction: float, min_eval: int) -> Dict[str, int]:
    if real_total <= 0:
        return {"train": 8 * min_eval, "val": min_eval, "test": min_eval}
    fraction = max(0.0, min(0.8, synthetic_final_fraction))
    target_total = int(round(real_total * fraction / max(1e-8, 1.0 - fraction)))
    k = max(int(min_eval), int(math.ceil(target_total / 10.0)))
    return {"train": 8 * k, "val": k, "test": k}


def build_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    loaded: Dict[str, Tuple[List[Dict[str, Any]], Optional[Path]]] = {
        split: load_split(input_dir, split) for split in ["train", "val", "test"]
    }
    raw_total = sum(len(rows) for rows, _ in loaded.values())
    if raw_total == 0 and not args.allow_empty_input:
        searched = {split: [str(p) for p in split_candidates(input_dir, split)] for split in ["train", "val", "test"]}
        raise RuntimeError(
            "No QRA data was loaded. This script now fails instead of silently generating synthetic-only data.\n"
            f"input_dir={input_dir}\nsearched={json.dumps(searched, ensure_ascii=False, indent=2)}"
        )

    report: Dict[str, Any] = {
        "base": "filtered_QRA_plus_synthetic_hard_cases_v6",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "raw_total_loaded": raw_total,
        "require_reasoning_match": bool(args.require_reasoning_match),
        "allow_short_text": bool(args.allow_short_text),
        "synthetic_final_fraction_target": args.synthetic_final_fraction,
        "min_synthetic_eval": args.min_synthetic_eval,
        "synthetic_split_rule": "8:1:1",
        "input_files": {split: str(path) if path else None for split, (_, path) in loaded.items()},
        "splits": {},
    }

    split_rows: Dict[str, List[Dict[str, Any]]] = {}
    global_seen: set[str] = set()

    for split in ["train", "val", "test"]:
        rows, path = loaded[split]
        real_rows, stats, seen = filter_qra_split(
            rows,
            split=split,
            require_reasoning_match=bool(args.require_reasoning_match),
            allow_short_text=bool(args.allow_short_text),
            max_drop_examples=int(args.max_drop_examples),
        )
        split_rows[split] = real_rows
        global_seen.update(seen)
        report["splits"][split] = {**stats, "input_file": str(path) if path else None}

    real_total = sum(len(rows) for rows in split_rows.values())
    quotas = compute_synth_quotas(real_total, float(args.synthetic_final_fraction), int(args.min_synthetic_eval))
    report["real_total_kept"] = real_total
    report["synthetic_quotas"] = quotas

    rng = random.Random(args.seed)
    for split in ["train", "val", "test"]:
        synth_cases = generate_synthetic_cases(
            count=quotas[split],
            seed=args.seed + {"train": 1001, "val": 2002, "test": 3003}[split],
            seen_answers=global_seen,
        )
        synth_rows = [make_synthetic_sample(split, i, c) for i, c in enumerate(synth_cases)]
        all_rows = split_rows[split] + synth_rows
        rng.shuffle(all_rows)
        split_rows[split] = all_rows
        info = report["splits"][split]
        info["synthetic_added"] = len(synth_rows)
        info["written"] = len(all_rows)
        info["synthetic_fraction_final"] = (len(synth_rows) / len(all_rows)) if all_rows else 0.0

    total_written = sum(len(v) for v in split_rows.values())
    report["total_written"] = total_written
    report["final_split_counts"] = {split: len(split_rows[split]) for split in ["train", "val", "test"]}
    if total_written:
        report["final_split_ratio"] = {
            split: len(split_rows[split]) / total_written for split in ["train", "val", "test"]
        }
        synth_total = sum(report["splits"][s]["synthetic_added"] for s in ["train", "val", "test"])
        report["synthetic_fraction_overall"] = synth_total / total_written

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in split_rows.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)
    (output_dir / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return report


def main() -> None:
    script_dir = Path(__file__).resolve().parent

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
        default=str(script_dir / "data" / "S1K_RL"),
        help="Output directory. Default: rl_qra_pipeline/data/S1K_RL.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--synthetic-final-fraction",
        type=float,
        default=0.25,
        help="Target synthetic fraction in the final dataset. Default 0.25. Quotas are rounded to 8:1:1.",
    )
    parser.add_argument(
        "--min-synthetic-eval",
        type=int,
        default=5,
        help="Minimum synthetic cases for val and test each. Train gets 8x this number. Default 5.",
    )
    parser.add_argument(
        "--require-reasoning-match",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, drop real QRA rows whose answer cannot be found in reasoning. Default false to avoid over-dropping.",
    )
    parser.add_argument(
        "--allow-short-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, allow generic short alphanumeric answers. Default false, keeping math/option/number forms only.",
    )
    parser.add_argument("--max-drop-examples", type=int, default=20)
    parser.add_argument("--allow-empty-input", action="store_true")
    args = parser.parse_args()

    report = build_dataset(args)

    print(f"Wrote {report['output_dir']}")
    print(f"Loaded raw QRA rows: {report['raw_total_loaded']}")
    print(f"Kept real rows: {report['real_total_kept']}")
    print(f"Synthetic quotas: {report['synthetic_quotas']}")
    for split in ["train", "val", "test"]:
        info = report["splits"][split]
        print(
            f"[{split}] input={info.get('input_file')} raw={info['raw_total']} "
            f"real_kept={info['real_kept']} synth={info['synthetic_added']} "
            f"written={info['written']} synth_frac={info['synthetic_fraction_final']:.3f}"
        )
    print("final_split_counts:", json.dumps(report.get("final_split_counts", {}), ensure_ascii=False))
    print("final_split_ratio:", json.dumps(report.get("final_split_ratio", {}), ensure_ascii=False))


if __name__ == "__main__":
    main()
