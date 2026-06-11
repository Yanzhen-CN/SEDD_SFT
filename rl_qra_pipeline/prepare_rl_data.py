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
ANSWER_KEYS = ["answer", "solution", "final_answer", "target", "output", "reference_completion", "completion"]

VALID_UNITS = [
    "m/s^2", "m/s", "mm", "cm", "km", "kg", "ms", "Hz",
    "m", "g", "s", "N", "J", "W", "V", "A",
]
UNIT_PATTERN = "(?:" + "|".join(re.escape(u) for u in sorted(VALID_UNITS, key=len, reverse=True)) + ")"
NUM_PATTERN = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
VAR_PATTERN = r"[A-Za-z][A-Za-z0-9_]*"

PURE_SYMBOLS = {
    "", "$", "$$", r"\(", r"\)", r"\[", r"\]",
    "[", "]", "(", ")", "{", "}", ".", ",", ":", ";",
    "=", "-", "+", "*", "/", "\\", "|", "<", ">", "_", "^",
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
    r"the\s+required\s+answer\s+is",
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


def is_anchor_like_text(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or "").lower())
    return compact in {"answer:", "answer", "finalanswer:", "finalanswer", "solution:", "solution"}


def is_bad_answer_segment_key(key: str) -> bool:
    key_l = str(key or "").lower()
    # In QRA data there is often an answer_anchor segment before the real answer.
    # Never treat it as the answer itself.
    return any(x in key_l for x in ["anchor", "prefix", "prompt", "header"])


def get_segment_text(row: Dict[str, Any], wanted: List[str]) -> str:
    wanted_lower = {w.lower() for w in wanted}
    segs = row.get("segments")
    wants_answer = "answer" in wanted_lower or "target" in wanted_lower or "solution" in wanted_lower

    def clean_candidate(key: str, text: Any) -> str:
        if wants_answer and is_bad_answer_segment_key(key):
            return ""
        out = str(text or "").strip()
        if wants_answer and is_anchor_like_text(out):
            return ""
        return out

    if isinstance(segs, dict):
        # 1) Exact key match first. This prevents answer_anchor from shadowing answer.
        for key, value in segs.items():
            key_l = str(key).lower()
            if key_l in wanted_lower:
                text = value.get("text", "") if isinstance(value, dict) else value
                out = clean_candidate(key_l, text)
                if out:
                    return out

        # 2) For answers, prefer train=True segments over fuzzy name matching.
        if wants_answer:
            for key, value in segs.items():
                if is_bad_answer_segment_key(str(key)):
                    continue
                if isinstance(value, dict) and value.get("train") is True:
                    out = clean_candidate(str(key), value.get("text", ""))
                    if out:
                        return out

        # 3) Fuzzy match only after exact and train=True fail.
        for key, value in segs.items():
            key_l = str(key).lower()
            if wants_answer and is_bad_answer_segment_key(key_l):
                continue
            if any(w in key_l for w in wanted_lower):
                text = value.get("text", "") if isinstance(value, dict) else value
                out = clean_candidate(key_l, text)
                if out:
                    return out

    if isinstance(segs, list):
        # 1) Exact named segment first.
        for item in segs:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("key") or item.get("type") or "").lower()
            if name in wanted_lower:
                out = clean_candidate(name, item.get("text", ""))
                if out:
                    return out

        # 2) train=True for answer.
        if wants_answer:
            for item in segs:
                name = str(item.get("name") or item.get("key") or item.get("type") or "").lower()
                if is_bad_answer_segment_key(name):
                    continue
                if item.get("train") is True:
                    out = clean_candidate(name, item.get("text", ""))
                    if out:
                        return out

        # 3) Fuzzy named segment.
        for item in segs:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("key") or item.get("type") or "").lower()
            if wants_answer and is_bad_answer_segment_key(name):
                continue
            if any(w in name for w in wanted_lower):
                out = clean_candidate(name, item.get("text", ""))
                if out:
                    return out

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
    # This is the critical part: prepared QRA often stores answer only in segments/train=True,
    # not as top-level row["answer"].
    text = get_segment_text(row, ["answer", "target", "solution"])
    if text:
        return text
    return pick_first(row, ANSWER_KEYS)


# ----------------------------- answer cleaning / validity -----------------------------

def normalize_latex_units(text: str) -> str:
    out = str(text or "").translate(FULLWIDTH_TRANS)
    out = re.sub(r"\\(?:mathrm|text)\s*\{\s*~?\s*([^{}]+?)\s*\}", lambda m: " " + m.group(1).strip(), out)
    out = out.replace("\\,", " ").replace("\\;", " ").replace("~", " ")
    out = out.replace("\\cdot", "*").replace("\\times", "*")
    out = out.replace("\\pi", "pi").replace("π", "pi")
    out = re.sub(r"\\sqrt\s*\{\s*([^{}]+?)\s*\}", r"sqrt(\1)", out)
    out = re.sub(r"\\frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}", r"\1/\2", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def strip_math_wrappers(text: str) -> str:
    out = str(text or "").strip()
    out = re.sub(r"(?is)^\s*(?:answer|solution|final answer|final solution|the answer|the solution|our answer)\s*(?:is|=|:)?\s*", "", out).strip()
    for left, right in [("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")]:
        if out.startswith(left) and out.endswith(right) and len(out) > len(left) + len(right):
            out = out[len(left):-len(right)].strip()
    boxed = re.fullmatch(r"\\boxed\{(.+)\}", out, flags=re.S)
    if boxed:
        out = boxed.group(1).strip()
    out = out.strip().strip("` ")
    if len(out) > 1 and out[-1] in ".;":
        out = out[:-1].strip()
    out = normalize_latex_units(out)
    return out


def normalize_answer(raw: str) -> str:
    out = strip_math_wrappers(raw)
    out = re.sub(r"\s+", " ", out).strip()
    # Normalize spacing in units while keeping symbolic expressions compact.
    out = re.sub(rf"^({NUM_PATTERN})\s*({UNIT_PATTERN})$", r"\1 \2", out)
    out = re.sub(r"\s*([=<>])\s*", r"\1", out)
    out = re.sub(r"\s*,\s*", ",", out)
    return out


def balanced_basic(text: str) -> bool:
    pairs = [("(", ")"), ("[", "]"), ("{", "}")]
    for left, right in pairs:
        if text.count(left) != text.count(right):
            return False
    return True


def is_pure_symbol(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if compact in PURE_SYMBOLS:
        return True
    return not re.search(r"[A-Za-z0-9]", compact)


def classify_answer(answer: str) -> Tuple[bool, str, str]:
    ans = normalize_answer(answer)
    compact = ans.replace(" ", "")

    if not ans:
        return False, "empty", "empty_answer"
    if ans in PURE_SYMBOLS or is_pure_symbol(ans):
        return False, "pure_symbol", "pure_symbol"
    if BAD_CHARS_RE.search(ans):
        return False, "bad_chars", "bad_chars"
    if not balanced_basic(ans):
        return False, "unbalanced", "unbalanced_brackets"
    if "$$" in ans or ans in {"$", "$$"}:
        return False, "math_delimiter", "math_delimiter_only"
    if len(ans) > 120:
        return False, "too_long", "too_long"

    # option letter
    if re.fullmatch(r"[A-E]", ans.strip(), flags=re.I):
        return True, "option", ans.strip().upper()

    # numeric
    if re.fullmatch(NUM_PATTERN, compact):
        return True, "number", ans

    # number with unit
    m_unit = re.fullmatch(rf"({NUM_PATTERN})\s*({UNIT_PATTERN})", ans)
    if m_unit:
        return True, "unit_decimal", f"{m_unit.group(1)} {m_unit.group(2)}"

    # interval with valid endpoint order
    m_int = re.fullmatch(rf"([\(\[])(?:\s*)({NUM_PATTERN})(?:\s*),(?:\s*)({NUM_PATTERN})(?:\s*)([\)\]])", ans)
    if m_int:
        left = float(m_int.group(2))
        right = float(m_int.group(3))
        if left < right:
            return True, "interval", f"{m_int.group(1)}{m_int.group(2)},{m_int.group(3)}{m_int.group(4)}"
        return False, "interval", "interval_endpoint_order"

    # inequality
    if re.fullmatch(rf"{VAR_PATTERN}(?:<=|>=|<|>){NUM_PATTERN}", compact):
        return True, "inequality", compact

    # equation
    if compact.count("=") == 1:
        left, right = compact.split("=", 1)
        if re.fullmatch(VAR_PATTERN, left) and right and not right.startswith("="):
            if re.fullmatch(r"[A-Za-z0-9_+\-*/^().]+", right):
                return True, "equation", compact
        return False, "equation", "invalid_equation"

    # symbolic expression: must contain some math structure and legal chars
    if re.fullmatch(r"[A-Za-z0-9_+\-*/^().]+", compact):
        has_math_structure = any(x in compact for x in ["sqrt", "^", "/", "*", "+", "-"])
        if has_math_structure:
            return True, "symbolic_expression", compact

    return False, "unsupported", "unsupported_answer_form"


def candidate_answers_from_text(text: str) -> List[str]:
    source = normalize_latex_units(text)
    candidates: List[str] = []

    patterns = [
        rf"[\(\[]\s*{NUM_PATTERN}\s*,\s*{NUM_PATTERN}\s*[\)\]]",
        rf"{VAR_PATTERN}\s*(?:<=|>=|<|>)\s*{NUM_PATTERN}",
        rf"{VAR_PATTERN}\s*=\s*[A-Za-z0-9_+\-*/^().]+",
        rf"{NUM_PATTERN}\s*{UNIT_PATTERN}",
        rf"{NUM_PATTERN}",
        r"[A-E]",
        r"[A-Za-z0-9_+\-*/^().]+",
    ]
    for pat in patterns:
        for m in re.finditer(pat, source):
            cand = normalize_answer(m.group(0))
            ok, _, norm = classify_answer(cand)
            if ok and norm not in candidates:
                candidates.append(norm)
    return candidates


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    matches = list(FINAL_MARKER_RE.finditer(text))
    for match in reversed(matches):
        tail = text[match.end(): match.end() + 260]
        tail = tail.split("\n\n", 1)[0]
        # Stop at a likely sentence boundary, but do not break decimal numbers.
        tail = re.split(r"(?<!\d)[.;](?!\d)|\n", tail, maxsplit=1)[0]
        candidates = candidate_answers_from_text(tail)
        if candidates:
            return candidates[-1]
    return ""


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", "", normalize_answer(text).lower())


def reasoning_conflicts(reasoning: str, answer: str) -> bool:
    """Conservative mismatch check.

    Only drop when reasoning has an explicit final-answer marker and the extracted final answer is valid
    but different from answer. Plain containment mismatch is too noisy for QRA and would over-drop.
    """
    extracted = extract_final_answer(reasoning)
    if not extracted:
        return False
    return normalize_for_match(extracted) != normalize_for_match(answer)


# ----------------------------- row transform -----------------------------

def make_segments(question: str, reasoning: str, answer: str) -> Dict[str, Any]:
    return {
        "question": {"text": f"Question:\n{question.strip()}\n\n", "train": False},
        "reasoning": {"text": f"Reasoning:\n{reasoning.strip()}\n", "train": False},
        "answer_anchor": {"text": "\nAnswer:\n", "train": False},
        "answer": {"text": str(answer).strip(), "train": True},
    }


def update_qra_row(row: Dict[str, Any], split: str, idx: int, answer: str, kind: str) -> Dict[str, Any]:
    new = copy.deepcopy(row)
    q = get_question(row)
    r = get_reasoning(row)
    if not q:
        q = str(row.get("text", "")).strip()
    if not r:
        r = "First, we solve the problem and derive the final result."

    new["id"] = str(row.get("id") or row.get("uid") or f"qra_{split}_{idx:06d}")
    new["source"] = "filtered_qra"
    new["answer"] = answer
    new["solution"] = answer
    new["answer_kind"] = kind
    new["question"] = q
    new["reasoning"] = r
    new["segments"] = make_segments(q, r, answer)
    return new


# ----------------------------- synthetic generation -----------------------------

def final_sentence(rng: random.Random, answer: str) -> str:
    return rng.choice(FINAL_SENTENCES).format(answer=answer)


def synth_interval(rng: random.Random) -> Tuple[str, str, str, str]:
    a = rng.randint(-9, 5)
    b = rng.randint(a + 1, a + 12)
    left = rng.choice(["(", "["])
    right = rng.choice([")", "]"])
    ans = f"{left}{a},{b}{right}"
    q_templates = [
        f"Write the interval for all x such that {a} {'<' if left == '(' else '<='} x {'<' if right == ')' else '<='} {b}.",
        f"Convert the boundary condition from {a} to {b} into interval notation.",
        f"Give the solution interval with {'open' if left == '(' else 'closed'} left endpoint {a} and {'open' if right == ')' else 'closed'} right endpoint {b}.",
    ]
    r_templates = [
        f"To solve, identify the lower and upper boundaries. The lower endpoint is {'excluded' if left == '(' else 'included'} and the upper endpoint is {'excluded' if right == ')' else 'included'}.",
        f"First, read the inequality endpoints. Then choose {'(' if left == '(' else '['} for the left boundary and {')' if right == ')' else ']'} for the right boundary.",
        f"We keep the numbers in increasing order and encode the endpoint inclusion with brackets.",
    ]
    return rng.choice(q_templates), rng.choice(r_templates), ans, "interval"


def synth_inequality(rng: random.Random) -> Tuple[str, str, str, str]:
    var = rng.choice(["x", "y", "t", "r"])
    op = rng.choice([">", ">=", "<", "<="])
    c = rng.randint(-8, 12)
    ans = f"{var}{op}{c}"
    q = rng.choice([
        f"Write the final inequality for {var} with boundary {c} and relation {op}.",
        f"Express the solution as a one-sided inequality using variable {var}.",
        f"If the solution set is all {var} satisfying {var} {op} {c}, give the short answer.",
    ])
    r = rng.choice([
        f"First, isolate {var} on one side. The boundary value is {c} and the direction is {op}.",
        f"To solve, keep the variable on the left and preserve the inequality direction.",
        f"The result should be written as a compact inequality with no extra explanation.",
    ])
    return q, r, ans, "inequality"


def synth_equation(rng: random.Random) -> Tuple[str, str, str, str]:
    templates = []
    x = rng.randint(-8, 12)
    a = rng.choice([2, 3, 4, 5])
    b = rng.randint(-6, 8)
    c = a * x + b
    templates.append((
        f"Solve {a}x + {b} = {c}.",
        f"First, subtract {b} from both sides to get {a}x={a*x}. Then divide by {a}.",
        f"x={x}",
    ))
    m = rng.choice([-3, -2, -1, 1, 2, 3, 4])
    intercept = rng.randint(-8, 8)
    templates.append((
        f"Write the line with slope {m} and intercept {intercept}.",
        f"Use y=mx+b. Here the slope is {m} and the intercept is {intercept}.",
        f"y={m}x{intercept:+d}",
    ))
    templates += [
        ("Write Newton's second law solved for acceleration.", "Start from F=ma and divide both sides by m.", "a=F/m"),
        ("Write distance as speed times time.", "Distance equals speed multiplied by time.", "d=v*t"),
        ("Write the area formula for a circle using pi and r.", "The area is pi times the square of the radius.", "A=pi*r^2"),
    ]
    q, r, ans = rng.choice(templates)
    return q, r, ans, "equation"


def simplify_fraction(num: int, den: int) -> Tuple[int, int]:
    g = math.gcd(abs(num), abs(den))
    return num // g, den // g


def synth_symbolic(rng: random.Random) -> Tuple[str, str, str, str]:
    choice = rng.choice(["sqrt", "square", "fraction", "physics", "factor"])
    if choice == "sqrt":
        n = rng.choice([2, 3, 5, 6, 7])
        k = rng.randint(2, 9)
        q = f"Simplify sqrt({k*k*n})."
        r = f"First, write {k*k*n} as {k*k} times {n}. Then sqrt({k*k})={k}."
        ans = f"{k}sqrt({n})"
    elif choice == "square":
        c = rng.randint(1, 9)
        q = f"Expand (x+{c})^2."
        r = f"Use (a+b)^2=a^2+2ab+b^2. Here 2 times {c} is {2*c}."
        ans = f"x^2+{2*c}x+{c*c}"
    elif choice == "fraction":
        den = rng.choice([6, 8, 10, 12, 14, 16, 18, 20])
        num = rng.randint(2, den - 1)
        sn, sd = simplify_fraction(num, den)
        q = f"Simplify {num}/{den}."
        r = f"Divide the numerator and denominator by their greatest common divisor."
        ans = f"{sn}/{sd}"
    elif choice == "physics":
        q = "Write the constant-acceleration relation for final speed squared."
        r = "The relation uses initial speed, acceleration, and displacement."
        ans = "v^2=u^2+2as"
    else:
        c = rng.randint(1, 9)
        q = f"Factor x^2+{2*c}x+{c*c}."
        r = "Recognize the perfect-square trinomial pattern."
        ans = f"(x+{c})^2"
    return q, r, ans, "symbolic_expression"


def synth_unit_decimal(rng: random.Random) -> Tuple[str, str, str, str]:
    unit = rng.choice(["m", "mm", "cm", "kg", "g", "s", "ms", "N", "J", "m/s", "m/s^2"])
    val = rng.choice([round(rng.uniform(-9.9, 9.9), 2), round(rng.uniform(0.1, 5.0), 3), -9.8, 1.903, -2.50, 0.125])
    if unit in {"m", "mm", "m/s^2"} and rng.random() < 0.3:
        val = rng.choice([1.903, -2.50, -9.8, 0.125])
    ans = f"{val:g} {unit}"
    q = rng.choice([
        f"Give the measured value {val:g} with unit {unit} as the final answer.",
        f"Write the numerical result with its unit {unit}.",
        f"The final computed quantity has value {val:g} in {unit}. Provide the short answer.",
    ])
    r = rng.choice([
        "First, compute the numeric value and then attach the requested unit.",
        "To solve, keep the sign and decimal precision, then append the unit.",
        "The result should contain the number followed by the unit.",
    ])
    return q, r, ans, "unit_decimal"


def generate_synthetic_cases(count: int, seed: int, seen_answers: set[str]) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    generators = [synth_interval, synth_inequality, synth_equation, synth_symbolic, synth_unit_decimal]
    out: List[Dict[str, str]] = []
    attempts = 0
    while len(out) < count and attempts < count * 100:
        attempts += 1
        gen = generators[attempts % len(generators)] if attempts <= len(generators) * 2 else rng.choice(generators)
        q, r, ans, kind = gen(rng)
        ok, _, norm = classify_answer(ans)
        if not ok:
            continue
        key = normalize_for_match(norm)
        if key in seen_answers:
            continue
        seen_answers.add(key)
        reasoning = r.rstrip() + " " + final_sentence(rng, norm)
        out.append({"question": q, "reasoning": reasoning, "answer": norm, "kind": kind})
    if len(out) < count:
        raise RuntimeError(f"Could only generate {len(out)} synthetic cases out of requested {count}.")
    return out


def make_synthetic_sample(split: str, idx: int, case: Dict[str, str]) -> Dict[str, Any]:
    answer = case["answer"]
    question = case["question"]
    reasoning = case["reasoning"]
    return {
        "id": f"s1f_rl_synth_{split}_{idx:06d}",
        "source": "synthetic_hard_case",
        "answer_kind": case.get("kind", "synthetic"),
        "question": question,
        "reasoning": reasoning,
        "answer": answer,
        "solution": answer,
        "segments": make_segments(question, reasoning, answer),
    }


# ----------------------------- dataset build -----------------------------

def filter_qra_split(
    rows: List[Dict[str, Any]],
    split: str,
    strict_reasoning_match: bool,
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
        raw_answer = get_answer_text(row)
        reasoning = get_reasoning(row)

        source = "qra_answer_segment"
        answer = normalize_answer(raw_answer)

        ok, kind, normalized_or_reason = classify_answer(answer)
        if not ok:
            # If the train=True/answer segment is actually a full completion, extract the last final-answer marker from it.
            extracted = extract_final_answer(raw_answer)
            if extracted:
                ok2, kind2, normalized2 = classify_answer(extracted)
                if ok2:
                    ok, kind, normalized_or_reason = True, kind2, normalized2
                    source = "answer_segment_final_marker"

        if not ok:
            extracted = extract_final_answer(reasoning)
            if extracted:
                ok2, kind2, normalized2 = classify_answer(extracted)
                if ok2:
                    ok, kind, normalized_or_reason = True, kind2, normalized2
                    source = "reasoning_final_marker"

        if not ok:
            key = f"drop_{normalized_or_reason}"
            stats[key] = stats.get(key, 0) + 1
            if len(stats["drop_examples"]) < max_drop_examples:
                stats["drop_examples"].append({"idx": idx, "reason": key, "raw_answer": raw_answer[:200]})
            continue

        answer = normalized_or_reason
        if reasoning_conflicts(reasoning, answer):
            key = "drop_reasoning_final_marker_conflict"
            stats[key] = stats.get(key, 0) + 1
            if len(stats["drop_examples"]) < max_drop_examples:
                stats["drop_examples"].append({"idx": idx, "reason": key, "answer": answer, "reasoning_tail": reasoning[-240:]})
            continue

        if strict_reasoning_match and normalize_for_match(answer) not in normalize_for_match(reasoning):
            key = "drop_strict_reasoning_mismatch"
            stats[key] = stats.get(key, 0) + 1
            if len(stats["drop_examples"]) < max_drop_examples:
                stats["drop_examples"].append({"idx": idx, "reason": key, "answer": answer})
            continue

        out.append(update_qra_row(row, split, idx, answer, kind))
        stats["real_kept"] += 1
        stats[f"kept_{kind}"] = stats.get(f"kept_{kind}", 0) + 1
        stats[f"kept_from_{source}"] = stats.get(f"kept_from_{source}", 0) + 1
        seen_answers.add(normalize_for_match(answer))

    return out, stats, seen_answers


def compute_synth_quotas(real_total: int, synthetic_final_fraction: float, min_eval: int) -> Dict[str, int]:
    fraction = max(0.0, min(0.7, synthetic_final_fraction))
    if real_total <= 0:
        k = int(min_eval)
        return {"train": 8 * k, "val": k, "test": k}
    target_total = int(round(real_total * fraction / max(1e-8, 1.0 - fraction)))
    k = max(int(min_eval), int(math.ceil(target_total / 10.0)))
    return {"train": 8 * k, "val": k, "test": k}


def build_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = repo_dir / input_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_dir / output_dir

    loaded: Dict[str, Tuple[List[Dict[str, Any]], Optional[Path]]] = {
        split: load_split(input_dir, split) for split in ["train", "val", "test"]
    }
    raw_total = sum(len(rows) for rows, _ in loaded.values())
    if raw_total == 0:
        searched = {split: [str(p) for p in split_candidates(input_dir, split)] for split in ["train", "val", "test"]}
        raise RuntimeError(
            "No QRA data was loaded. This script refuses to generate synthetic-only data.\n"
            f"input_dir={input_dir}\nsearched={json.dumps(searched, ensure_ascii=False, indent=2)}\n"
            "Pass --input-dir to the real QRA directory, for example --input-dir rl_qra_pipeline/data/QRA."
        )

    report: Dict[str, Any] = {
        "base": "copy_filtered_QRA_plus_synthetic_hard_cases_v8_anchor_fixed",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "raw_total_loaded": raw_total,
        "strict_reasoning_match": bool(args.strict_reasoning_match),
        "synthetic_final_fraction_target": args.synthetic_final_fraction,
        "min_synthetic_eval": args.min_synthetic_eval,
        "synthetic_split_rule": "8:1:1",
        "splits": {},
    }

    split_rows: Dict[str, List[Dict[str, Any]]] = {}
    global_seen: set[str] = set()

    for split in ["train", "val", "test"]:
        rows, path = loaded[split]
        real_rows, stats, seen = filter_qra_split(
            rows,
            split=split,
            strict_reasoning_match=bool(args.strict_reasoning_match),
            max_drop_examples=int(args.max_drop_examples),
        )
        split_rows[split] = real_rows
        global_seen.update(seen)
        report["splits"][split] = {**stats, "input_file": str(path) if path else None}

    real_total = sum(len(rows) for rows in split_rows.values())
    if real_total == 0 and not args.allow_zero_real:
        output_dir.mkdir(parents=True, exist_ok=True)
        failed_report = output_dir / "build_report_failed_zero_real.json"
        report["real_total_kept"] = 0
        failed_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(
            "QRA files were loaded, but 0 real rows survived filtering. Refusing to output synthetic-only data.\n"
            f"A debug report was written to {failed_report}.\n"
            "Most likely causes: answer_anchor was selected instead of answer, answer segment contains full completion, "
            "or filters are still too strict."
        )

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
    synth_total = sum(report["splits"][s]["synthetic_added"] for s in ["train", "val", "test"])
    report["total_written"] = total_written
    report["final_split_counts"] = {split: len(split_rows[split]) for split in ["train", "val", "test"]}
    report["final_split_ratio"] = {split: len(split_rows[split]) / total_written for split in ["train", "val", "test"]} if total_written else {}
    report["synthetic_fraction_overall"] = synth_total / total_written if total_written else 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in split_rows.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)
    (output_dir / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        default="rl_qra_pipeline/data/QRA",
        help="Input QRA data directory. Default: rl_qra_pipeline/data/QRA.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rl_qra_pipeline/data/S1F_RL",
        help="Output directory. Default: rl_qra_pipeline/data/S1F_RL.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--synthetic-final-fraction",
        type=float,
        default=0.30,
        help="Target synthetic fraction in final dataset. Default 0.30.",
    )
    parser.add_argument(
        "--min-synthetic-eval",
        type=int,
        default=5,
        help="Minimum synthetic cases for val and test. Train gets 8x this number. Default 5.",
    )
    parser.add_argument(
        "--strict-reasoning-match",
        action="store_true",
        help="Drop rows whose normalized answer is not literally found in reasoning. Off by default because it over-drops QRA.",
    )
    parser.add_argument("--allow-zero-real", action="store_true")
    parser.add_argument("--max-drop-examples", type=int, default=30)
    args = parser.parse_args()

    report = build_dataset(args)

    print(f"Wrote {report['output_dir']}")
    print(f"Loaded raw QRA rows: {report['raw_total_loaded']}")
    print(f"Kept real QRA rows: {report['real_total_kept']}")
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
    print(f"synthetic_fraction_overall: {report.get('synthetic_fraction_overall', 0.0):.3f}")


if __name__ == "__main__":
    main()
