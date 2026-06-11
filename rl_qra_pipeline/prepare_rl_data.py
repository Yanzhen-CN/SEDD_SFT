from __future__ import annotations

import argparse
import copy
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


QRA_SEGMENT_ORDER = [
    "user_label",
    "user",
    "assistant_label",
    "reasoning_label",
    "reasoning",
    "answer_label",
    "answer",
]

SPLIT_ALIASES = {
    "train": ["train"],
    "val": ["val", "valid", "validation"],
    "test": ["test"],
}

BAD_MOJIBAKE = ["�", "Ã", "Â", "â€", "â€™", "â€œ", "â€�", "ï¼"]
BAD_PUNCT = set("》《【】『』「」")
PURE_SYMBOL_SET = {
    "", "$", "$$", r"\(", r"\)", r"\[", r"\]",
    "[", "]", "(", ")", "{", "}", ".", ",", ":", ";",
    "=", "-", "+", "*", "/", "\\", "|", "<", ">", "_", "^",
}

FULLWIDTH_TRANS = str.maketrans({
    "−": "-", "–": "-", "—": "-",
    "，": ",", "。": ".", "：": ":", "；": ";",
    "（": "(", "）": ")", "［": "[", "］": "]",
})

VALID_UNITS = [
    "m/s^2", "m/s", "mm", "cm", "km", "kg", "ms", "Hz",
    "m", "g", "s", "N", "J", "W", "V", "A",
]
UNIT_PATTERN = "(?:" + "|".join(re.escape(u) for u in sorted(VALID_UNITS, key=len, reverse=True)) + ")"
NUM_PATTERN = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"

FINAL_SENTENCES = [
    "Therefore, the final answer is {answer}.",
    "Thus, the final solution is {answer}.",
    "So the solution is {answer}.",
    "The answer should be {answer}.",
    "Finally, the answer is {answer}.",
    "Hence, the answer is {answer}.",
    "This gives the final answer as {answer}.",
]


def clean(text: Any) -> str:
    return str(text or "").strip()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_split(input_dir: Path, split: str) -> Tuple[List[Dict[str, Any]], Optional[Path]]:
    for name in SPLIT_ALIASES[split]:
        for path in [
            input_dir / f"{name}.jsonl",
            input_dir / f"{name}.json",
            input_dir / name / "data.jsonl",
            input_dir / name / "data.json",
        ]:
            if path.suffix == ".jsonl":
                rows = read_jsonl(path)
            elif path.exists():
                obj = json.loads(path.read_text(encoding="utf-8"))
                rows = obj if isinstance(obj, list) else obj.get("data", []) if isinstance(obj, dict) else []
                rows = [r for r in rows if isinstance(r, dict)]
            else:
                rows = []
            if rows:
                return rows, path
    return [], None


# ----------------------------- segment parsing -----------------------------

def get_seg(row: Dict[str, Any], key: str) -> str:
    segs = row.get("segments")
    if isinstance(segs, dict):
        value = segs.get(key)
        if isinstance(value, dict):
            return clean(value.get("text", ""))
        return clean(value)
    if isinstance(segs, list):
        for item in segs:
            if not isinstance(item, dict):
                continue
            name = clean(item.get("name") or item.get("key") or item.get("type")).lower()
            if name == key.lower():
                return clean(item.get("text", ""))
    return ""


def get_train_segments(row: Dict[str, Any]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    segs = row.get("segments")
    if isinstance(segs, dict):
        for key, value in segs.items():
            if isinstance(value, dict) and value.get("train") is True:
                text = clean(value.get("text", ""))
                if text:
                    out.append((str(key), text))
    elif isinstance(segs, list):
        for item in segs:
            if not isinstance(item, dict):
                continue
            if item.get("train") is True:
                name = clean(item.get("name") or item.get("key") or item.get("type"))
                text = clean(item.get("text", ""))
                if text:
                    out.append((name, text))
    return out


def extract_question(row: Dict[str, Any]) -> str:
    for key in ["question", "user", "prompt", "input", "problem"]:
        value = clean(row.get(key)) or get_seg(row, key)
        if value:
            value = re.sub(r"(?is)^\s*(?:User|Question|Problem|Prompt)\s*[:：]?\s*", "", value).strip()
            return value
    return ""


def extract_reasoning(row: Dict[str, Any]) -> str:
    for key in ["reasoning", "rationale", "thinking", "cot", "chain_of_thought"]:
        value = clean(row.get(key)) or get_seg(row, key)
        if value:
            value = re.sub(r"(?is)^\s*(?:Reasoning|Rationale|Thinking|CoT)\s*[:：]?\s*", "", value).strip()
            return value
    return ""


def extract_answer_raw(row: Dict[str, Any]) -> Tuple[str, str]:
    """Match the real QRA adapter first.

    Root QRA samples from sft_reasoning_pipeline/prepare_reasoning_data.py store:
      row["answer"] and segments["answer"]["text"] with train=True.
    Older QA samples may store answer in segments["assistant"] with train=True.
    """
    if clean(row.get("answer")):
        return clean(row.get("answer")), "top_answer"
    if clean(row.get("solution")):
        return clean(row.get("solution")), "top_solution"

    ans = get_seg(row, "answer")
    if ans:
        return ans, "segments.answer"

    # Older answer-only data uses assistant as the train=True target.
    assistant = get_seg(row, "assistant")
    if assistant and any(k.lower() == "assistant" for k, _ in get_train_segments(row)):
        return assistant, "segments.assistant_train"

    train_segs = get_train_segments(row)
    preferred = ["answer", "assistant", "completion", "target", "output"]
    for pref in preferred:
        for key, text in train_segs:
            if key.lower() == pref:
                return text, f"train_segment.{key}"

    if len(train_segs) == 1:
        key, text = train_segs[0]
        return text, f"single_train_segment.{key}"

    return "", "missing_answer"


# ----------------------------- answer filter -----------------------------

def normalize_latex_units(text: str) -> str:
    out = clean(text).translate(FULLWIDTH_TRANS)
    out = re.sub(r"\\(?:mathrm|text)\s*\{\s*~?\s*([^{}]+?)\s*\}", lambda m: " " + m.group(1).strip(), out)
    out = out.replace("\\,", " ").replace("\\;", " ").replace("~", " ")
    out = out.replace("\\cdot", "*").replace("\\times", "*")
    out = out.replace("\\pi", "pi").replace("π", "pi")
    out = re.sub(r"\\sqrt\s*\{\s*([^{}]+?)\s*\}", r"sqrt(\1)", out)
    out = re.sub(r"\\frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}", r"\1/\2", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def strip_answer_wrappers(raw: str) -> str:
    text = clean(raw)
    text = re.sub(r"(?is)^\s*(?:answer|solution|final answer|final solution|the answer|the solution|our answer)\s*(?:is|=|:|：)?\s*", "", text).strip()
    boxed = re.fullmatch(r"\\boxed\{(.+)\}", text, flags=re.S)
    if boxed:
        text = boxed.group(1).strip()
    for left, right in [("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")]:
        if text.startswith(left) and text.endswith(right) and len(text) > len(left) + len(right):
            text = text[len(left):-len(right)].strip()
    text = text.strip().strip("` '")
    if len(text) > 1 and text.endswith((".", ";", "。")) and not re.search(r"\d\.\d$", text):
        text = text[:-1].strip()
    return normalize_latex_units(text)


def balanced_brackets(text: str) -> bool:
    stack: List[str] = []
    pair = {')': '(', ']': '[', '}': '{'}
    opens = set(pair.values())
    for ch in text:
        if ch in opens:
            stack.append(ch)
        elif ch in pair:
            if not stack or stack[-1] != pair[ch]:
                return False
            stack.pop()
    return not stack


def is_interval_answer(text: str) -> bool:
    compact = re.sub(r"\s+", "", clean(text))
    return re.fullmatch(r"[\[\(]" + NUM_PATTERN + r"," + NUM_PATTERN + r"[\]\)]", compact) is not None


def pure_symbol(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if compact in PURE_SYMBOL_SET:
        return True
    return not re.search(r"[A-Za-z0-9]", compact)


def symbol_noise_ratio(text: str) -> float:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return 1.0
    allowed = set("._,;:!?+-*/=()[]{}<>\\/$%°'\"`~^&|#@")
    noisy = 0
    for ch in compact:
        if ch.isalnum() or ch in allowed:
            continue
        noisy += 1
    return noisy / max(1, len(compact))


def validate_answer(raw_answer: str, max_chars: int = 160) -> Tuple[bool, str, str, str]:
    """Conservative *drop bad only* filter.

    Do not whitelist only math forms. Keep normal short text answers from QRA.
    Drop only clearly broken targets: pure symbols, mojibake, malformed brackets,
    display-math leftovers, very long formula-looking strings, and symbol-noisy strings.
    """
    raw = clean(raw_answer)
    if not raw:
        return False, "", "empty", "empty_answer"

    if any(bad in raw for bad in BAD_MOJIBAKE):
        return False, "", "mojibake", "mojibake"
    if any(ch in raw for ch in BAD_PUNCT):
        return False, "", "bad_punctuation", "bad_punctuation"
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", raw):
        return False, "", "control_char", "control_char"

    # Multi-line display math is almost always a long derivation answer, not a target answer.
    if "\n" in raw and ("$$" in raw or r"\[" in raw or r"\begin" in raw):
        return False, "", "multiline_formula", "multiline_formula"

    ans = strip_answer_wrappers(raw)
    ans = re.sub(r"\s+", " ", ans).strip()
    ans = re.sub(r"\s*([=<>])\s*", r"\1", ans)
    ans = re.sub(r"\s*,\s*", ",", ans)
    ans = re.sub(rf"^({NUM_PATTERN})\s*({UNIT_PATTERN})$", r"\1 \2", ans)

    if not ans:
        return False, "", "empty_after_clean", "empty_after_clean"
    if ans in PURE_SYMBOL_SET or pure_symbol(ans):
        return False, "", "pure_symbol", "pure_symbol"
    if "$$" in ans or ans in {"$", "$$"}:
        return False, "", "math_delimiter", "math_delimiter_only"
    if not balanced_brackets(ans) and not is_interval_answer(ans):
        return False, "", "unbalanced_brackets", "unbalanced_brackets"
    if len(ans) > max_chars:
        return False, "", "too_long", "too_long"

    # Very long LaTeX-like formula remnants are usually bad for this RL short-answer set.
    latex_like = any(x in ans for x in ["\\frac", "\\sqrt", "\\left", "\\right", "\\sin", "\\cos", "\\begin"])
    if latex_like and len(ans) > 80:
        return False, "", "long_latex", "long_latex"

    if len(re.sub(r"\s+", "", ans)) >= 8 and symbol_noise_ratio(ans) > 0.25:
        return False, "", "symbol_noise", "symbol_noise"

    # Catch common broken strings like 3)5>, x=, =3, x==3.
    compact = ans.replace(" ", "")
    if re.search(r"\d[\)\]>]+\d", compact):
        return False, "", "malformed_symbol_sequence", "malformed_symbol_sequence"
    if compact.startswith("=") or compact.endswith("=") or "==" in compact:
        return False, "", "bad_equation", "bad_equation"

    # Normalize option letters.
    if re.fullmatch(r"\(?[A-Ea-e]\)?", ans):
        ans = ans.strip("()").upper()


    return True, ans, "kept", "kept"


# ----------------------------- answer type / mini validation -----------------------------

MINI_EVAL_TYPE_PRIORITY = [
    "single_integer",
    "decimal",
    "unit_decimal",
    "interval",
    "inequality",
    "equation",
    "fraction",
    "symbolic",
    "boolean",
    "short_text",
    "other",
]


def infer_answer_type(answer: Any) -> str:
    """Infer a compact answer type for stratified mini validation selection.

    This is intentionally heuristic: it is used only to choose a small, fixed
    val-mini subset with broad answer-form coverage. Full validation still
    uses the complete val split.
    """
    ans = strip_answer_wrappers(clean(answer))
    compact = re.sub(r"\s+", "", ans)
    lower = compact.lower()
    if not compact:
        return "other"
    if lower in {"true", "false", "yes", "no"}:
        return "boolean"
    if is_interval_answer(compact):
        return "interval"
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:<=|>=|<|>)" + NUM_PATTERN, compact):
        return "inequality"
    if "=" in compact:
        return "equation"
    if re.fullmatch(r"[+-]?\d+/[1-9]\d*", compact):
        return "fraction"
    if re.fullmatch(r"[-+]?\d+", compact):
        return "single_integer"
    if re.fullmatch(r"[-+]?(?:\d+\.\d+|\.\d+)", compact):
        return "decimal"
    if re.fullmatch(rf"{NUM_PATTERN}{UNIT_PATTERN}", compact) or re.fullmatch(rf"{NUM_PATTERN}{UNIT_PATTERN.replace('\\\\', '\\\\')}", compact):
        return "unit_decimal"
    if re.search(r"\d", compact) and re.search(r"[A-Za-z]", compact):
        # e.g. 3.2 m, 5kg, 2sqrt(3), x^2+4x+4
        if re.search(UNIT_PATTERN, compact):
            return "unit_decimal"
        return "symbolic"
    if any(x in compact for x in ["sqrt", "^", "pi", "*", "\\frac"]):
        return "symbolic"
    return "short_text"


def sample_id(row: Dict[str, Any], fallback: int) -> str:
    return clean(row.get("id") or row.get("sample_id") or f"row_{fallback:06d}")


def select_val_mini(rows: List[Dict[str, Any]], size: int, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select a fixed, stratified mini-validation subset from val rows.

    The output is deterministic for a given seed and is written as
    val-mini.jsonl.  Training can then use val-mini for cheap
    every-step eval while full validation still uses val.jsonl.
    """
    size = max(0, int(size))
    if size <= 0 or not rows:
        return [], {
            "strategy": "disabled" if size <= 0 else "empty_val",
            "requested_size": size,
            "selected_size": 0,
            "selected": [],
            "type_counts": {},
        }

    rng = random.Random(seed)
    buckets: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    type_counts: Dict[str, int] = {}
    for idx, row in enumerate(rows):
        t = clean(row.get("answer_type")) or infer_answer_type(row.get("answer", row.get("solution", "")))
        row["answer_type"] = t
        buckets.setdefault(t, []).append((idx, row))
        type_counts[t] = type_counts.get(t, 0) + 1
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected: List[Tuple[int, Dict[str, Any]]] = []
    selected_ids: set[str] = set()

    def add_candidate(pair: Tuple[int, Dict[str, Any]]) -> bool:
        idx, row = pair
        sid = sample_id(row, idx)
        if sid in selected_ids:
            return False
        selected.append(pair)
        selected_ids.add(sid)
        return True

    # First pass: choose at most one sample from each preferred type.
    for t in MINI_EVAL_TYPE_PRIORITY:
        if len(selected) >= size:
            break
        bucket = buckets.get(t, [])
        if bucket:
            add_candidate(bucket[0])

    # Second pass: fill remaining slots from all rows, shuffled deterministically.
    remaining: List[Tuple[int, Dict[str, Any]]] = []
    for bucket in buckets.values():
        remaining.extend(bucket)
    rng.shuffle(remaining)
    for pair in remaining:
        if len(selected) >= size:
            break
        add_candidate(pair)

    # Preserve val order in the written mini file for easier inspection.
    selected = sorted(selected, key=lambda x: x[0])
    mini_rows = [row for _, row in selected]
    manifest_rows = []
    for idx, row in selected:
        manifest_rows.append({
            "source_val_index": idx,
            "id": sample_id(row, idx),
            "answer_type": row.get("answer_type", infer_answer_type(row.get("answer", row.get("solution", "")))),
            "answer": row.get("answer", row.get("solution", "")),
            "source": row.get("source", ""),
            "question_preview": clean(row.get("question"))[:160],
        })
    return mini_rows, {
        "strategy": "stratified_answer_type",
        "requested_size": size,
        "selected_size": len(mini_rows),
        "type_priority": MINI_EVAL_TYPE_PRIORITY,
        "type_counts": type_counts,
        "selected_type_counts": {r["answer_type"]: sum(1 for x in manifest_rows if x["answer_type"] == r["answer_type"]) for r in manifest_rows},
        "selected": manifest_rows,
    }


# ----------------------------- output row construction -----------------------------

def make_segments(question: str, reasoning: str, answer: str) -> Dict[str, Dict[str, Any]]:
    return {
        "user_label": {"text": "User: ", "train": False},
        "user": {"text": clean(question), "train": False},
        "assistant_label": {"text": "\nAssistant:\n", "train": False},
        "reasoning_label": {"text": "Reasoning:\n", "train": False},
        "reasoning": {"text": clean(reasoning), "train": False},
        "answer_label": {"text": "\n\nAnswer:\n", "train": False},
        "answer": {"text": clean(answer), "train": True},
    }


def make_filtered_qra(row: Dict[str, Any], split: str, idx: int, answer: str, method: str) -> Dict[str, Any]:
    q = extract_question(row)
    r = extract_reasoning(row)
    if not q:
        q = clean(row.get("text")) or f"Question {idx}"
    if not r:
        r = "First, solve the problem using the given information."

    out = copy.deepcopy(row)
    out["id"] = clean(row.get("id")) or f"qra_{split}_{idx:06d}"
    out["mode"] = "QRA"
    out["split"] = split
    out["source"] = "filtered_qra"
    out["answer"] = answer
    out["solution"] = answer
    out["answer_type"] = infer_answer_type(answer)
    out["question"] = q
    out["reasoning"] = r
    out["answer_filter_method"] = method
    out["segment_order"] = QRA_SEGMENT_ORDER
    out["segments"] = make_segments(q, r, answer)
    return out


# ----------------------------- synthetic generation -----------------------------

def final_sentence(rng: random.Random, answer: str) -> str:
    return rng.choice(FINAL_SENTENCES).format(answer=answer)


def synth_interval(rng: random.Random) -> Tuple[str, str, str, str]:
    a = rng.randint(-12, 6)
    b = rng.randint(a + 1, a + 15)
    left = rng.choice(["(", "["])
    right = rng.choice([")", "]"])
    ans = f"{left}{a},{b}{right}"
    q = rng.choice([
        f"Write the interval for all x such that {a} {'<' if left == '(' else '<='} x {'<' if right == ')' else '<='} {b}.",
        f"Convert the boundary condition from {a} to {b} into interval notation.",
        f"Give the solution interval with {'open' if left == '(' else 'closed'} left endpoint {a} and {'open' if right == ')' else 'closed'} right endpoint {b}.",
    ])
    r = rng.choice([
        f"To solve, identify the two endpoints and whether each endpoint is included. The left endpoint is {'excluded' if left == '(' else 'included'} and the right endpoint is {'excluded' if right == ')' else 'included'}.",
        f"First, keep the smaller endpoint on the left and the larger endpoint on the right. Then choose the correct bracket type for each boundary.",
        f"The interval notation is determined by endpoint order and inclusion status.",
    ])
    return q, r + " " + final_sentence(rng, ans), ans, "synthetic_interval"


def synth_inequality(rng: random.Random) -> Tuple[str, str, str, str]:
    var = rng.choice(["x", "y", "t", "r"])
    c = rng.randint(-9, 12)
    op = rng.choice([">", ">=", "<", "<="])
    ans = f"{var}{op}{c}"
    q = rng.choice([
        f"State the final inequality for {var} if the boundary is {c} and the relation is {op}.",
        f"Write the solution as a simple inequality using variable {var}.",
        f"Convert the described condition into symbolic inequality form.",
    ])
    r = rng.choice([
        f"First, isolate {var}. Then keep the comparison direction as {op}.",
        f"We translate the verbal condition directly into an inequality with {var} on the left.",
        f"The threshold is {c}, and the comparison operator is {op}.",
    ])
    return q, r + " " + final_sentence(rng, ans), ans, "synthetic_inequality"


def synth_equation(rng: random.Random) -> Tuple[str, str, str, str]:
    kind = rng.choice(["linear", "line", "formula"])
    if kind == "linear":
        var = rng.choice(["x", "y", "a", "t"])
        val = rng.randint(-12, 18)
        ans = f"{var}={val}"
        q = f"Solve for {var} and give only the final equation."
        r = f"First, isolate {var} on one side of the equation. The computed value is {val}."
    elif kind == "line":
        m = rng.choice([-4, -3, -2, 2, 3, 4, 5])
        b = rng.randint(-9, 9)
        ans = f"y={m}x{b:+d}".replace("+0", "")
        q = f"Write the line with slope {m} and intercept {b}."
        r = f"Use y=mx+b. Here m={m} and b={b}, so substitute them into the formula."
    else:
        ans = rng.choice(["a=F/m", "v=d/t", "F=ma", "A=pi*r^2", "p=mv"])
        q = "Write the requested physical formula in compact symbolic form."
        r = "First, identify the standard relationship and isolate the requested quantity."
    return q, r + " " + final_sentence(rng, ans), ans, "synthetic_equation"


def synth_symbolic(rng: random.Random) -> Tuple[str, str, str, str]:
    kind = rng.choice(["sqrt", "square", "fraction", "physics", "monomial"])
    if kind == "sqrt":
        outside = rng.randint(2, 9)
        inside = rng.choice([2, 3, 5, 6, 7])
        ans = f"{outside}sqrt({inside})"
        q = f"Simplify sqrt({outside * outside * inside})."
        r = f"First, factor out the perfect square {outside * outside}. Then sqrt({outside * outside})={outside}."
    elif kind == "square":
        a = rng.randint(1, 7)
        ans = f"x^2+{2*a}x+{a*a}"
        q = f"Expand (x+{a})^2."
        r = "Use the identity (x+a)^2=x^2+2ax+a^2 and substitute the value of a."
    elif kind == "fraction":
        den = rng.choice([4, 5, 6, 7, 8, 9])
        num = rng.randint(1, den - 1)
        ans = f"{num}/{den}"
        q = f"Give the simplified fraction form with numerator {num} and denominator {den}."
        r = "The fraction is already in the target compact form."
    elif kind == "physics":
        ans = rng.choice(["v^2=u^2+2as", "KE=1/2mv^2", "p=mv"])
        q = "Write the compact symbolic expression for the relation."
        r = "Recall the standard formula and write it without extra explanation."
    else:
        coeff = rng.randint(2, 9)
        power = rng.choice([2, 3])
        ans = f"{coeff}x^{power}"
        q = f"Simplify the product into a compact monomial with coefficient {coeff}."
        r = f"Combine the repeated x factors into x^{power} and keep the coefficient."
    return q, r + " " + final_sentence(rng, ans), ans, "synthetic_symbolic"


def synth_unit_decimal(rng: random.Random) -> Tuple[str, str, str, str]:
    unit = rng.choice(VALID_UNITS)
    value = rng.uniform(-12, 12)
    decimals = rng.choice([1, 2, 3])
    ans = f"{value:.{decimals}f} {unit}"
    if ans.startswith("-0.0"):
        ans = ans.replace("-0.0", "0.0", 1)
    q = rng.choice([
        f"Report the measured value using unit {unit}.",
        f"Write the final numerical value with its unit {unit}.",
        f"Give the signed decimal result together with the unit {unit}.",
    ])
    r = rng.choice([
        "First, keep the sign of the value and preserve the unit.",
        "The final response should contain the decimal number followed by the unit.",
        "We only need the compact numerical result with the measurement unit.",
    ])
    return q, r + " " + final_sentence(rng, ans), ans, "synthetic_unit_decimal"


SYNTH_GENERATORS = [synth_interval, synth_inequality, synth_equation, synth_symbolic, synth_unit_decimal]


def make_synthetic_row(split: str, idx: int, q: str, r: str, ans: str, source: str) -> Dict[str, Any]:
    return {
        "id": f"s1k_rl_{source}_{split}_{idx:06d}",
        "mode": "QRA",
        "split": split,
        "source": source,
        "question": q,
        "reasoning": r,
        "answer": ans,
        "solution": ans,
        "answer_type": infer_answer_type(ans),
        "segment_order": QRA_SEGMENT_ORDER,
        "segments": make_segments(q, r, ans),
    }


def generate_synthetic(split: str, count: int, seed: int, seen_answers: set[str]) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    attempts = 0
    while len(rows) < count and attempts < count * 50 + 1000:
        attempts += 1
        gen = SYNTH_GENERATORS[(len(rows) + attempts) % len(SYNTH_GENERATORS)]
        q, r, ans, source = gen(rng)
        ok, ans_norm, _, reason = validate_answer(ans)
        if not ok:
            continue
        key = re.sub(r"\s+", "", ans_norm.lower())
        if key in seen_answers:
            continue
        seen_answers.add(key)
        rows.append(make_synthetic_row(split, len(rows), q, r, ans_norm, source))
    if len(rows) < count:
        raise RuntimeError(f"Only generated {len(rows)}/{count} synthetic rows for {split}.")
    return rows


def compute_synth_quotas(real_counts: Dict[str, int], final_fraction: float, min_eval: int) -> Dict[str, int]:
    real_total = sum(real_counts.values())
    if real_total <= 0:
        return {"train": 8 * min_eval, "val": min_eval, "test": min_eval}
    final_fraction = max(0.0, min(0.75, final_fraction))
    total_synth = int(round(real_total * final_fraction / max(1e-6, 1.0 - final_fraction)))
    total_synth = max(total_synth, 10 * min_eval)
    val = max(min_eval, int(round(total_synth * 0.1)))
    test = max(min_eval, int(round(total_synth * 0.1)))
    train = max(8 * min_eval, total_synth - val - test)
    return {"train": train, "val": val, "test": test}


# ----------------------------- build -----------------------------

def build_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    loaded: Dict[str, List[Dict[str, Any]]] = {}
    input_files: Dict[str, str] = {}
    for split in ["train", "val", "test"]:
        rows, path = load_split(input_dir, split)
        loaded[split] = rows
        input_files[split] = str(path) if path else ""

    raw_total = sum(len(v) for v in loaded.values())
    if raw_total == 0:
        raise RuntimeError(f"No QRA rows found under {input_dir}. Expected train/validation/test jsonl files.")

    output_splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    stats: Dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_files": input_files,
        "raw_total_loaded": raw_total,
        "split_stats": {},
        "drop_reasons": {},
        "drop_examples": [],
        "kept_examples": [],
    }

    seen_answers: set[str] = set()
    for split, rows in loaded.items():
        kept = 0
        split_drops: Dict[str, int] = {}
        answer_sources: Dict[str, int] = {}
        for idx, row in enumerate(rows):
            raw_answer, source = extract_answer_raw(row)
            answer_sources[source] = answer_sources.get(source, 0) + 1
            ok, answer, _, reason = validate_answer(raw_answer, max_chars=args.max_answer_chars)
            if not ok:
                split_drops[reason] = split_drops.get(reason, 0) + 1
                stats["drop_reasons"][reason] = stats["drop_reasons"].get(reason, 0) + 1
                if len(stats["drop_examples"]) < args.max_drop_examples:
                    stats["drop_examples"].append({
                        "split": split,
                        "id": row.get("id"),
                        "answer_source": source,
                        "reason": reason,
                        "raw_answer_preview": clean(raw_answer)[:240],
                        "segment_keys": list((row.get("segments") or {}).keys()) if isinstance(row.get("segments"), dict) else None,
                        "top_keys": list(row.keys())[:30],
                    })
                continue
            out = make_filtered_qra(row, split, idx, answer, source)
            output_splits[split].append(out)
            kept += 1
            seen_answers.add(re.sub(r"\s+", "", answer.lower()))
            if len(stats["kept_examples"]) < args.max_kept_examples:
                stats["kept_examples"].append({
                    "split": split,
                    "id": out.get("id"),
                    "answer_source": source,
                    "answer": answer,
                    "question_preview": out.get("question", "")[:160],
                })

        stats["split_stats"][split] = {
            "raw": len(rows),
            "real_kept": kept,
            "real_dropped": len(rows) - kept,
            "answer_sources": answer_sources,
            "drop_reasons": split_drops,
        }

    real_counts = {split: len(rows) for split, rows in output_splits.items()}
    real_total_kept = sum(real_counts.values())
    if real_total_kept == 0 and not args.allow_zero_real:
        debug_path = output_dir / "build_report_failed.json"
        write_json(debug_path, stats)
        raise RuntimeError(
            f"QRA files loaded ({raw_total} rows), but 0 real rows survived filtering. "
            f"Wrote debug report to {debug_path}. This usually means answer extraction mismatched the QRA format."
        )

    quotas = compute_synth_quotas(real_counts, args.synthetic_final_fraction, args.min_synthetic_eval)
    stats["real_counts_before_synthetic"] = real_counts
    stats["synthetic_final_fraction_target"] = args.synthetic_final_fraction
    stats["synthetic_quotas"] = quotas

    for split in ["train", "val", "test"]:
        synth = generate_synthetic(split, quotas[split], args.seed + {"train": 101, "val": 202, "test": 303}[split], seen_answers)
        output_splits[split].extend(synth)
        random.Random(args.seed + {"train": 11, "val": 22, "test": 33}[split]).shuffle(output_splits[split])
        stats["split_stats"][split]["synthetic_added"] = len(synth)
        stats["split_stats"][split]["written"] = len(output_splits[split])

    if output_dir.exists() and args.overwrite:
        for name in [
            "train.jsonl", "val.jsonl", "val.json", "valid.jsonl", "validation.jsonl",
            "val-mini.jsonl", "validation-mini.jsonl", "val_mini_manifest.json", "validation_mini_manifest.json",
            "test.jsonl", "build_report.json", "build_report_failed.json"
        ]:
            path = output_dir / name
            if path.exists():
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    val_mini, mini_report = select_val_mini(
        output_splits["val"],
        size=int(args.mini_eval_size),
        seed=int(args.seed) + 404,
    )

    write_jsonl(output_dir / "train.jsonl", output_splits["train"])
    write_jsonl(output_dir / "val.jsonl", output_splits["val"])
    # Do not write a duplicate validation.jsonl by default.  The full validation
    # split is val.jsonl; the cheap fixed subset is val-mini.jsonl.
    write_jsonl(output_dir / "val-mini.jsonl", val_mini)
    write_jsonl(output_dir / "test.jsonl", output_splits["test"])
    write_json(output_dir / "val_mini_manifest.json", mini_report)

    final_counts = {split: len(rows) for split, rows in output_splits.items()}
    final_total = sum(final_counts.values())
    synth_total = sum(quotas.values())
    stats["final_split_counts"] = final_counts
    stats["val_mini"] = mini_report
    stats["synthetic_total"] = synth_total
    stats["synthetic_fraction_final"] = round(synth_total / max(1, final_total), 4)
    stats["note"] = "S1K_RL = filtered copy of QRA plus synthetic hard short-answer cases. Real QRA samples are not re-split. Full validation is val.jsonl; fixed mini validation is val-mini.jsonl."

    write_json(output_dir / "build_report.json", stats)
    return stats


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    parser = argparse.ArgumentParser(description="Build S1K_RL by filtering QRA and adding synthetic hard final-answer cases.")
    parser.add_argument("--input-dir", type=str, default=str(script_dir / "data" / "QRA"))
    parser.add_argument("--output-dir", type=str, default=str(script_dir / "data" / "S1K_RL"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-final-fraction", type=float, default=0.35,
                        help="Target fraction of synthetic examples in the final dataset. Default 0.35.")
    parser.add_argument("--min-synthetic-eval", type=int, default=5,
                        help="Minimum synthetic examples in val and test. Train minimum is 8x this value.")
    parser.add_argument("--mini-eval-size", type=int, default=8,
                        help="Number of stratified validation examples written to val-mini.jsonl. Set 0 to disable.")
    parser.add_argument("--max-answer-chars", type=int, default=160)
    parser.add_argument("--max-drop-examples", type=int, default=30)
    parser.add_argument("--max-kept-examples", type=int, default=20)
    parser.add_argument("--allow-zero-real", action="store_true")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.set_defaults(overwrite=True)
    args = parser.parse_args()

    report = build_dataset(args)
    print(json.dumps({
        "input_dir": report["input_dir"],
        "output_dir": report["output_dir"],
        "raw_total_loaded": report["raw_total_loaded"],
        "real_counts_before_synthetic": report["real_counts_before_synthetic"],
        "synthetic_quotas": report["synthetic_quotas"],
        "final_split_counts": report["final_split_counts"],
        "val_mini_count": report.get("val_mini", {}).get("selected_size", 0),
        "val_mini_types": report.get("val_mini", {}).get("selected_type_counts", {}),
        "synthetic_fraction_final": report["synthetic_fraction_final"],
        "top_drop_reasons": sorted(report["drop_reasons"].items(), key=lambda x: x[1], reverse=True)[:10],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
