from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


SHORT_MAX_CHARS = 96
SHORT_MAX_TOKENS_APPROX = 28

QUESTION_KEYS = [
    "question",
    "problem",
    "prompt",
    "input",
    "query",
]

REASONING_KEYS = [
    "deepseek_thinking_trajectory",
    "gemini_thinking_trajectory",
    "reasoning",
    "rationale",
    "thinking",
    "cot",
    "chain_of_thought",
]

SOLUTION_KEYS = [
    "solution",
    "answer",
    "final_answer",
    "target",
    "output",
]

BAD_LONG_MARKERS = [
    "\\sum",
    "\\int",
    "\\prod",
    "\\lim",
    "\\begin",
    "\\end",
    "aligned",
    "equation",
    "cases",
]

FINAL_TEMPLATES = [
    " Therefore, the final answer is {answer}.",
    " Thus, the final solution is {answer}.",
    " Finally, the answer is {answer}.",
    " Hence, the answer is {answer}.",
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

    for left, right in [
        ("$$", "$$"),
        ("$", "$"),
        (r"\(", r"\)"),
        (r"\[", r"\]"),
    ]:
        if text.startswith(left) and text.endswith(right) and len(text) > len(left) + len(right):
            text = text[len(left) : -len(right)].strip()

    boxed = re.fullmatch(r"\\boxed\{(.+)\}", text, flags=re.S)
    if boxed:
        text = boxed.group(1).strip()

    text = text.strip().strip("` ")
    if len(text) > 1 and text[-1] in ".;":
        text = text[:-1].strip()

    text = normalize_spaces(text)
    return text


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

    # Long chained LaTeX equations such as r = ... = ... = 1.903 m should be removed.
    if len(raw) > 120 and raw.count("=") >= 2 and any(x in raw for x in ["\\frac", "\\sqrt", "\\sin", "\\cos", "\\left", "\\right"]):
        return True

    return False


def looks_like_short_answer(raw: str) -> Tuple[bool, str, str]:
    raw = str(raw or "").strip()
    if not raw:
        return False, "", "empty_solution"

    if has_long_derivation_shape(raw):
        return False, "", "long_derivation_or_display_math"

    cleaned = strip_short_answer_wrappers(raw)
    if not cleaned:
        return False, "", "empty_after_clean"
    if cleaned in {"$", "$$", r"\[", r"\]", r"\(", r"\)"}:
        return False, "", "math_delimiter_only"
    if "\n" in cleaned:
        return False, "", "newline_after_clean"
    if len(cleaned) > SHORT_MAX_CHARS:
        return False, "", "too_long"
    if len(cleaned.split()) > SHORT_MAX_TOKENS_APPROX:
        return False, "", "too_many_words"

    if any(marker in cleaned for marker in BAD_LONG_MARKERS):
        return False, "", "complex_latex"

    compact = cleaned.replace(" ", "")

    patterns = [
        # single option
        r"^[A-Za-z]$",
        # integer / decimal
        r"^[+-]?\d+$",
        r"^[+-]?(?:\d+\.\d+|\.\d+)$",
        # decimal with units
        r"^[+-]?(?:\d+\.\d+|\.\d+|\d+)(?:m|mm|cm|km|kg|g|s|ms|N|J|W|V|A|Hz|m/s|m/s\^2)$",
        # interval
        r"^[\(\[][+-]?(?:\d+(?:\.\d+)?|\.\d+),[+-]?(?:\d+(?:\.\d+)?|\.\d+)[\)\]]$",
        # equation / symbolic expression
        r"^[A-Za-z][A-Za-z0-9_]*=[A-Za-z0-9_+\-*/^().\\]+$",
        r"^[A-Za-z0-9_+\-*/^=().,\[\]{}\\]+$",
    ]

    if any(re.fullmatch(pattern, compact) for pattern in patterns):
        return True, cleaned, "kept"

    return False, "", "unsupported_short_form"


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


def make_sample(
    row: Dict[str, Any],
    split: str,
    idx: int,
    answer: str,
    synthetic: bool = False,
) -> Dict[str, Any]:
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

    return {
        "id": f"s1f_rl_{split}_{sid}",
        "source": "s1f_rl_synthetic" if synthetic else "s1f_rl_from_s1k",
        "question": question,
        "reasoning": reasoning,
        "answer": answer,
        "solution": answer,
        "segments": {
            "question": {
                "text": f"Question:\n{question.strip()}\n\n",
                "train": False,
            },
            "reasoning": {
                "text": f"Reasoning:\n{reasoning.strip()}\n",
                "train": False,
            },
            "answer_anchor": {
                "text": "\nAnswer:\n",
                "train": False,
            },
            "answer": {
                "text": str(answer).strip(),
                "train": True,
            },
        },
    }


EXTRA_CASES: List[Dict[str, str]] = [
    # Intervals
    {
        "id": "interval_open_01",
        "question": "Solve x^2 - x - 12 < 0 and write the solution interval.",
        "reasoning": "First, factor x^2 - x - 12 as (x-4)(x+3). The expression is negative between the roots. Therefore, the final answer is (-3,4).",
        "solution": "(-3,4)",
    },
    {
        "id": "interval_closed_open_01",
        "question": "Solve 2 <= x + 1 < 5 and write the interval for x.",
        "reasoning": "First, subtract 1 from each part. Then 1 <= x < 4. Thus, the final solution is [1,4).",
        "solution": "[1,4)",
    },
    {
        "id": "interval_closed_01",
        "question": "Solve |x| <= 2 and write the interval.",
        "reasoning": "First, |x| <= 2 means x is at least -2 and at most 2. Therefore, the final answer is [-2,2].",
        "solution": "[-2,2]",
    },
    {
        "id": "interval_open_closed_01",
        "question": "Solve -2 < x <= 2 and write the interval.",
        "reasoning": "First, the left endpoint is excluded and the right endpoint is included. Therefore, the final answer is (-2,2].",
        "solution": "(-2,2]",
    },
    {
        "id": "interval_closed_open_02",
        "question": "Solve 2 <= x < 5.",
        "reasoning": "First, the lower endpoint is included and the upper endpoint is excluded. Finally, the answer is [2,5).",
        "solution": "[2,5)",
    },
    {
        "id": "interval_open_02",
        "question": "Solve (x+2)(x-7) < 0.",
        "reasoning": "First, the roots are -2 and 7. Since the quadratic is negative between the roots, the final answer is (-2,7).",
        "solution": "(-2,7)",
    },
    {
        "id": "interval_open_closed_02",
        "question": "Write the interval for all x such that 3 < x <= 9.",
        "reasoning": "First, 3 is excluded and 9 is included. Therefore, the final answer is (3,9].",
        "solution": "(3,9]",
    },
    {
        "id": "interval_closed_02",
        "question": "Solve -1 <= x <= 6.",
        "reasoning": "First, both endpoints are included. Thus, the final solution is [-1,6].",
        "solution": "[-1,6]",
    },

    # Equations
    {
        "id": "equation_linear_01",
        "question": "Solve 2x + 1 = 7.",
        "reasoning": "First, subtract 1 to get 2x=6. Then divide by 2. Therefore, the final answer is x=3.",
        "solution": "x=3",
    },
    {
        "id": "equation_linear_02",
        "question": "Find y when y - 4 = 9.",
        "reasoning": "First, add 4 to both sides. Then y=13. Finally, the answer is y=13.",
        "solution": "y=13",
    },
    {
        "id": "equation_formula_01",
        "question": "Write Newton's second law solved for acceleration.",
        "reasoning": "First, Newton's second law is F=ma. Then divide by m to isolate acceleration. Therefore, the final answer is a=F/m.",
        "solution": "a=F/m",
    },
    {
        "id": "equation_line_01",
        "question": "Write the line with slope 2 and intercept 4.",
        "reasoning": "First, use y=mx+b. With m=2 and b=4, the final solution is y=2x+4.",
        "solution": "y=2x+4",
    },
    {
        "id": "equation_line_02",
        "question": "Write the line with slope 2 and intercept -5.",
        "reasoning": "First, use y=mx+b. With m=2 and b=-5, the final answer is y=2x-5.",
        "solution": "y=2x-5",
    },
    {
        "id": "equation_area_01",
        "question": "Write the area formula for a circle.",
        "reasoning": "First, the area of a circle depends on pi and r squared. Therefore, the final answer is A=pi*r^2.",
        "solution": "A=pi*r^2",
    },

    # Symbolic expressions
    {
        "id": "symbolic_sqrt_01",
        "question": "Simplify sqrt(50).",
        "reasoning": "First, write 50 as 25 times 2. Then sqrt(25)=5, so sqrt(50)=5sqrt(2). Thus, the final answer is 5sqrt(2).",
        "solution": "5sqrt(2)",
    },
    {
        "id": "symbolic_square_01",
        "question": "Expand (x+2)^2.",
        "reasoning": "First, use (a+b)^2=a^2+2ab+b^2. Then (x+2)^2=x^2+4x+4. Therefore, the final answer is x^2+4x+4.",
        "solution": "x^2+4x+4",
    },
    {
        "id": "symbolic_fraction_01",
        "question": "Simplify 6/8.",
        "reasoning": "First, divide numerator and denominator by 2. Then 6/8 becomes 3/4. Therefore, the final answer is 3/4.",
        "solution": "3/4",
    },
    {
        "id": "symbolic_physics_01",
        "question": "Write the kinematics relation for final speed squared.",
        "reasoning": "First, the constant-acceleration equation relates final speed, initial speed, acceleration, and displacement. Finally, the answer is v^2=u^2+2as.",
        "solution": "v^2=u^2+2as",
    },
    {
        "id": "symbolic_monomial_01",
        "question": "Simplify x*x*3.",
        "reasoning": "First, x times x is x^2. Then multiply by 3. Therefore, the final answer is 3x^2.",
        "solution": "3x^2",
    },
    {
        "id": "symbolic_parentheses_01",
        "question": "Factor x^2+2x+1.",
        "reasoning": "First, recognize the perfect square pattern. Therefore, the final answer is (x+1)^2.",
        "solution": "(x+1)^2",
    },

    # Unit decimals / signed decimals
    {
        "id": "unit_decimal_m_01",
        "question": "A measured length is 1.903 meters. Give the final value with unit.",
        "reasoning": "First, keep the decimal value and attach the meter unit. Therefore, the final answer is 1.903 m.",
        "solution": "1.903 m",
    },
    {
        "id": "unit_decimal_mm_01",
        "question": "A displacement is negative two and a half millimeters. Write it numerically with unit.",
        "reasoning": "First, negative two and a half is -2.50. The unit is millimeters. Thus, the final solution is -2.50 mm.",
        "solution": "-2.50 mm",
    },
    {
        "id": "unit_decimal_m_02",
        "question": "Write one eighth of a meter as a decimal with unit.",
        "reasoning": "First, one eighth is 0.125. Then attach the meter unit. Therefore, the final answer is 0.125 m.",
        "solution": "0.125 m",
    },
    {
        "id": "unit_decimal_kg_01",
        "question": "Write 350 grams in kilograms.",
        "reasoning": "First, divide grams by 1000 to convert to kilograms. Then 350 g is 0.35 kg. Finally, the answer is 0.35 kg.",
        "solution": "0.35 kg",
    },
    {
        "id": "unit_decimal_cm_01",
        "question": "A radius is 3.14 centimeters. Give the answer with unit.",
        "reasoning": "First, preserve the decimal value. The unit is centimeters. Therefore, the final answer is 3.14 cm.",
        "solution": "3.14 cm",
    },
    {
        "id": "unit_decimal_accel_01",
        "question": "Use the standard approximate gravitational acceleration near Earth with unit.",
        "reasoning": "First, the standard approximate value is negative when downward is taken as negative. Therefore, the final answer is -9.8 m/s^2.",
        "solution": "-9.8 m/s^2",
    },
    {
        "id": "signed_decimal_01",
        "question": "Compute -7.5 + 2.25.",
        "reasoning": "First, subtract the magnitudes because the signs differ. Then 7.5-2.25=5.25 and the larger magnitude is negative. Finally, the answer is -5.25.",
        "solution": "-5.25",
    },
    {
        "id": "decimal_01",
        "question": "Compute 19 divided by 10.",
        "reasoning": "First, dividing by 10 moves the decimal point one place left. Thus, the final solution is 1.9.",
        "solution": "1.9",
    },
]


def split_extra_cases(seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    cases = EXTRA_CASES[:]
    rng.shuffle(cases)

    n = len(cases)
    n_train = int(n * 0.8)
    n_val = max(1, int(n * 0.1))
    n_test = n - n_train - n_val

    if n_test <= 0:
        n_test = 1
        n_train = n - n_val - n_test

    return {
        "train": cases[:n_train],
        "val": cases[n_train : n_train + n_val],
        "test": cases[n_train + n_val :],
    }


def build_split(
    rows: List[Dict[str, Any]],
    split: str,
    seed: int,
    add_synthetic: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    out: List[Dict[str, Any]] = []
    stats: Dict[str, int] = {
        "total": len(rows),
        "kept": 0,
        "synthetic_added": 0,
    }

    for idx, row in enumerate(rows):
        solution = pick_first(row, SOLUTION_KEYS)
        ok, answer, reason = looks_like_short_answer(solution)

        if not ok:
            key = f"drop_{reason}"
            stats[key] = stats.get(key, 0) + 1
            continue

        out.append(make_sample(row, split, idx, answer, synthetic=False))
        stats["kept"] += 1

    start = len(out)
    for j, row in enumerate(add_synthetic):
        ok, answer, reason = looks_like_short_answer(row.get("solution", ""))

        if not ok:
            key = f"synthetic_drop_{reason}"
            stats[key] = stats.get(key, 0) + 1
            continue

        out.append(make_sample(row, split, start + j, answer, synthetic=True))
        stats["synthetic_added"] += 1

    split_seed_offset = {
        "train": 11,
        "val": 17,
        "test": 23,
    }.get(split, 0)

    rng = random.Random(seed + split_seed_offset)
    rng.shuffle(out)

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
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(repo_dir / "data" / "S1K"),
        help="Original S1K split directory, normally repo_root/data/S1K.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(script_dir / "data" / "S1F_RL"),
        help="Output directory. Default: rl_qra_pipeline/data/S1F_RL.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    extra_by_split = split_extra_cases(args.seed)

    summary: Dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "synthetic_total": len(EXTRA_CASES),
        "synthetic_types": {
            "interval": 8,
            "equation": 6,
            "symbolic_expression": 6,
            "unit_decimal_or_signed_decimal": 8,
        },
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        rows = load_split(input_dir, split)
        built, stats = build_split(
            rows=rows,
            split=split,
            seed=args.seed,
            add_synthetic=extra_by_split.get(split, []),
        )

        write_jsonl(output_dir / f"{split}.jsonl", built)

        summary["splits"][split] = {
            **stats,
            "written": len(built),
            "synthetic_assigned": len(extra_by_split.get(split, [])),
        }

        print(
            f"[{split}] "
            f"total={stats['total']} "
            f"kept={stats['kept']} "
            f"synthetic_added={stats['synthetic_added']} "
            f"written={len(built)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "build_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()