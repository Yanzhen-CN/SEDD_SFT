"""
Root-level S1K splitter/cleaner for SEDD SFT pipelines.

Outputs only the standard split structure:
  data/S1K/train.jsonl
  data/S1K/validation.jsonl
  data/S1K/test.jsonl
  data/S1K/manifest.json

  data/S1K_light/train.jsonl
  data/S1K_light/validation.jsonl
  data/S1K_light/test.jsonl
  data/S1K_light/manifest.json

Design:
  1) Clean the Answer field from `solution` first.
     - If solution is already a scalar / option / word / one-sentence short answer, keep it directly.
     - If solution contains explanation/reasoning, extract only the final answer from structured markers.
  2) Select a teacher reasoning trace and optionally verify that it reaches the cleaned answer.
  3) Keep question, labels and answer fixed. Only compress the reasoning body.
     Compression is complete-sentence based: no half-sentence truncation.
  4) Write train / validation / test / manifest only.

Pipeline adapters later add the segment masks for QA / QAR / QRA.
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml
from transformers import GPT2TokenizerFast

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, total=None, desc=None, **kwargs):
        if desc:
            print(desc, flush=True)
        return iterable if iterable is not None else range(total or 0)

REPO_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_DIR / "sft_answer_pipeline" / "answer_config.yaml"
DEFAULT_RAW_OUTPUT_DIR = REPO_DIR / "data" / "S1K"
DEFAULT_LIGHT_OUTPUT_DIR = REPO_DIR / "data" / "S1K_light"
ELLIPSIS = "..."
_TOKEN_CACHE: Dict[str, int] = {}

EXPLANATION_HEADINGS = (
    "explanation", "definition", "wordplay", "reasoning", "solution", "parse", "clue",
    "defn", "def", "analysis", "proof", "working", "work", "steps", "rationale",
)
BAD_ANSWER_PREFIXES = tuple(x + ":" for x in EXPLANATION_HEADINGS) + (
    "anagram of", "homophone of", "hidden in", "definition of", "clue is",
)
REASONING_CUES = [
    "according to", "based on", "given", "note that", "observe", "suppose", "assume",
    "because", "since", "therefore", "thus", "hence", "consequently", "as a result",
    "so", "then", "next", "first", "second", "finally", "we need", "we have",
    "we get", "we obtain", "this implies", "it follows", "solve", "compute", "calculate",
    "derive", "simplify", "substitute", "factor", "expand", "evaluate", "check", "verify",
    "equation", "case", "wait", "actually", "alternatively",
]
ANSWER_CUES = [
    "final answer", "the answer is", "answer is", "therefore the answer", "so the answer",
    "hence the answer", "we get", "we obtain", "equals", "is equal to", "\\boxed", "####",
]


def clean(text) -> str:
    return str(text or "").strip()


def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_s1k(data_cfg):
    if data_cfg.get("arrow_path"):
        from datasets import Dataset
        return Dataset.from_file(data_cfg["arrow_path"])
    from datasets import load_dataset
    return load_dataset(data_cfg.get("source_dataset", "simplescaling/s1K-1.1"), split="train")


def token_count(tokenizer, text) -> int:
    text = clean(text)
    cached = _TOKEN_CACHE.get(text)
    if cached is not None:
        return cached
    n = len(tokenizer(text, add_special_tokens=False).input_ids)
    if len(_TOKEN_CACHE) < 200000:
        _TOKEN_CACHE[text] = n
    return n


def _percentile(sorted_values: List[int], q: float):
    if not sorted_values:
        return 0
    idx = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * q))))
    return sorted_values[idx]


def summary(values: Iterable[int]) -> dict:
    values = sorted(int(v) for v in values)
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": values[0],
        "avg": round(sum(values) / len(values), 2),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "max": values[-1],
    }


def split_indices(n_rows: int, valid_ratio: float, test_ratio: float, seed: int):
    indices = list(range(n_rows))
    random.Random(seed).shuffle(indices)
    n_valid = max(1, int(n_rows * valid_ratio)) if n_rows > 1 and valid_ratio > 0 else 0
    n_test = max(1, int(n_rows * test_ratio)) if n_rows > 1 and test_ratio > 0 else 0
    if n_valid + n_test >= n_rows:
        raise ValueError("valid_ratio + test_ratio leaves no training data.")
    valid = set(indices[:n_valid])
    test = set(indices[n_valid : n_valid + n_test])
    return valid, test


def assign_split(index: int, valid_indices: set, test_indices: set) -> str:
    if index in valid_indices:
        return "validation"
    if index in test_indices:
        return "test"
    return "train"


def split_sentences(text) -> List[str]:
    text = clean(text)
    if not text:
        return []
    text = re.sub(r"\r\n?", "\n", text)
    units: List[str] = []
    for block in re.split(r"\n{2,}", text):
        block = clean(block)
        if not block:
            continue
        pieces = re.split(r"(?<=[。！？!?])\s+|(?<=[.!?])\s+(?=[A-Z0-9\\\(\[])", block)
        for piece in pieces:
            piece = clean(piece)
            if not piece:
                continue
            lines = [clean(x) for x in piece.split("\n") if clean(x)]
            if len(lines) > 1:
                units.extend(lines)
            else:
                units.append(piece)
    out = []
    for unit in units:
        if unit:
            out.append(unit)
    return out


def strip_markdown_answer_prefix(text: str) -> str:
    text = clean(text)
    text = re.sub(r"^\s*#{0,6}\s*", "", text).strip()
    text = re.sub(r"^\s*(?:final\s+)?answer\s*[:：=]\s*", "", text, flags=re.I).strip()
    text = re.sub(r"^\s*(?:the\s+)?(?:final\s+)?answer\s+(?:is|=)\s*", "", text, flags=re.I).strip()
    return text


def clean_answer_candidate(candidate: str) -> str:
    """Clean a candidate without adding tokens such as 'is'."""
    candidate = clean(candidate)
    if not candidate:
        return ""

    # Stop before explanation blocks if a regex captured too much.
    heading = r"(?:" + "|".join(EXPLANATION_HEADINGS) + r")"
    candidate = re.split(r"(?im)\n\s*#{0,6}\s*" + heading + r"\s*:", candidate, maxsplit=1)[0]
    candidate = re.split(r"(?im)\s+#{1,6}\s*" + heading + r"\s*:", candidate, maxsplit=1)[0]

    candidate = strip_markdown_answer_prefix(candidate)
    candidate = re.sub(r"^\s*(?:is|=)\s+", "", candidate, flags=re.I).strip()  # fixes 'is 181'
    candidate = re.split(r"\s+(?:because|since|as|which|therefore|so)\b", candidate, maxsplit=1, flags=re.I)[0]
    candidate = re.sub(r"^[-*•\s]+", "", candidate).strip()
    candidate = re.sub(r"^\*\*(.+?)\*\*$", r"\1", candidate).strip()
    candidate = candidate.strip(" \t\n\r'\"“”‘’`")
    candidate = re.sub(r"^\((.+)\)$", r"\1", candidate).strip()
    candidate = re.sub(r"[。!?]+$", "", candidate).strip()
    if not re.search(r"\d\.\d$", candidate):
        candidate = re.sub(r"\.$", "", candidate).strip()
    return candidate


def reject_bad_answer_candidate(text: str) -> bool:
    t = clean(text)
    low = t.lower().strip()
    if not t:
        return True
    if low.startswith(BAD_ANSWER_PREFIXES):
        return True
    if re.search(r"(?im)^\s*#{0,6}\s*(?:" + "|".join(EXPLANATION_HEADINGS) + r")\s*:", t):
        return True
    return False


def is_atomic_solution_answer(solution: str, tokenizer, max_tokens: int, max_chars: int) -> bool:
    """True when solution is already the final answer: number, option, word, phrase, or one short sentence.

    These cases should be kept directly rather than over-parsed.
    """
    s = clean(solution)
    if not s or reject_bad_answer_candidate(s):
        return False
    if "\n" in s:
        return False
    if re.search(r"(?i)^\s*#{1,6}\s*(?:answer|solution|explanation)\s*:", s):
        return False
    if re.search(r"(?i)\b(explanation|definition|wordplay|anagram of|proof|reasoning)\b", s):
        return False
    if len(s) > max_chars:
        return False
    if token_count(tokenizer, s) > max_tokens:
        return False
    # Scalar / option / boxed / short phrase.
    if re.fullmatch(r"[A-Da-d]", s) or re.fullmatch(r"\(?[A-Da-d]\)?", s):
        return True
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?(?:\s*[%°])?", s):
        return True
    if re.fullmatch(r"\\boxed\{[^{}]+\}", s):
        return True
    if len(split_sentences(s)) <= 1 and token_count(tokenizer, s) <= min(max_tokens, 80):
        return True
    return False


def candidate_ok(candidate: str, tokenizer, max_tokens: int, max_chars: int) -> bool:
    c = clean(candidate)
    if not c or reject_bad_answer_candidate(c):
        return False
    if len(c) > max_chars:
        return False
    if token_count(tokenizer, c) > max_tokens:
        return False
    # Multi-sentence is allowed if short enough, but reject obvious reasoning blocks.
    if "\n" in c:
        return False
    if len(split_sentences(c)) > 2 and token_count(tokenizer, c) > 80:
        return False
    if re.search(r"\b(first|second|next|then|therefore|because|since|we need to|we can|let us|let's)\b", c, flags=re.I):
        if token_count(tokenizer, c) > 40:
            return False
    return True


def add_solution_answer_candidates(solution: str, candidates: List[Tuple[str, str]]) -> None:
    text = clean(solution)
    if not text:
        return

    # Markdown line: ### Answer: GOATHERDS
    for line in text.splitlines():
        m = re.match(r"^\s*#{0,6}\s*(?:final\s+)?answer\s*[:：=]\s*(.+?)\s*$", line, flags=re.I)
        if m:
            candidates.append((clean_answer_candidate(m.group(1)), "solution_markdown_answer_line"))

    # Markdown block: Answer:\n...\n### Explanation:
    heading = r"(?:" + "|".join(EXPLANATION_HEADINGS) + r")"
    pat = re.compile(
        r"(?is)(?:^|\n)\s*#{0,6}\s*(?:final\s+)?answer\s*[:：=]\s*(.*?)"
        r"(?=\n\s*#{0,6}\s*" + heading + r"\s*:|\n\s*\*\*|\Z)"
    )
    for m in pat.finditer(text):
        block = clean(m.group(1))
        if block:
            first = [clean(x) for x in block.splitlines() if clean(x)]
            if first:
                candidates.append((clean_answer_candidate(first[0]), "solution_markdown_answer_block_first_line"))
            candidates.append((clean_answer_candidate(block), "solution_markdown_answer_block"))

    # GSM8K / boxed.
    for m in re.finditer(r"####\s*([^\n]+)", text):
        candidates.append((clean_answer_candidate(m.group(1)), "solution_gsm8k_hash"))
    for m in re.finditer(r"\\boxed\s*\{([^{}]+)\}", text, flags=re.DOTALL):
        candidates.append((clean_answer_candidate(m.group(1)), "solution_boxed"))

    # Explicit phrases. Avoid the old bug by requiring a separator/is after 'final answer'.
    patterns = [
        r"(?i)(?:the\s+)?final\s+answer\s*(?:is|=|:|：)\s*[\"“']?([^\n\"”']{1,260})[\"”']?",
        r"(?i)(?:the\s+)?answer\s*(?:is|=|:|：)\s*[\"“']?([^\n\"”']{1,260})[\"”']?",
        r"(?i)(?:therefore|so|hence),?\s+(?:the\s+)?answer\s*(?:is|=)\s*[\"“']?([^\n\"”']{1,220})[\"”']?",
    ]
    for pat in patterns:
        for m in reversed(list(re.finditer(pat, text))[-8:]):
            candidates.append((clean_answer_candidate(m.group(1)), "solution_explicit_answer_marker"))

    # Conservative tail fallback only when the tail itself looks like an answer, not an explanation label.
    lines = [clean(x) for x in text.splitlines() if clean(x)]
    for line in reversed(lines[-4:]):
        c = clean_answer_candidate(line)
        if c and not reject_bad_answer_candidate(c):
            candidates.append((c, "solution_safe_tail_line"))


def extract_clean_answer(solution, tokenizer, data_cfg) -> Tuple[str, str, dict]:
    solution = clean(solution)
    if not solution:
        return "", "empty_solution", {}

    max_length = int(data_cfg.get("_max_length", 1024))
    frac = float(data_cfg.get("max_answer_fraction_of_max_length", 0.33))
    # Old configs may still contain max_answer_tokens: 64. Treat it as a soft lower bound, not a hard cap.
    config_tok = int(data_cfg.get("max_answer_tokens", 0) or 0)
    max_tokens = max(config_tok, max(1, int(max_length * frac)))
    max_chars = int(data_cfg.get("max_answer_chars", 1200))

    if is_atomic_solution_answer(solution, tokenizer, max_tokens=max_tokens, max_chars=max_chars):
        # Keep simple scalar/word/option answers directly. No added prefixes such as 'is'.
        return clean_answer_candidate(solution), "solution_atomic", {"max_answer_tokens": max_tokens}

    candidates: List[Tuple[str, str]] = []
    add_solution_answer_candidates(solution, candidates)

    seen = set()
    rejected = []
    for cand, method in candidates:
        cand = clean_answer_candidate(cand)
        key = re.sub(r"\s+", " ", cand.lower()).strip()
        if not cand or key in seen:
            continue
        seen.add(key)
        if candidate_ok(cand, tokenizer, max_tokens=max_tokens, max_chars=max_chars):
            return cand, method, {"max_answer_tokens": max_tokens, "candidate_count": len(candidates)}
        if len(rejected) < 6:
            rejected.append({"method": method, "candidate": cand[:160]})

    # Whole-solution fallback only if it is one short non-reasoning answer.
    whole = clean_answer_candidate(solution)
    if candidate_ok(whole, tokenizer, max_tokens=max_tokens, max_chars=max_chars):
        return whole, "whole_clean_solution", {"max_answer_tokens": max_tokens, "candidate_count": len(candidates)}

    return "", "long_or_unextractable_answer", {"max_answer_tokens": max_tokens, "candidate_count": len(candidates), "rejected_preview": rejected}


def normalize_for_match(text) -> str:
    text = clean(text).lower()
    text = re.sub(r"\\boxed\s*\{([^{}]+)\}", r"\1", text)
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9.+\-*/=]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_matches_reasoning(answer: str, reasoning: str, data_cfg: dict) -> Tuple[bool, str]:
    if not bool(data_cfg.get("require_answer_in_reasoning", True)):
        return True, "disabled"
    answer = clean(answer)
    reasoning = clean(reasoning)
    if not answer or not reasoning:
        return False, "empty_answer_or_reasoning"

    a = normalize_for_match(answer)
    if len(a) < 2:
        return True, "answer_too_short_relaxed"
    if len(answer) > int(data_cfg.get("max_answer_chars_for_reasoning_match", 200)):
        return True, "answer_too_long_relaxed"
    r = normalize_for_match(reasoning)
    if a and a in r:
        return True, "normalized_answer_substring"
    if re.fullmatch(r"[a-zA-Z][a-zA-Z\-]{1,}", answer):
        if re.search(r"\b" + re.escape(answer.lower()) + r"\b", reasoning.lower()):
            return True, "word_boundary_match"
    return False, "answer_not_found_in_reasoning"


def choose_reasoning(row, priority, data_cfg) -> Tuple[Optional[str], Optional[str], str]:
    text_source = data_cfg.get("reasoning_text_source", "thinking")
    fields = {
        "deepseek": {"thinking": "deepseek_thinking_trajectory", "attempt": "deepseek_attempt"},
        "gemini": {"thinking": "gemini_thinking_trajectory", "attempt": "gemini_attempt"},
    }
    for source in priority:
        order = ["attempt", "thinking"] if text_source == "attempt" else ["thinking", "attempt"]
        for variant in order:
            field = fields.get(source, {}).get(variant)
            text = clean(row.get(field)) if field else ""
            if text:
                return source, field, text
    return None, None, ""


def qar_text(question: str, reasoning: str, answer: str) -> str:
    return f"User: {clean(question)}\nAssistant:\nReasoning:\n{clean(reasoning)}\n\nAnswer:\n{clean(answer)}"


def q_text_with_empty_reasoning(question: str, answer: str) -> str:
    return f"User: {clean(question)}\nAssistant:\nReasoning:\n\n\nAnswer:\n{clean(answer)}"


def sentence_has_math(text) -> bool:
    return bool(
        re.search(r"\\\(|\\\[|\\frac|\\sqrt|\\boxed|[$=<>^_{}]|\d+\s*[+\-*/=]\s*\d+", text)
        or re.search(r"\b(equation|solve|compute|calculate|simplify|substitute|factor|expand|evaluate|sum|product)\b", text, re.I)
    )


def sentence_score(sentence, answer, index, n_sentences) -> float:
    low = sentence.lower()
    s_norm = normalize_for_match(sentence)
    a_norm = normalize_for_match(answer)
    score = 0.0
    if a_norm and len(a_norm) >= 2 and a_norm in s_norm:
        score += 8.0
    if any(cue in low for cue in ANSWER_CUES):
        score += 5.0
    if any(cue in low for cue in REASONING_CUES):
        score += 3.0
    if sentence_has_math(sentence):
        score += 2.0
    words = sentence.split()
    if 6 <= len(words) <= 80:
        score += 1.0
    if index == 0:
        score += 2.0
    if index == n_sentences - 1:
        score += 3.0
    return score


def assemble_sentences(sentences: List[str], selected_indices: List[int]) -> str:
    selected = sorted(set(selected_indices))
    out, prev = [], None
    for idx in selected:
        if prev is not None and idx > prev + 1:
            out.append(ELLIPSIS)
        out.append(clean(sentences[idx]))
        prev = idx
    return "\n".join(x for x in out if x)


def compress_reasoning_to_budget(reasoning, answer, tokenizer, budget, crop_cfg):
    reasoning = clean(reasoning)
    if budget <= 0:
        return "", "no_budget", {"budget_tokens": budget}
    sentences = split_sentences(reasoning)
    if not sentences:
        return "", "no_sentences", {"budget_tokens": budget}

    n = len(sentences)
    sent_tokens = [token_count(tokenizer, s) for s in sentences]
    scores = [sentence_score(s, answer, i, n) for i, s in enumerate(sentences)]
    first_k = int(crop_cfg.get("keep_first_sentences", 2))
    last_k = int(crop_cfg.get("keep_last_sentences", 2))
    anchor_k = int(crop_cfg.get("keep_anchor_sentences", 5))
    stride_k = int(crop_cfg.get("max_stride_sentences", 6))

    priorities: Dict[int, Tuple[float, str]] = {}

    def add(idx, priority, tag):
        if 0 <= idx < n:
            old = priorities.get(idx)
            if old is None or priority > old[0]:
                priorities[idx] = (priority, tag)

    for i in range(min(first_k, n)):
        add(i, 100 + scores[i], "first")
    for i in range(max(0, n - last_k), n):
        add(i, 95 + scores[i], "last")
    top = sorted(range(n), key=lambda i: (scores[i], i), reverse=True)
    for i in top[:anchor_k]:
        add(i, 85 + scores[i], "anchor")
    if stride_k > 0 and n > first_k + last_k:
        for j in range(1, stride_k + 1):
            add(round(j * (n - 1) / (stride_k + 1)), 40 + scores[round(j * (n - 1) / (stride_k + 1))], "stride")
    for i in top:
        add(i, 10 + scores[i], "filler")

    selected, tags = [], {}
    for _, idx, tag in sorted(((p, i, tag) for i, (p, tag) in priorities.items()), reverse=True):
        if sent_tokens[idx] > budget:
            continue
        candidate = sorted(set(selected + [idx]))
        text = assemble_sentences(sentences, candidate)
        if token_count(tokenizer, text) <= budget:
            selected = candidate
            tags[idx] = tag
    if not selected:
        for i in sorted(range(n), key=lambda j: sent_tokens[j]):
            if sent_tokens[i] <= budget:
                selected = [i]
                tags[i] = "shortest_fallback"
                break
    text = assemble_sentences(sentences, selected)
    return text, ("sentence_anchor_crop" if text else "crop_failed"), {
        "budget_tokens": budget,
        "cropped_tokens": token_count(tokenizer, text) if text else 0,
        "num_sentences": n,
        "selected_indices": selected,
        "selected_tags": {str(k): v for k, v in tags.items()},
    }


def raw_row_for_output(row: dict, idx: int, split: str) -> dict:
    out = dict(row)
    out["id"] = f"s1k_{idx}"
    out["source_index"] = idx
    out["split"] = split
    return out


def process_and_split(config: dict) -> dict:
    data_cfg = dict(config.get("data", {}) or {})
    model_cfg = config.get("model", {})
    max_length = int(model_cfg.get("max_length", 1024))
    data_cfg["_max_length"] = max_length
    raw_output_dir = Path(data_cfg.get("raw_output_dir") or DEFAULT_RAW_OUTPUT_DIR)
    light_output_dir = Path(data_cfg.get("light_output_dir") or DEFAULT_LIGHT_OUTPUT_DIR)
    priority = data_cfg.get("reasoning_source_priority", ["deepseek", "gemini"])
    answer_field = data_cfg.get("answer_field", "solution")
    crop_cfg = dict(data_cfg.get("reasoning_crop", {}) or {})
    crop_cfg.setdefault("enabled", True)
    crop_cfg.setdefault("force", True)
    crop_cfg.setdefault("max_reasoning_tokens", 384)
    crop_cfg.setdefault("safety_margin_tokens", 8)

    print("[1/5] Loading tokenizer and S1K data...", flush=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e9)
    raw_rows = [dict(r) for r in load_s1k(data_cfg)]
    print(f"Loaded {len(raw_rows)} raw rows.", flush=True)

    valid_indices, test_indices = split_indices(
        len(raw_rows),
        float(data_cfg.get("valid_ratio", 0.1)),
        float(data_cfg.get("test_ratio", 0.1)),
        int(data_cfg.get("seed", 42)),
    )

    print("[2/5] Writing raw S1K split files...", flush=True)
    raw_splits = {"train": [], "validation": [], "test": []}
    for idx, row in enumerate(tqdm(raw_rows, desc="write data/S1K", dynamic_ncols=True)):
        split = assign_split(idx, valid_indices, test_indices)
        raw_splits[split].append(raw_row_for_output(row, idx, split))
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in raw_splits.items():
        write_jsonl(raw_output_dir / f"{split}.jsonl", rows)
    write_json(raw_output_dir / "manifest.json", {
        "format": "Original S1K rows split into train/validation/test. No cleaning or masks.",
        "source_dataset": data_cfg.get("source_dataset", "simplescaling/s1K-1.1"),
        "rows": {k: len(v) for k, v in raw_splits.items()},
        "files": ["train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"],
    })

    print("[3/5] Cleaning answers and selecting matching reasoning traces...", flush=True)
    light_splits = {"train": [], "validation": [], "test": []}
    counters = {
        "raw_rows": len(raw_rows),
        "missing_question": 0,
        "missing_reasoning": 0,
        "missing_answer": 0,
        "reasoning_answer_mismatch": 0,
        "usable_after_answer_cleaning": 0,
        "full_reasoning_within_budget": 0,
        "need_reasoning_compression": 0,
        "compressed_success": 0,
        "compression_failed": 0,
    }
    answer_extract_counts: Dict[str, int] = {}
    answer_match_counts: Dict[str, int] = {}
    reasoning_field_counts: Dict[str, int] = {}
    reasoning_source_counts: Dict[str, int] = {}
    crop_method_counts: Dict[str, int] = {}
    token_stats_all = {"question_tokens": [], "answer_tokens": [], "reasoning_tokens": [], "qar_total_tokens": []}
    skipped_examples = {"missing_answer": [], "reasoning_answer_mismatch": []}

    for idx, row in enumerate(tqdm(raw_rows, desc="clean answer + crop reasoning", dynamic_ncols=True)):
        split = assign_split(idx, valid_indices, test_indices)
        question = clean(row.get("question"))
        if not question:
            counters["missing_question"] += 1
            continue
        source, reasoning_field, raw_reasoning = choose_reasoning(row, priority, data_cfg)
        if not raw_reasoning:
            counters["missing_reasoning"] += 1
            continue

        answer, answer_method, answer_meta = extract_clean_answer(row.get(answer_field), tokenizer, data_cfg)
        if not answer:
            counters["missing_answer"] += 1
            if len(skipped_examples["missing_answer"]) < 10:
                skipped_examples["missing_answer"].append({
                    "source_index": idx,
                    "question_preview": question[:120],
                    "solution_preview": clean(row.get(answer_field))[:240],
                    "answer_method": answer_method,
                    "answer_meta": answer_meta,
                })
            continue

        ok, match_method = answer_matches_reasoning(answer, raw_reasoning, data_cfg)
        if not ok:
            counters["reasoning_answer_mismatch"] += 1
            if len(skipped_examples["reasoning_answer_mismatch"]) < 10:
                skipped_examples["reasoning_answer_mismatch"].append({
                    "source_index": idx,
                    "question_preview": question[:120],
                    "answer": answer,
                    "match_method": match_method,
                    "reasoning_tail_preview": raw_reasoning[-240:],
                })
            continue

        fixed_tokens = token_count(tokenizer, q_text_with_empty_reasoning(question, answer))
        available_budget = max_length - fixed_tokens - int(crop_cfg.get("safety_margin_tokens", 8))
        max_reasoning_tokens = int(crop_cfg.get("max_reasoning_tokens", 384))
        force_crop = bool(crop_cfg.get("force", True))
        reasoning_budget = available_budget
        if force_crop and max_reasoning_tokens > 0:
            reasoning_budget = min(reasoning_budget, max_reasoning_tokens)

        raw_reasoning_tokens = token_count(tokenizer, raw_reasoning)
        if available_budget <= 0:
            counters["compression_failed"] += 1
            continue

        needs_crop = raw_reasoning_tokens > available_budget or (force_crop and raw_reasoning_tokens > reasoning_budget)
        if needs_crop:
            counters["need_reasoning_compression"] += 1
            reasoning, crop_method, crop_meta = compress_reasoning_to_budget(raw_reasoning, answer, tokenizer, reasoning_budget, crop_cfg)
            total_tokens = token_count(tokenizer, qar_text(question, reasoning, answer)) if reasoning else 10**9
            if not reasoning or total_tokens > max_length:
                counters["compression_failed"] += 1
                crop_method_counts[crop_method] = crop_method_counts.get(crop_method, 0) + 1
                continue
            counters["compressed_success"] += 1
        else:
            reasoning = raw_reasoning
            total_tokens = token_count(tokenizer, qar_text(question, reasoning, answer))
            crop_method = "full_reasoning_within_budget"
            crop_meta = {}
            counters["full_reasoning_within_budget"] += 1

        crop_method_counts[crop_method] = crop_method_counts.get(crop_method, 0) + 1
        answer_extract_counts[answer_method] = answer_extract_counts.get(answer_method, 0) + 1
        answer_match_counts[match_method] = answer_match_counts.get(match_method, 0) + 1
        reasoning_field_counts[reasoning_field] = reasoning_field_counts.get(reasoning_field, 0) + 1
        reasoning_source_counts[source] = reasoning_source_counts.get(source, 0) + 1
        counters["usable_after_answer_cleaning"] += 1

        q_tok = token_count(tokenizer, question)
        a_tok = token_count(tokenizer, answer)
        r_tok = token_count(tokenizer, reasoning)
        for k, v in [("question_tokens", q_tok), ("answer_tokens", a_tok), ("reasoning_tokens", r_tok), ("qar_total_tokens", total_tokens)]:
            token_stats_all[k].append(v)

        light_row = {
            "id": f"s1k_{idx}",
            "source_index": idx,
            "split": split,
            "question": question,
            "reasoning": reasoning,
            "answer": answer,
            "reasoning_source": source,
            "reasoning_field": reasoning_field,
            "answer_field": answer_field,
            "answer_extract_method": answer_method,
            "answer_match_method": match_method,
            "reasoning_processing": {
                "method": crop_method,
                "fixed_tokens_without_reasoning": fixed_tokens,
                "available_reasoning_budget_tokens": available_budget,
                "reasoning_budget_tokens": reasoning_budget,
                "raw_reasoning_tokens": raw_reasoning_tokens,
                "reasoning_tokens": r_tok,
                **crop_meta,
            },
            "token_stats": {
                "question_tokens": q_tok,
                "answer_tokens": a_tok,
                "reasoning_tokens": r_tok,
                "qar_total_tokens": total_tokens,
            },
        }
        light_splits[split].append(light_row)

    print("[4/5] Writing S1K_light split files...", flush=True)
    light_output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in light_splits.items():
        write_jsonl(light_output_dir / f"{split}.jsonl", rows)

    print("[5/5] Writing S1K_light manifest...", flush=True)
    manifest = {
        "format": "S1K_light: cleaned final answer + complete-sentence reasoning compression. No train masks.",
        "source_dataset": data_cfg.get("source_dataset", "simplescaling/s1K-1.1"),
        "max_length": max_length,
        "files": ["train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"],
        "rows": {k: len(v) for k, v in light_splits.items()},
        "counters": counters,
        "answer_extract_counts": answer_extract_counts,
        "answer_match_counts": answer_match_counts,
        "reasoning_source_counts": reasoning_source_counts,
        "reasoning_field_counts": reasoning_field_counts,
        "crop_method_counts": crop_method_counts,
        "token_stats": {k: summary(v) for k, v in token_stats_all.items()},
        "answer_cleaning_note": "Atomic solution answers such as numbers, options and one-word answers are kept directly. Longer/multi-line solutions are cleaned to final-answer only when structured markers are available.",
        "skipped_examples": skipped_examples,
    }
    write_json(light_output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Prepare data/S1K and data/S1K_light splits for SEDD SFT pipelines.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    process_and_split(load_config(args.config))


if __name__ == "__main__":
    main()
