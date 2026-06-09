"""
Root-level S1K preparation for SEDD SFT pipelines.

This script does the shared data work once and writes two data folders:
  data/S1K/       : raw S1K rows, kept for inspection/debugging
  data/S1K_light/ : cleaned/light rows used by QA, QAR and QRA adapters

Current intended order:
  1) Clean/extract the final answer from `solution` first.
     The Answer section should be the final answer only, not the explanation/reasoning.
  2) Optionally verify that the selected reasoning trace actually contains/matches that answer.
     This filters out cases where the teacher reasoning solved a different answer.
  3) Compute the fixed length of question + anchors + cleaned answer, then crop only the reasoning body.
     The question, labels and cleaned answer are never cropped here.
  4) Split the processed S1K_light rows into train / validation / test.

S1K_light rows are content-level rows only. They do not contain train masks.
Pipeline adapters create masks later:
  QA : question -> answer
  QAR: question -> reasoning + answer
  QRA: question + teacher reasoning -> answer
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

REASONING_CUES = [
    "according to", "based on", "given", "given that", "note that", "notice that",
    "observe that", "recall that", "suppose", "assume", "consider", "let ",
    "because", "since", "therefore", "thus", "hence", "consequently", "as a result",
    "so", "then", "next", "first", "second", "finally", "we need", "we have",
    "we get", "we obtain", "we can", "this implies", "it follows", "leads to",
    "solve", "compute", "calculate", "derive", "simplify", "substitute", "factor",
    "expand", "evaluate", "check", "verify", "equation", "constraint", "case",
    "wait", "actually", "alternatively",
]
ANSWER_CUES = [
    "final answer", "the answer is", "answer is", "therefore the answer",
    "so the answer", "hence the answer", "we get", "we obtain", "equals",
    "is equal to", "\\boxed", "####",
]
EXPLANATION_HEADINGS = (
    "explanation", "definition", "wordplay", "reasoning", "solution", "parse", "clue",
    "defn", "def", "analysis", "proof", "working", "work", "steps",
)
BAD_ANSWER_PREFIXES = (
    "defn:", "definition:", "wordplay:", "parse:", "explanation:", "clue:",
    "def:", "cryptic:", "reasoning:", "solution:", "analysis:", "proof:",
    "working:", "work:", "steps:", "anagram of", "homophone of", "hidden in",
)


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
        raise ValueError("valid_ratio + test_ratio leaves no training data after filtering.")
    valid = set(indices[:n_valid])
    test = set(indices[n_valid : n_valid + n_test])
    return valid, test


def split_sentences(text) -> List[str]:
    """Split into complete sentence-like units; no token-level half-sentence truncation."""
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
    out, seen = [], set()
    for unit in units:
        key = re.sub(r"\s+", " ", unit).strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(unit)
    return out


def _strip_boxed(text: str) -> str:
    text = clean(text)
    m = re.fullmatch(r"\\boxed\s*\{(.+)\}", text, flags=re.DOTALL)
    if m:
        return clean(m.group(1))
    return text


def clean_answer_candidate(candidate: str) -> str:
    """Clean a possible final-answer string, but do not keep explanations."""
    candidate = clean(candidate)
    if not candidate:
        return ""

    # Cut off any explanation heading accidentally captured after the answer.
    candidate = re.split(
        r"(?im)\n\s*#{0,6}\s*(?:" + "|".join(EXPLANATION_HEADINGS) + r")\s*:",
        candidate,
        maxsplit=1,
    )[0]
    candidate = re.split(
        r"(?im)\s+#{1,6}\s*(?:" + "|".join(EXPLANATION_HEADINGS) + r")\s*:",
        candidate,
        maxsplit=1,
    )[0]

    # Strip markdown headings/answer labels.
    candidate = re.sub(r"^\s*#{0,6}\s*(?:final\s+)?answer\s*:\s*", "", candidate, flags=re.I).strip()
    candidate = re.sub(r"^\s*(?:the\s+)?(?:final\s+)?answer\s+(?:is|=)\s*", "", candidate, flags=re.I).strip()

    # If explicit answer phrase captured a whole clause, keep the answer-like prefix.
    candidate = re.split(r"\s+(?:because|since|as|which|that\s+is\s+why|therefore|so)\b", candidate, maxsplit=1, flags=re.I)[0]

    # Remove common markdown/emphasis/wrappers.
    candidate = re.sub(r"^[-*•\s]+", "", candidate).strip()
    candidate = re.sub(r"^\*\*(.+?)\*\*$", r"\1", candidate).strip()
    candidate = candidate.strip(" \t\n\r'\"“”‘’`")
    candidate = re.sub(r"^\((.+)\)$", r"\1", candidate).strip()

    # Prefer first nonempty line if a remaining candidate is multiline. The Answer field
    # should not include the explanation block.
    lines = [clean(x) for x in candidate.splitlines() if clean(x)]
    if lines:
        candidate = lines[0]

    # Remove final punctuation unless it is likely part of a formula/decimal.
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
    # Explanation-like snippets should not become the Answer section.
    if low.startswith(("anagram of", "homophone of", "hidden in", "definition of", "clue is")):
        return True
    if "…" in t and not re.search(r"[A-Za-z0-9]", t.replace("…", "")):
        return True
    return False


def looks_like_clean_answer(text, max_chars=1200, max_tokens=None, tokenizer=None) -> bool:
    """Accept a final answer that may be a scalar, phrase, or short natural-language answer.

    It may be longer than a single token, but it should not be a multi-section reasoning block.
    """
    text = clean(text)
    if not text or reject_bad_answer_candidate(text):
        return False
    if len(text) > max_chars:
        return False
    if max_tokens is not None and tokenizer is not None and token_count(tokenizer, text) > max_tokens:
        return False

    # Reject obvious reasoning blocks, even if they fit the token budget.
    if "\n" in text:
        return False
    if len(split_sentences(text)) > 2 and token_count(tokenizer, text) > 80:
        return False
    if re.search(r"\b(first|second|next|then|therefore|because|since|we need to|we can|let us|let's)\b", text, flags=re.I):
        # Allow short phrase answers that happen to contain a cue word, but reject long reasoning-like text.
        if token_count(tokenizer, text) > 40:
            return False
    return True


def add_markdown_answer_candidates(text: str, method_prefix: str, candidates: List[Tuple[str, str]]) -> None:
    """Extract from structured answer sections before any tail fallback.

    Handles:
      ### Answer: GOATHERDS
      Answer:\nGOATHERDS\n### Explanation: ...
    """
    text = clean(text)
    if not text:
        return

    # Line form: `### Answer: GOATHERDS`.
    for line in text.splitlines():
        m = re.match(r"^\s*#{0,6}\s*(?:final\s+)?answer\s*:\s*(.+?)\s*$", line, flags=re.I)
        if m:
            candidates.append((clean_answer_candidate(m.group(1)), f"{method_prefix}_markdown_answer_line"))

    # Block form: `Answer:` followed by lines until next heading.
    heading = r"(?:" + "|".join(EXPLANATION_HEADINGS) + r")"
    block_pattern = re.compile(
        r"(?is)(?:^|\n)\s*#{0,6}\s*(?:final\s+)?answer\s*:\s*(.*?)"
        r"(?=\n\s*#{0,6}\s*" + heading + r"\s*:|\n\s*\*\*|\Z)"
    )
    for m in block_pattern.finditer(text):
        block = clean(m.group(1))
        if not block:
            continue
        lines = [clean(x) for x in block.splitlines() if clean(x)]
        if lines:
            candidates.append((clean_answer_candidate(lines[0]), f"{method_prefix}_markdown_answer_block_first_line"))
        candidates.append((clean_answer_candidate(block), f"{method_prefix}_markdown_answer_block"))


def add_explicit_answer_candidates(text: str, method_prefix: str, candidates: List[Tuple[str, str]]) -> None:
    text = clean(text)
    if not text:
        return
    patterns = [
        r"(?i)(?:final\s+answer|the\s+answer\s+is|answer\s+is|answer\s*:|therefore\s+the\s+answer\s+is|so\s+the\s+answer\s+is)\s*[\"“']?([^\n\"”']{1,220})[\"”']?",
        r"(?i)(?:we\s+get|we\s+obtain)\s+[\"“']?([^\n\"”']{1,160})[\"”']?",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        for m in reversed(matches[-8:]):
            candidates.append((clean_answer_candidate(m.group(1)), f"{method_prefix}_explicit_answer_marker"))


def add_math_answer_candidates(text: str, method_prefix: str, candidates: List[Tuple[str, str]]) -> None:
    text = clean(text)
    if not text:
        return
    for m in re.finditer(r"\\boxed\s*\{([^{}]+)\}", text, flags=re.DOTALL):
        candidates.append((clean("\\boxed{" + m.group(1) + "}"), f"{method_prefix}_boxed"))
    for m in re.finditer(r"####\s*([^\n]+)", text):
        candidates.append((clean_answer_candidate(m.group(1)), f"{method_prefix}_gsm8k_hash"))


def add_safe_tail_candidates(solution: str, candidates: List[Tuple[str, str]]) -> None:
    """Very low-priority fallback. Only add tail lines that look answer-like.

    This avoids the old bug where `Defn: ...` became the final answer.
    """
    lines = [clean(x) for x in clean(solution).splitlines() if clean(x)]
    for line in reversed(lines[-5:]):
        c = clean_answer_candidate(line)
        if c and not reject_bad_answer_candidate(c):
            candidates.append((c, "solution_safe_tail_line"))

    sentences = split_sentences(solution)
    for sent in reversed(sentences[-5:]):
        c = clean_answer_candidate(sent)
        if c and not reject_bad_answer_candidate(c):
            candidates.append((c, "solution_safe_tail_sentence"))

    tail = clean(solution)[-1200:]
    expr_patterns = [
        r"\\boxed\s*\{[^{}]+\}",
        r"[A-Za-z]?\s*=\s*[+-]?\d+(?:\.\d+)?(?:/\d+)?",
        r"[+-]?\d+(?:\.\d+)?(?:/\d+)?(?:\s*[%°])?",
    ]
    for pattern in expr_patterns:
        matches = re.findall(pattern, tail)
        for val in reversed(matches[-3:]):
            candidates.append((clean_answer_candidate(val), "solution_tail_numeric_or_formula"))


def extract_clean_answer(solution, tokenizer, data_cfg, reasoning_text="") -> Tuple[str, str, dict]:
    """Extract the Answer field from solution and clean out reasoning/explanation.

    Priority:
      1. structured answer sections from solution, e.g. `### Answer: ...`
      2. boxed / GSM8K hash from solution
      3. explicit answer phrases from solution
      4. explicit answer phrases from reasoning, as fallback only
      5. very conservative tail candidates
      6. whole solution only if it is already a clean answer
    """
    solution = clean(solution)
    reasoning_text = clean(reasoning_text)
    if not solution and not reasoning_text:
        return "", "empty_solution_and_reasoning", {"candidate_count": 0}

    max_chars = int(data_cfg.get("max_answer_chars", 1200))
    explicit_max_tokens = int(data_cfg.get("max_answer_tokens", 256))
    effective_max_tokens = int(data_cfg.get("_effective_max_answer_tokens", explicit_max_tokens))
    max_tokens = max(1, min(explicit_max_tokens, effective_max_tokens))

    candidates: List[Tuple[str, str]] = []
    add_markdown_answer_candidates(solution, "solution", candidates)
    add_math_answer_candidates(solution, "solution", candidates)
    add_explicit_answer_candidates(solution, "solution", candidates)

    # Fallback to reasoning markers only after trying solution answer sections.
    add_markdown_answer_candidates(reasoning_text, "reasoning", candidates)
    add_explicit_answer_candidates(reasoning_text, "reasoning", candidates)
    add_math_answer_candidates(reasoning_text, "reasoning", candidates)

    add_safe_tail_candidates(solution, candidates)

    seen = set()
    rejected_preview = []
    for candidate, method in candidates:
        candidate = clean_answer_candidate(candidate)
        candidate = _strip_boxed(candidate) if "boxed" not in method else candidate
        candidate = re.sub(r"^\*\*(.+?)\*\*$", r"\1", candidate).strip()
        key = re.sub(r"\s+", " ", candidate.lower()).strip()
        if not candidate or key in seen:
            continue
        seen.add(key)
        if looks_like_clean_answer(candidate, max_chars=max_chars, max_tokens=max_tokens, tokenizer=tokenizer):
            return candidate, method, {"candidate_count": len(candidates), "max_answer_tokens": max_tokens}
        if len(rejected_preview) < 8:
            rejected_preview.append({"method": method, "candidate": candidate[:160]})

    whole = clean_answer_candidate(solution)
    if looks_like_clean_answer(whole, max_chars=max_chars, max_tokens=max_tokens, tokenizer=tokenizer):
        return whole, "whole_clean_solution", {"candidate_count": len(candidates), "max_answer_tokens": max_tokens}

    return "", "long_or_unextractable_answer", {
        "candidate_count": len(candidates),
        "max_answer_tokens": max_tokens,
        "rejected_preview": rejected_preview,
    }


def normalize_for_match(text) -> str:
    text = clean(text).lower()
    text = re.sub(r"\\boxed\s*\{([^{}]+)\}", r"\1", text)
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9.+\-*/=]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_matches_reasoning(answer: str, reasoning: str, data_cfg: dict) -> Tuple[bool, str]:
    """Light correctness filter: keep reasoning only if it appears to reach the selected answer.

    This is intentionally string-based and conservative; it is not a semantic judge.
    If the answer is too long to match exactly, the check is relaxed.
    """
    if not bool(data_cfg.get("require_answer_in_reasoning", True)):
        return True, "disabled"

    answer = clean(answer)
    reasoning = clean(reasoning)
    if not answer or not reasoning:
        return False, "empty_answer_or_reasoning"

    max_exact_chars = int(data_cfg.get("max_answer_chars_for_reasoning_match", 200))
    if len(answer) > max_exact_chars:
        # Long natural-language answers may not appear verbatim in reasoning; do not over-filter them.
        return True, "answer_too_long_for_exact_match_relaxed"

    a = normalize_for_match(answer)
    r = normalize_for_match(reasoning)
    if not a:
        return False, "empty_normalized_answer"
    if len(a) >= 2 and a in r:
        return True, "normalized_answer_substring"

    # For single-word answers, require a word boundary match before normalization fallback.
    if re.fullmatch(r"[a-zA-Z][a-zA-Z\-]{1,}", answer):
        if re.search(r"\b" + re.escape(answer.lower()) + r"\b", reasoning.lower()):
            return True, "word_boundary_match"

    # Try explicit answer candidates from reasoning and compare them to the selected answer.
    reason_candidates: List[Tuple[str, str]] = []
    add_markdown_answer_candidates(reasoning, "reasoning", reason_candidates)
    add_explicit_answer_candidates(reasoning, "reasoning", reason_candidates)
    add_math_answer_candidates(reasoning, "reasoning", reason_candidates)
    for cand, method in reason_candidates:
        c = normalize_for_match(clean_answer_candidate(cand))
        if c and (c == a or c in a or a in c):
            return True, f"reasoning_marker_match:{method}"

    return False, "answer_not_found_in_reasoning"


def choose_reasoning(row, priority, data_cfg) -> Tuple[Optional[str], Optional[str], str]:
    text_source = data_cfg.get("reasoning_text_source", "thinking")
    field_variants = {
        "deepseek": {"thinking": "deepseek_thinking_trajectory", "attempt": "deepseek_attempt"},
        "gemini": {"thinking": "gemini_thinking_trajectory", "attempt": "gemini_attempt"},
    }
    for source in priority:
        variants = field_variants.get(source, {})
        order = ["attempt", "thinking"] if text_source == "attempt" else ["thinking", "attempt"]
        for variant in order:
            field = variants.get(variant)
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
    lower = sentence.lower()
    score = 0.0
    answer_norm = normalize_for_match(answer)
    sentence_norm = normalize_for_match(sentence)
    if answer_norm and len(answer_norm) >= 2 and answer_norm in sentence_norm:
        score += 8.0
    if any(cue in lower for cue in ANSWER_CUES):
        score += 5.0
    if any(cue in lower for cue in REASONING_CUES):
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
    if not selected_indices:
        return ""
    selected = sorted(set(selected_indices))
    out = []
    prev = None
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
    sentence_tokens = [token_count(tokenizer, s) for s in sentences]
    scores = [sentence_score(s, answer, i, n) for i, s in enumerate(sentences)]

    first_k = int(crop_cfg.get("keep_first_sentences", 2))
    last_k = int(crop_cfg.get("keep_last_sentences", 2))
    anchor_k = int(crop_cfg.get("keep_anchor_sentences", 5))
    stride_k = int(crop_cfg.get("max_stride_sentences", 6))

    priorities: Dict[int, Tuple[float, str]] = {}

    def add_priority(idx: int, priority: float, tag: str):
        if 0 <= idx < n:
            old = priorities.get(idx)
            if old is None or priority > old[0]:
                priorities[idx] = (priority, tag)

    for i in range(min(first_k, n)):
        add_priority(i, 100.0 + scores[i], "first")
    for i in range(max(0, n - last_k), n):
        add_priority(i, 95.0 + scores[i], "last")

    top_indices = sorted(range(n), key=lambda i: (scores[i], -abs(i - n)), reverse=True)
    for i in top_indices[:anchor_k]:
        add_priority(i, 85.0 + scores[i], "anchor")

    if stride_k > 0 and n > first_k + last_k:
        for j in range(1, stride_k + 1):
            i = round(j * (n - 1) / (stride_k + 1))
            add_priority(i, 40.0 + scores[i], "stride")

    for i in top_indices:
        add_priority(i, 10.0 + scores[i], "filler")

    ordered = sorted(((p, i, tag) for i, (p, tag) in priorities.items()), reverse=True)
    selected: List[int] = []
    selected_tags: Dict[int, str] = {}

    for _, idx, tag in ordered:
        if sentence_tokens[idx] > budget:
            continue
        candidate = sorted(set(selected + [idx]))
        candidate_text = assemble_sentences(sentences, candidate)
        if token_count(tokenizer, candidate_text) <= budget:
            selected = candidate
            selected_tags[idx] = tag

    if not selected:
        for idx in sorted(range(n), key=lambda i: sentence_tokens[i]):
            if sentence_tokens[idx] <= budget:
                selected = [idx]
                selected_tags[idx] = "shortest_fallback"
                break

    assembled = assemble_sentences(sentences, selected)
    method = "sentence_anchor_crop" if assembled else "crop_failed"
    meta = {
        "budget_tokens": budget,
        "cropped_tokens": token_count(tokenizer, assembled) if assembled else 0,
        "num_sentences": n,
        "selected_indices": selected,
        "selected_tags": {str(k): v for k, v in selected_tags.items()},
    }
    return assembled, method, meta


def source_char_stats(rows) -> dict:
    fields = {
        "deepseek_thinking": "deepseek_thinking_trajectory",
        "gemini_thinking": "gemini_thinking_trajectory",
        "deepseek_attempt": "deepseek_attempt",
        "gemini_attempt": "gemini_attempt",
        "solution": "solution",
    }
    stats = {}
    for name, field in fields.items():
        vals = [len(clean(row.get(field))) for row in rows if clean(row.get(field))]
        stats[name] = {"field": field, "nonempty": len(vals), "char_lengths": summary(vals)}
    return stats


def process_and_split(config: dict) -> dict:
    data_cfg = dict(config.get("data", {}) or {})
    model_cfg = config.get("model", {})
    raw_output_dir = Path(data_cfg.get("raw_output_dir") or DEFAULT_RAW_OUTPUT_DIR)
    output_dir = Path(data_cfg.get("light_output_dir") or DEFAULT_LIGHT_OUTPUT_DIR)
    max_length = int(model_cfg.get("max_length", 1024))

    answer_fraction = float(data_cfg.get("max_answer_fraction_of_max_length", 0.33))
    data_cfg["_effective_max_answer_tokens"] = max(1, int(max_length * answer_fraction))
    priority = data_cfg.get("reasoning_source_priority", ["deepseek", "gemini"])
    crop_cfg = data_cfg.get("reasoning_crop", {}) or {}

    print("[1/5] Loading tokenizer and S1K data...", flush=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e9)
    raw_rows = list(load_s1k(data_cfg))
    print(f"Loaded {len(raw_rows)} raw rows.", flush=True)

    raw_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing original S1K rows to {raw_output_dir} ...", flush=True)
    write_jsonl(raw_output_dir / "raw.jsonl", (dict(row) for row in tqdm(raw_rows, desc="write raw S1K", dynamic_ncols=True)))
    write_json(raw_output_dir / "manifest.json", {
        "format": "Original S1K rows saved before answer cleaning and reasoning compression.",
        "source_dataset": data_cfg.get("source_dataset", "simplescaling/s1K-1.1"),
        "raw_rows": len(raw_rows),
        "output_file": str(raw_output_dir / "raw.jsonl"),
        "source_char_stats": source_char_stats(raw_rows),
    })

    print("[2/5] Cleaning/extracting answers first, then verifying reasoning-answer match...", flush=True)
    candidates = []
    counters = {
        "raw_rows": len(raw_rows),
        "missing_question": 0,
        "missing_reasoning": 0,
        "missing_answer": 0,
        "reasoning_answer_mismatch": 0,
        "usable_after_answer_cleaning": 0,
        "full_within_reasoning_budget": 0,
        "need_reasoning_compression": 0,
        "force_reasoning_compression": 0,
        "compressed_success": 0,
        "compression_failed": 0,
        "post_process_rows": 0,
    }
    answer_extract_counts: Dict[str, int] = {}
    answer_match_counts: Dict[str, int] = {}
    answer_field_counts: Dict[str, int] = {}
    reasoning_source_counts: Dict[str, int] = {}
    reasoning_field_counts: Dict[str, int] = {}
    skipped_examples: Dict[str, List[dict]] = {"missing_answer": [], "reasoning_answer_mismatch": []}

    answer_field = data_cfg.get("answer_field", "solution")
    for idx, row in enumerate(tqdm(raw_rows, desc="step1 answer clean + reasoning verify", dynamic_ncols=True)):
        row = dict(row)
        question = clean(row.get("question"))
        if not question:
            counters["missing_question"] += 1
            continue
        source, reasoning_field, raw_reasoning = choose_reasoning(row, priority, data_cfg)
        if not raw_reasoning:
            counters["missing_reasoning"] += 1
            continue

        answer, answer_method, answer_meta = extract_clean_answer(row.get(answer_field), tokenizer, data_cfg, raw_reasoning)
        if not answer:
            counters["missing_answer"] += 1
            if len(skipped_examples["missing_answer"]) < 20:
                skipped_examples["missing_answer"].append({
                    "source_index": idx,
                    "question_preview": question[:160],
                    "solution_preview": clean(row.get(answer_field))[:240],
                    "answer_method": answer_method,
                    "answer_meta": answer_meta,
                })
            continue

        ok_match, match_method = answer_matches_reasoning(answer, raw_reasoning, data_cfg)
        if not ok_match:
            counters["reasoning_answer_mismatch"] += 1
            if len(skipped_examples["reasoning_answer_mismatch"]) < 20:
                skipped_examples["reasoning_answer_mismatch"].append({
                    "source_index": idx,
                    "question_preview": question[:160],
                    "answer": answer,
                    "answer_method": answer_method,
                    "match_method": match_method,
                    "reasoning_tail_preview": raw_reasoning[-320:],
                })
            continue

        candidates.append({
            "id": f"s1k_{idx}",
            "source_index": idx,
            "question": question,
            "answer": answer,
            "raw_reasoning": raw_reasoning,
            "reasoning_source": source,
            "reasoning_field": reasoning_field,
            "answer_field": answer_field,
            "answer_extract_method": answer_method,
            "answer_extract_meta": answer_meta,
            "answer_match_method": match_method,
        })
        counters["usable_after_answer_cleaning"] += 1
        answer_extract_counts[answer_method] = answer_extract_counts.get(answer_method, 0) + 1
        answer_match_counts[match_method] = answer_match_counts.get(match_method, 0) + 1
        answer_field_counts[answer_field] = answer_field_counts.get(answer_field, 0) + 1
        reasoning_source_counts[source] = reasoning_source_counts.get(source, 0) + 1
        reasoning_field_counts[reasoning_field] = reasoning_field_counts.get(reasoning_field, 0) + 1

    print("[3/5] Computing fixed prefix/suffix length and reasoning-only budgets...", flush=True)
    within, need_crop = [], []
    margin = int(crop_cfg.get("safety_margin_tokens", 8))
    force_crop = bool(crop_cfg.get("force", True))
    max_reasoning_tokens = int(crop_cfg.get("max_reasoning_tokens", 384))

    for item in tqdm(candidates, desc="step2 fixed length + reasoning budget", dynamic_ncols=True):
        raw_reasoning = item["raw_reasoning"]
        fixed_tokens = token_count(tokenizer, q_text_with_empty_reasoning(item["question"], item["answer"]))
        available_reasoning_budget = max_length - fixed_tokens - margin
        reasoning_budget = available_reasoning_budget
        if force_crop and max_reasoning_tokens > 0:
            reasoning_budget = min(available_reasoning_budget, max_reasoning_tokens)

        item["fixed_tokens_without_reasoning"] = fixed_tokens
        item["available_reasoning_budget_tokens"] = available_reasoning_budget
        item["compression_budget_tokens"] = reasoning_budget
        item["reasoning_full_tokens"] = token_count(tokenizer, raw_reasoning)
        item["question_tokens"] = token_count(tokenizer, item["question"])
        item["answer_tokens"] = token_count(tokenizer, item["answer"])
        item["original_qar_total_tokens_est"] = fixed_tokens + item["reasoning_full_tokens"]

        if available_reasoning_budget <= 0:
            item["compression_reason"] = "no_reasoning_budget_after_fixed_parts"
            need_crop.append(item)
            counters["need_reasoning_compression"] += 1
            continue

        needs_crop_for_context = item["reasoning_full_tokens"] > available_reasoning_budget
        needs_crop_for_light_reasoning = force_crop and item["reasoning_full_tokens"] > reasoning_budget

        if not needs_crop_for_context and not needs_crop_for_light_reasoning:
            final_total = token_count(tokenizer, qar_text(item["question"], raw_reasoning, item["answer"]))
            if final_total <= max_length:
                item["reasoning"] = raw_reasoning
                item["reasoning_processing"] = {
                    "method": "full_reasoning_within_budget",
                    "fixed_tokens_without_reasoning": fixed_tokens,
                    "available_reasoning_budget_tokens": available_reasoning_budget,
                    "reasoning_budget_tokens": reasoning_budget,
                    "original_total_tokens_est": item["original_qar_total_tokens_est"],
                    "final_total_tokens": final_total,
                    "reasoning_tokens": item["reasoning_full_tokens"],
                    "force_crop": force_crop,
                }
                within.append(item)
                counters["full_within_reasoning_budget"] += 1
                continue

        item["compression_reason"] = "over_context_budget" if needs_crop_for_context else "over_light_reasoning_budget"
        need_crop.append(item)
        counters["need_reasoning_compression"] += 1
        if needs_crop_for_light_reasoning and not needs_crop_for_context:
            counters["force_reasoning_compression"] += 1

    print(f"[4/5] Compressing reasoning for {len(need_crop)} rows by complete sentences...", flush=True)
    processed = list(within)
    crop_method_counts: Dict[str, int] = {"full_reasoning_within_budget": len(within)}
    for item in tqdm(need_crop, desc="step3 reasoning-only sentence crop", dynamic_ncols=True):
        budget = item.get("compression_budget_tokens", 0)
        cropped, method, meta = compress_reasoning_to_budget(
            item["raw_reasoning"], item["answer"], tokenizer, budget, crop_cfg
        ) if bool(crop_cfg.get("enabled", True)) else ("", "crop_disabled", {"budget_tokens": budget})

        final_total = token_count(tokenizer, qar_text(item["question"], cropped, item["answer"])) if cropped else item.get("fixed_tokens_without_reasoning", 0)
        if not clean(cropped) or final_total > max_length:
            counters["compression_failed"] += 1
            crop_method_counts[method] = crop_method_counts.get(method, 0) + 1
            continue

        item["reasoning"] = cropped
        item["reasoning_processing"] = {
            "method": method,
            "compression_reason": item.get("compression_reason"),
            "fixed_tokens_without_reasoning": item.get("fixed_tokens_without_reasoning"),
            "available_reasoning_budget_tokens": item.get("available_reasoning_budget_tokens"),
            "original_total_tokens_est": item.get("original_qar_total_tokens_est"),
            "final_total_tokens": final_total,
            "reasoning_tokens": token_count(tokenizer, cropped),
            "reasoning_budget_tokens": budget,
            **meta,
        }
        processed.append(item)
        counters["compressed_success"] += 1
        crop_method_counts[method] = crop_method_counts.get(method, 0) + 1

    processed = sorted(processed, key=lambda x: int(x.get("source_index", 10**12)))

    print("[5/5] Splitting processed rows and writing JSONL files...", flush=True)
    valid_indices, test_indices = split_indices(
        len(processed),
        float(data_cfg.get("valid_ratio", 0.1)),
        float(data_cfg.get("test_ratio", 0.1)),
        int(data_cfg.get("seed", 42)),
    )
    split_rows = {"train": [], "validation": [], "test": []}
    for processed_idx, item in enumerate(tqdm(processed, desc="step4 split/write prepare", dynamic_ncols=True)):
        split = "validation" if processed_idx in valid_indices else "test" if processed_idx in test_indices else "train"
        item["split"] = split
        item["processed_index"] = processed_idx
        reasoning = clean(item["reasoning"])
        out = {
            "id": item["id"],
            "source_index": item["source_index"],
            "processed_index": processed_idx,
            "split": split,
            "question": item["question"],
            "answer": item["answer"],
            "reasoning": reasoning,
            "reasoning_full": item["raw_reasoning"],
            "reasoning_source": item["reasoning_source"],
            "reasoning_field": item["reasoning_field"],
            "answer_field": item["answer_field"],
            "answer_extract_method": item["answer_extract_method"],
            "answer_extract_meta": item["answer_extract_meta"],
            "answer_match_method": item["answer_match_method"],
            "reasoning_processing": item["reasoning_processing"],
            "token_stats": {
                "question_tokens": item["question_tokens"],
                "answer_tokens": item["answer_tokens"],
                "reasoning_tokens": token_count(tokenizer, reasoning),
                "reasoning_full_tokens": item["reasoning_full_tokens"],
                "qar_total_tokens": token_count(tokenizer, qar_text(item["question"], reasoning, item["answer"])),
            },
        }
        split_rows[split].append(out)

    counters["post_process_rows"] = len(processed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in split_rows.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)

    all_list = [row for rows in split_rows.values() for row in rows]
    manifest = {
        "format": "S1K_light rows after answer cleaning, reasoning-answer verification, reasoning-only complete-sentence compression, then split. No train masks here.",
        "source_dataset": data_cfg.get("source_dataset", "simplescaling/s1K-1.1"),
        "raw_output_dir": str(raw_output_dir),
        "light_output_dir": str(output_dir),
        "max_length": max_length,
        "split_order": "clean_answer_from_solution -> verify_reasoning_contains_answer -> compute_fixed_answer_and_reasoning_budget -> compress_reasoning_only -> split",
        "light_rows": {split: len(rows) for split, rows in split_rows.items()},
        "counters": counters,
        "reasoning_source_priority": priority,
        "reasoning_text_source": data_cfg.get("reasoning_text_source", "thinking"),
        "reasoning_source_counts": reasoning_source_counts,
        "reasoning_field_counts": reasoning_field_counts,
        "answer_field_counts": answer_field_counts,
        "answer_extract_counts": answer_extract_counts,
        "answer_match_counts": answer_match_counts,
        "answer_config": {
            "max_answer_chars": data_cfg.get("max_answer_chars", 1200),
            "max_answer_tokens": data_cfg.get("max_answer_tokens", 256),
            "max_answer_fraction_of_max_length": data_cfg.get("max_answer_fraction_of_max_length", 0.33),
            "effective_max_answer_tokens": data_cfg.get("_effective_max_answer_tokens"),
            "require_answer_in_reasoning": data_cfg.get("require_answer_in_reasoning", True),
            "max_answer_chars_for_reasoning_match": data_cfg.get("max_answer_chars_for_reasoning_match", 200),
            "question_deduplication": "disabled_by_default; source_index is preserved",
        },
        "crop_config": crop_cfg,
        "crop_method_counts": crop_method_counts,
        "skipped_examples": skipped_examples,
        "light_token_stats": {
            "question_tokens": summary(row["token_stats"]["question_tokens"] for row in all_list),
            "answer_tokens": summary(row["token_stats"]["answer_tokens"] for row in all_list),
            "reasoning_tokens": summary(row["token_stats"]["reasoning_tokens"] for row in all_list),
            "reasoning_full_tokens": summary(row["token_stats"]["reasoning_full_tokens"] for row in all_list),
            "qar_total_tokens": summary(row["token_stats"]["qar_total_tokens"] for row in all_list),
        },
        "source_char_stats_before_processing": source_char_stats(raw_rows),
        "note": "Answer is cleaned first and should contain only the final answer, not explanation/reasoning. Only the reasoning body is compressed; gaps are represented by '...'.",
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Prepare data/S1K and data/S1K_light splits for SEDD SFT pipelines.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Config file to read data/model options from.")
    args = parser.parse_args()
    process_and_split(load_config(args.config))


if __name__ == "__main__":
    main()
