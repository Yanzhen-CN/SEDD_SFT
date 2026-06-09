"""
Root-level S1K split/data preparation for SEDD SFT pipelines.

This file does the shared, expensive data work once, in the order requested:
  1) filter rows by extracting a short final answer from `solution`
  2) compute QAR total token length and find rows that need compression
  3) compress only overlength reasoning, using complete sentence-like units only
  4) split the processed rows into train / validation / test

Output files:
  data/S1K/raw.jsonl and data/S1K/manifest.json contain original S1K rows.
  data/S1K_light/train.jsonl, validation.jsonl, test.jsonl contain processed light rows with no train masks.
  data/S1K_light/manifest.json contains processing/split statistics.

Pipeline-specific prepare files read these S1K_light rows and only create segment masks:
  QA:  question -> answer
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
    return len(tokenizer(clean(text), add_special_tokens=False).input_ids)


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
    """Split into complete sentence-like units; never returns half-token crops."""
    text = clean(text)
    if not text:
        return []
    text = re.sub(r"\r\n?", "\n", text)
    units: List[str] = []
    for block in re.split(r"\n{2,}", text):
        block = clean(block)
        if not block:
            continue
        # First split on strong sentence boundaries. For math derivations, keep lines complete.
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
    # Remove exact duplicates while keeping order. This reduces repeated model traces.
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


def looks_like_short_answer(text, max_chars=320, max_tokens=None, tokenizer=None) -> bool:
    text = clean(text)
    if not text:
        return False
    if len(text) > max_chars:
        return False
    if max_tokens is not None and tokenizer is not None and token_count(tokenizer, text) > max_tokens:
        return False
    scalar_patterns = [
        r"^[+-]?\d+(?:\.\d+)?(?:/\d+)?(?:\s*[%°])?$",
        r"^[A-E]$",
        r"^(yes|no|true|false)$",
        r"^\\boxed\s*\{.+\}$",
        r"^[A-Za-z0-9_\\{}^+\-*/=().,\s]+$",
    ]
    if any(re.fullmatch(p, text, flags=re.IGNORECASE | re.DOTALL) for p in scalar_patterns):
        return True
    return len(text.split()) <= 40


def extract_short_answer(solution, tokenizer, data_cfg) -> Tuple[str, str]:
    """Extract concise final answer from solution; never use a long solution as Answer."""
    solution = clean(solution)
    if not solution:
        return "", "empty_solution"

    max_chars = int(data_cfg.get("max_answer_chars", 320))
    max_tokens = int(data_cfg.get("max_answer_tokens", 64))
    candidates: List[Tuple[str, str]] = []

    # Most reliable patterns first.
    for m in re.finditer(r"\\boxed\s*\{([^{}]+)\}", solution, flags=re.DOTALL):
        candidates.append((clean("\\boxed{" + m.group(1) + "}"), "boxed"))
    for m in re.finditer(r"####\s*([^\n]+)", solution):
        candidates.append((clean(m.group(1)), "gsm8k_hash"))

    marker_patterns = [
        r"(?i)(?:final\s+answer|the\s+answer\s+is|answer\s+is)\s*[:：]?\s*([^\n]+)",
        r"(?i)(?:therefore|thus|hence|so),?\s+(?:the\s+)?answer\s+is\s*([^\n]+)",
    ]
    for pattern in marker_patterns:
        for m in re.finditer(pattern, solution):
            candidates.append((clean(m.group(1)), "explicit_answer_marker"))

    lines = [clean(x) for x in solution.splitlines() if clean(x)]
    for line in reversed(lines[-5:]):
        candidates.append((line, "last_nonempty_line"))

    sentences = split_sentences(solution)
    for sent in reversed(sentences[-5:]):
        candidates.append((sent, "last_sentence"))

    tail = solution[-1200:]
    expr_patterns = [
        r"\\boxed\s*\{[^{}]+\}",
        r"[A-Za-z]?\s*=\s*[+-]?\d+(?:\.\d+)?(?:/\d+)?",
        r"[+-]?\d+(?:\.\d+)?(?:/\d+)?(?:\s*[%°])?",
    ]
    for pattern in expr_patterns:
        matches = re.findall(pattern, tail)
        for val in matches[-3:]:
            candidates.append((clean(val), "tail_numeric_or_formula"))

    seen = set()
    for candidate, method in candidates:
        candidate = clean(candidate).strip(" .。")
        candidate = _strip_boxed(candidate) if method != "boxed" else candidate
        key = candidate.lower()
        if not candidate or key in seen:
            continue
        seen.add(key)
        if looks_like_short_answer(candidate, max_chars=max_chars, max_tokens=max_tokens, tokenizer=tokenizer):
            return candidate, method

    if looks_like_short_answer(solution, max_chars=max_chars, max_tokens=max_tokens, tokenizer=tokenizer):
        return solution, "whole_short_solution"
    return "", "long_or_unextractable_answer"


def choose_reasoning(row, priority, data_cfg) -> Tuple[Optional[str], Optional[str], str]:
    """DeepSeek stays first by default; Gemini is only fallback unless config changes."""
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


def normalize_for_match(text) -> str:
    text = clean(text).lower()
    text = re.sub(r"\\boxed\s*\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"[^a-z0-9.+\-*/=]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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
    """Sentence-level compression. It never cuts inside a sentence-like unit."""
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

    # Extra filler candidates only if there is still budget.
    for i in top_indices:
        add_priority(i, 10.0 + scores[i], "filler")

    ordered = sorted(((p, i, tag) for i, (p, tag) in priorities.items()), reverse=True)
    selected: List[int] = []
    selected_tags: Dict[int, str] = {}

    for _, idx, tag in ordered:
        # Never cut a long sentence. If one sentence is too long, skip it.
        if sentence_tokens[idx] > budget:
            continue
        candidate = sorted(set(selected + [idx]))
        candidate_text = assemble_sentences(sentences, candidate)
        if token_count(tokenizer, candidate_text) <= budget:
            selected = candidate
            selected_tags[idx] = tag

    # Fallback: choose the shortest complete sentence that fits.
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
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    raw_output_dir = Path(data_cfg.get("raw_output_dir") or DEFAULT_RAW_OUTPUT_DIR)
    output_dir = Path(
        data_cfg.get("light_output_dir")
        or data_cfg.get("base_output_dir")
        or data_cfg.get("base_dir")
        or DEFAULT_LIGHT_OUTPUT_DIR
    )
    max_length = int(model_cfg.get("max_length", 1024))
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
        "format": "Original S1K rows saved before short-answer filtering and reasoning compression.",
        "source_dataset": data_cfg.get("source_dataset", "simplescaling/s1K-1.1"),
        "raw_rows": len(raw_rows),
        "output_file": str(raw_output_dir / "raw.jsonl"),
        "source_char_stats": source_char_stats(raw_rows),
    })

    # Step 1: filter by solution-derived short answer and required fields.
    print("[2/5] Filtering by short answer and required fields...", flush=True)
    candidates = []
    counters = {
        "raw_rows": len(raw_rows),
        "missing_question": 0,
        "missing_reasoning": 0,
        "missing_answer": 0,
        "usable_after_answer_filter": 0,
        "full_within_length": 0,
        "need_compression": 0,
        "compressed_success": 0,
        "compression_failed": 0,
        "post_process_rows": 0,
    }
    answer_extract_counts: Dict[str, int] = {}
    answer_field_counts: Dict[str, int] = {}
    reasoning_source_counts: Dict[str, int] = {}
    reasoning_field_counts: Dict[str, int] = {}

    answer_field = data_cfg.get("answer_field", "solution")
    for idx, row in enumerate(tqdm(raw_rows, desc="step1 answer filter", dynamic_ncols=True)):
        row = dict(row)
        question = clean(row.get("question"))
        if not question:
            counters["missing_question"] += 1
            continue
        source, reasoning_field, raw_reasoning = choose_reasoning(row, priority, data_cfg)
        if not raw_reasoning:
            counters["missing_reasoning"] += 1
            continue
        answer, answer_method = extract_short_answer(row.get(answer_field), tokenizer, data_cfg)
        if not answer:
            counters["missing_answer"] += 1
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
        })
        counters["usable_after_answer_filter"] += 1
        answer_extract_counts[answer_method] = answer_extract_counts.get(answer_method, 0) + 1
        answer_field_counts[answer_field] = answer_field_counts.get(answer_field, 0) + 1
        reasoning_source_counts[source] = reasoning_source_counts.get(source, 0) + 1
        reasoning_field_counts[reasoning_field] = reasoning_field_counts.get(reasoning_field, 0) + 1

    # Step 2: compute total length once and separate rows needing compression.
    print("[3/5] Computing QAR total token lengths and marking rows that need compression...", flush=True)
    within, need_crop = [], []
    for item in tqdm(candidates, desc="step2 length check", dynamic_ncols=True):
        raw_reasoning = item["raw_reasoning"]
        full_total = token_count(tokenizer, qar_text(item["question"], raw_reasoning, item["answer"]))
        item["original_qar_total_tokens"] = full_total
        item["reasoning_full_tokens"] = token_count(tokenizer, raw_reasoning)
        item["question_tokens"] = token_count(tokenizer, item["question"])
        item["answer_tokens"] = token_count(tokenizer, item["answer"])
        if full_total <= max_length:
            item["reasoning"] = raw_reasoning
            item["reasoning_processing"] = {
                "method": "full_within_sample_budget",
                "original_total_tokens": full_total,
                "final_total_tokens": full_total,
                "reasoning_tokens": item["reasoning_full_tokens"],
            }
            within.append(item)
            counters["full_within_length"] += 1
        else:
            need_crop.append(item)
            counters["need_compression"] += 1

    # Step 3: compress overlength rows only.
    print(f"[4/5] Compressing {len(need_crop)} overlength rows by complete sentences...", flush=True)
    processed = list(within)
    crop_method_counts: Dict[str, int] = {"full_within_sample_budget": len(within)}
    margin = int(crop_cfg.get("safety_margin_tokens", 8))
    for item in tqdm(need_crop, desc="step3 sentence crop", dynamic_ncols=True):
        fixed_tokens = token_count(tokenizer, q_text_with_empty_reasoning(item["question"], item["answer"]))
        budget = max_length - fixed_tokens - margin
        cropped, method, meta = compress_reasoning_to_budget(
            item["raw_reasoning"], item["answer"], tokenizer, budget, crop_cfg
        ) if bool(crop_cfg.get("enabled", True)) else ("", "crop_disabled_overlength", {"budget_tokens": budget})

        final_total = token_count(tokenizer, qar_text(item["question"], cropped, item["answer"])) if cropped else fixed_tokens
        if not clean(cropped) or final_total > max_length:
            counters["compression_failed"] += 1
            crop_method_counts[method] = crop_method_counts.get(method, 0) + 1
            continue

        item["reasoning"] = cropped
        item["reasoning_processing"] = {
            "method": method,
            "original_total_tokens": item["original_qar_total_tokens"],
            "final_total_tokens": final_total,
            "reasoning_tokens": token_count(tokenizer, cropped),
            **meta,
        }
        processed.append(item)
        counters["compressed_success"] += 1
        crop_method_counts[method] = crop_method_counts.get(method, 0) + 1

    # Step 4: split after all filtering / compression.
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

    def all_items():
        for rows in split_rows.values():
            for row in rows:
                yield row

    all_list = list(all_items())
    manifest = {
        "format": "S1K_light content rows after short-answer filtering, length check, complete-sentence compression, then split. No train masks here.",
        "source_dataset": data_cfg.get("source_dataset", "simplescaling/s1K-1.1"),
        "raw_output_dir": str(raw_output_dir),
        "light_output_dir": str(output_dir),
        "max_length": max_length,
        "split_order": "filter_by_solution -> length_check -> compress_overlength -> split",
        "light_rows": {split: len(rows) for split, rows in split_rows.items()},
        "base_rows": {split: len(rows) for split, rows in split_rows.items()},  # backward compatibility
        "counters": counters,
        "reasoning_source_priority": priority,
        "reasoning_text_source": data_cfg.get("reasoning_text_source", "thinking"),
        "reasoning_source_counts": reasoning_source_counts,
        "reasoning_field_counts": reasoning_field_counts,
        "answer_field_counts": answer_field_counts,
        "answer_extract_counts": answer_extract_counts,
        "crop_config": crop_cfg,
        "crop_method_counts": crop_method_counts,
        "light_token_stats": {
            "question_tokens": summary(row["token_stats"]["question_tokens"] for row in all_list),
            "answer_tokens": summary(row["token_stats"]["answer_tokens"] for row in all_list),
            "reasoning_tokens": summary(row["token_stats"]["reasoning_tokens"] for row in all_list),
            "reasoning_full_tokens": summary(row["token_stats"]["reasoning_full_tokens"] for row in all_list),
            "qar_total_tokens": summary(row["token_stats"]["qar_total_tokens"] for row in all_list),
        },
        "source_char_stats_before_processing": source_char_stats(raw_rows),
        "note": "Only overlength rows are compressed. Compression uses complete sentence-like units; gaps are represented by '...'.",
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Prepare root-level data/S1K and data/S1K_light splits for SEDD SFT pipelines.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Config file to read data/model options from.")
    args = parser.parse_args()
    process_and_split(load_config(args.config))


if __name__ == "__main__":
    main()
