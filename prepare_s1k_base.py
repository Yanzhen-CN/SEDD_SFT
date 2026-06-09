"""
Root-level S1K base data preparation for SEDD SFT pipelines.

This file does the expensive / duplicated content processing exactly once:
  1) load S1K / S1K-1.1
  2) create or reuse train/validation/test split
  3) keep DeepSeek first by default, but report DS/Gemini length stats
  4) extract a short final answer from solution
  5) if full reasoning does not fit the SEDD context, crop it by COMPLETE
     sentences only: first sentences + answer/keyword/math anchor sentences +
     last sentences + stride sentences, joined with ellipses
  6) write base content JSONL files with no train masks

The pipeline-specific prepare files should only convert these base rows into
segments and train masks.
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml
from transformers import GPT2TokenizerFast

REPO_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_DIR / "sft_answer_pipeline" / "answer_config.yaml"
DEFAULT_OUTPUT_DIR = REPO_DIR / "data" / "s1k_base"
ELLIPSIS = "\n...\n"

QA_SEGMENT_ORDER = ["user_label", "user", "assistant_label", "answer_label", "answer"]
QAR_SEGMENT_ORDER = [
    "user_label",
    "user",
    "assistant_label",
    "reasoning_label",
    "reasoning",
    "answer_label",
    "answer",
]
QRA_SEGMENT_ORDER = QAR_SEGMENT_ORDER

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
    "final answer", "the answer is", "answer is", "therefore the answer", "so the answer",
    "hence the answer", "we get", "we obtain", "equals", "is equal to", "\\boxed",
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


def read_jsonl(path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_s1k(data_cfg):
    if data_cfg.get("arrow_path"):
        from datasets import Dataset

        return Dataset.from_file(data_cfg["arrow_path"])

    from datasets import load_dataset

    return load_dataset(data_cfg.get("source_dataset", "simplescaling/s1K-1.1"), split="train")


def load_existing_splits(split_dir: Optional[str]):
    if not split_dir:
        return None
    base = Path(split_dir)
    if not base.exists():
        return None
    splits = {}
    for split in ("train", "validation", "test"):
        path = base / f"{split}.jsonl"
        if not path.exists():
            return None
        splits[split] = read_jsonl(path)
    return splits


def split_indices(n_rows, valid_ratio, test_ratio, seed):
    indices = list(range(n_rows))
    random.Random(seed).shuffle(indices)
    n_valid = max(1, int(n_rows * valid_ratio)) if n_rows > 1 and valid_ratio > 0 else 0
    n_test = max(1, int(n_rows * test_ratio)) if n_rows > 1 and test_ratio > 0 else 0
    if n_valid + n_test >= n_rows:
        raise ValueError("valid_ratio + test_ratio leaves no training data.")
    return set(indices[:n_valid]), set(indices[n_valid : n_valid + n_test])


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


def collect_source_length_stats(rows, tokenizer) -> dict:
    fields = {
        "deepseek_thinking": "deepseek_thinking_trajectory",
        "gemini_thinking": "gemini_thinking_trajectory",
        "deepseek_attempt": "deepseek_attempt",
        "gemini_attempt": "gemini_attempt",
        "solution": "solution",
    }
    stats = {}
    for name, field in fields.items():
        char_lengths, token_lengths = [], []
        nonempty = 0
        for row in rows:
            text = clean(row.get(field))
            if not text:
                continue
            nonempty += 1
            char_lengths.append(len(text))
            token_lengths.append(token_count(tokenizer, text))
        stats[name] = {
            "field": field,
            "nonempty": nonempty,
            "char_lengths": summary(char_lengths),
            "token_lengths": summary(token_lengths),
        }
    return stats


def split_sentences(text) -> List[str]:
    """Split into complete sentence-ish units.

    We intentionally keep complete lines/equations. We never crop inside a unit;
    if a unit itself is too long for the remaining budget, it is skipped.
    """
    text = clean(text)
    if not text:
        return []
    text = re.sub(r"\r\n?", "\n", text)
    chunks = []
    for block in re.split(r"\n{2,}", text):
        block = clean(block)
        if not block:
            continue
        pieces = re.split(r"(?<=[。！？!?])\s+|(?<=[.!?])\s+(?=[A-Z0-9\\\(\[])", block)
        for piece in pieces:
            piece = clean(piece)
            if not piece:
                continue
            # Keep equations / bullets / code-like derivations as complete lines.
            for sub in piece.split("\n"):
                sub = clean(sub)
                if sub:
                    chunks.append(sub)
    return chunks


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
    """Extract concise final answer from solution.

    Long solution text is never used directly as the Answer section.
    """
    solution = clean(solution)
    if not solution:
        return "", "empty_solution"

    max_chars = int(data_cfg.get("max_answer_chars", 320))
    max_tokens = int(data_cfg.get("max_answer_tokens", 64))
    candidates: List[Tuple[str, str]] = []

    for m in re.finditer(r"\\boxed\s*\{([^{}]+)\}", solution, flags=re.DOTALL):
        candidates.append((clean("\\boxed{" + m.group(1) + "}"), "boxed"))
    for m in re.finditer(r"####\s*([^\n]+)", solution):
        candidates.append((clean(m.group(1)), "gsm8k_hash"))

    marker_patterns = [
        r"(?i)(?:final\s+answer|the\s+answer\s+is|answer\s+is|answer)\s*[:：]?\s*([^\n]+)",
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
    """Choose teacher reasoning source. DeepSeek stays first by default."""
    text_source = data_cfg.get("reasoning_text_source", "thinking")
    field_variants = {
        "deepseek": {"thinking": "deepseek_thinking_trajectory", "attempt": "deepseek_attempt"},
        "gemini": {"thinking": "gemini_thinking_trajectory", "attempt": "gemini_attempt"},
    }
    for source in priority:
        variants = field_variants.get(source, {})
        if text_source == "attempt":
            order = ["attempt", "thinking"]
        elif text_source == "auto":
            order = ["thinking", "attempt"]
        else:
            order = ["thinking", "attempt"]
        for variant in order:
            field = variants.get(variant)
            text = clean(row.get(field)) if field else ""
            if text:
                return source, field, text
    return None, None, ""


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
            out.append(ELLIPSIS.strip())
        out.append(clean(sentences[idx]))
        prev = idx
    return "\n".join(x for x in out if x)


def compress_reasoning_to_budget(reasoning, answer, tokenizer, budget, crop_cfg):
    reasoning = clean(reasoning)
    if budget <= 0:
        return "", "no_budget", {"budget_tokens": budget}

    original_tokens = token_count(tokenizer, reasoning)
    if original_tokens <= budget:
        return reasoning, "full_within_budget", {
            "budget_tokens": budget,
            "original_tokens": original_tokens,
            "selected_sentences": None,
        }

    sentences = split_sentences(reasoning)
    if not sentences:
        return "", "no_sentences", {"budget_tokens": budget, "original_tokens": original_tokens}

    n = len(sentences)
    scored = [(i, sentence_score(sent, answer, i, n)) for i, sent in enumerate(sentences)]
    score_map = {i: s for i, s in scored}

    first_k = int(crop_cfg.get("keep_first_sentences", 2))
    last_k = int(crop_cfg.get("keep_last_sentences", 2))
    anchor_k = int(crop_cfg.get("keep_anchor_sentences", 4))
    stride_k = int(crop_cfg.get("max_stride_sentences", 6))

    priority = []
    for i in range(min(first_k, n)):
        priority.append((100.0 + score_map[i], i, "first"))
    for i in range(max(0, n - last_k), n):
        priority.append((95.0 + score_map[i], i, "last"))

    top_scored = sorted(scored, key=lambda x: (x[1], -abs(x[0] - n)), reverse=True)
    for i, s in top_scored[:anchor_k]:
        priority.append((80.0 + s, i, "anchor"))

    if stride_k > 0 and n > first_k + last_k:
        for j in range(1, stride_k + 1):
            i = round(j * (n - 1) / (stride_k + 1))
            if 0 <= i < n:
                priority.append((20.0 + score_map[i], i, "stride"))

    for i, s in top_scored:
        priority.append((10.0 + s, i, "filler"))

    dedup = {}
    for p, i, tag in priority:
        if i not in dedup or p > dedup[i][0]:
            dedup[i] = (p, tag)
    ordered = sorted(((p, i, tag) for i, (p, tag) in dedup.items()), reverse=True)

    selected, selected_tags = [], {}
    for _, idx, tag in ordered:
        sentence = clean(sentences[idx])
        # Never cut a half sentence; skip oversized sentence instead.
        if token_count(tokenizer, sentence) > budget:
            continue
        candidate = sorted(set(selected + [idx]))
        assembled = assemble_sentences(sentences, candidate)
        if token_count(tokenizer, assembled) <= budget:
            selected = candidate
            selected_tags[idx] = tag

    assembled = assemble_sentences(sentences, selected)
    if not assembled:
        for idx in sorted(range(n), key=lambda i: token_count(tokenizer, sentences[i])):
            sent = clean(sentences[idx])
            if token_count(tokenizer, sent) <= budget:
                assembled = sent
                selected = [idx]
                selected_tags[idx] = "shortest_fallback"
                break

    method = "sentence_anchor_crop" if assembled else "crop_failed"
    meta = {
        "budget_tokens": budget,
        "original_tokens": original_tokens,
        "cropped_tokens": token_count(tokenizer, assembled) if assembled else 0,
        "num_sentences": n,
        "selected_indices": selected,
        "selected_tags": {str(k): v for k, v in selected_tags.items()},
    }
    return assembled, method, meta


def qar_text(question: str, reasoning: str, answer: str) -> str:
    return f"User: {clean(question)}\nAssistant:\nReasoning:\n{clean(reasoning)}\n\nAnswer:\n{clean(answer)}"


def q_text_with_empty_reasoning(question: str, answer: str) -> str:
    return f"User: {clean(question)}\nAssistant:\nReasoning:\n\n\nAnswer:\n{clean(answer)}"


def process_reasoning_for_qar_budget(question, raw_reasoning, answer, tokenizer, max_length, crop_cfg):
    raw_reasoning = clean(raw_reasoning)
    full = qar_text(question, raw_reasoning, answer)
    full_total = token_count(tokenizer, full)
    if full_total <= max_length:
        return raw_reasoning, {
            "method": "full_within_sample_budget",
            "original_total_tokens": full_total,
            "final_total_tokens": full_total,
            "reasoning_tokens": token_count(tokenizer, raw_reasoning),
        }

    margin = int(crop_cfg.get("safety_margin_tokens", 8))
    fixed = token_count(tokenizer, q_text_with_empty_reasoning(question, answer))
    budget = max_length - fixed - margin
    cropped, method, meta = compress_reasoning_to_budget(raw_reasoning, answer, tokenizer, budget, crop_cfg)
    final_total = token_count(tokenizer, qar_text(question, cropped, answer)) if cropped else fixed
    return cropped, {
        "method": method,
        "original_total_tokens": full_total,
        "final_total_tokens": final_total,
        "reasoning_tokens": token_count(tokenizer, cropped) if cropped else 0,
        **meta,
    }


def build_base(config: dict) -> dict:
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    output_dir = Path(data_cfg.get("base_output_dir") or data_cfg.get("base_dir") or DEFAULT_OUTPUT_DIR)
    max_length = int(model_cfg.get("max_length", 1024))
    priority = data_cfg.get("reasoning_source_priority", ["deepseek", "gemini"])
    crop_cfg = data_cfg.get("reasoning_crop", {}) or {}

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e9)

    existing_splits = load_existing_splits(data_cfg.get("split_dir"))
    raw_dataset = load_s1k(data_cfg)
    raw_rows = list(raw_dataset)

    if existing_splits is not None:
        split_rows = existing_splits
        raw_for_stats = [row for rows in existing_splits.values() for row in rows]
        split_source = data_cfg.get("split_dir")
    else:
        valid_indices, test_indices = split_indices(
            len(raw_rows),
            float(data_cfg.get("valid_ratio", 0.1)),
            float(data_cfg.get("test_ratio", 0.1)),
            int(data_cfg.get("seed", 42)),
        )
        split_rows = {"train": [], "validation": [], "test": []}
        for idx, row in enumerate(raw_rows):
            split = "validation" if idx in valid_indices else "test" if idx in test_indices else "train"
            row = dict(row)
            row["source_index"] = idx
            split_rows[split].append(row)
        raw_for_stats = raw_rows
        split_source = "generated_by_prepare_s1k_base"

    out_splits = {"train": [], "validation": [], "test": []}
    counters = {
        "skipped_missing_question": 0,
        "skipped_missing_reasoning": 0,
        "skipped_missing_answer": 0,
        "skipped_crop_failed": 0,
        "usable_rows": 0,
    }
    reasoning_source_counts, reasoning_field_counts = {}, {}
    answer_extract_counts, answer_field_counts = {}, {}
    crop_method_counts = {}

    for split, rows in split_rows.items():
        for local_idx, row in enumerate(rows):
            raw_idx = int(row.get("source_index", local_idx))
            question = clean(row.get("question"))
            source, reasoning_field, raw_reasoning = choose_reasoning(row, priority, data_cfg)
            answer_field = data_cfg.get("answer_field", "solution")
            answer, answer_method = extract_short_answer(row.get(answer_field), tokenizer, data_cfg)

            if not question:
                counters["skipped_missing_question"] += 1
                continue
            if not raw_reasoning:
                counters["skipped_missing_reasoning"] += 1
                continue
            if not answer:
                counters["skipped_missing_answer"] += 1
                continue

            reasoning, proc = process_reasoning_for_qar_budget(
                question, raw_reasoning, answer, tokenizer, max_length, crop_cfg
            ) if bool(crop_cfg.get("enabled", True)) else (
                raw_reasoning,
                {
                    "method": "crop_disabled",
                    "original_total_tokens": token_count(tokenizer, qar_text(question, raw_reasoning, answer)),
                    "final_total_tokens": token_count(tokenizer, qar_text(question, raw_reasoning, answer)),
                    "reasoning_tokens": token_count(tokenizer, raw_reasoning),
                },
            )
            if not clean(reasoning):
                counters["skipped_crop_failed"] += 1
                continue

            row_id = f"s1k_{raw_idx}"
            item = {
                "id": row_id,
                "source_index": raw_idx,
                "split": split,
                "question": question,
                "answer": answer,
                "reasoning": reasoning,
                "reasoning_full": raw_reasoning,
                "reasoning_source": source,
                "reasoning_field": reasoning_field,
                "answer_field": answer_field,
                "answer_extract_method": answer_method,
                "reasoning_processing": proc,
                "token_stats": {
                    "question_tokens": token_count(tokenizer, question),
                    "answer_tokens": token_count(tokenizer, answer),
                    "reasoning_tokens": token_count(tokenizer, reasoning),
                    "reasoning_full_tokens": token_count(tokenizer, raw_reasoning),
                    "qar_total_tokens": token_count(tokenizer, qar_text(question, reasoning, answer)),
                },
            }
            out_splits[split].append(item)
            counters["usable_rows"] += 1
            reasoning_source_counts[source] = reasoning_source_counts.get(source, 0) + 1
            reasoning_field_counts[reasoning_field] = reasoning_field_counts.get(reasoning_field, 0) + 1
            answer_extract_counts[answer_method] = answer_extract_counts.get(answer_method, 0) + 1
            answer_field_counts[answer_field] = answer_field_counts.get(answer_field, 0) + 1
            crop_method_counts[proc.get("method", "unknown")] = crop_method_counts.get(proc.get("method", "unknown"), 0) + 1

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in out_splits.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)

    def _tok_stats(field):
        return summary(item["token_stats"][field] for rows in out_splits.values() for item in rows)

    manifest = {
        "format": "Base S1K content rows. No train masks here; pipeline adapters create QA/QAR/QRA segment masks.",
        "source_dataset": data_cfg.get("source_dataset", "simplescaling/s1K-1.1"),
        "split_source": split_source,
        "output_dir": str(output_dir),
        "raw_rows": len(raw_rows),
        "base_rows": {split: len(rows) for split, rows in out_splits.items()},
        "counters": counters,
        "max_length": max_length,
        "reasoning_source_priority": priority,
        "reasoning_text_source": data_cfg.get("reasoning_text_source", "thinking"),
        "reasoning_source_counts": reasoning_source_counts,
        "reasoning_field_counts": reasoning_field_counts,
        "answer_field_counts": answer_field_counts,
        "answer_extract_counts": answer_extract_counts,
        "crop_config": crop_cfg,
        "crop_method_counts": crop_method_counts,
        "base_token_stats": {
            "question_tokens": _tok_stats("question_tokens"),
            "answer_tokens": _tok_stats("answer_tokens"),
            "reasoning_tokens": _tok_stats("reasoning_tokens"),
            "reasoning_full_tokens": _tok_stats("reasoning_full_tokens"),
            "qar_total_tokens": _tok_stats("qar_total_tokens"),
        },
        "source_length_stats_before_processing": collect_source_length_stats(raw_for_stats, tokenizer),
        "note": "Reasoning is cropped only by complete sentence-like units. First/last/answer-anchor/math/stride sentences are kept; gaps are represented by ellipses.",
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Prepare root-level S1K base content data for all SEDD SFT pipelines.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Config file to read data/model options from.")
    args = parser.parse_args()
    build_base(load_config(args.config))


if __name__ == "__main__":
    main()
