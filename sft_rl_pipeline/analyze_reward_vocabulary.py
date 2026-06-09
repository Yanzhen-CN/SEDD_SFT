"""Analyze real prepared S1K/QAR JSONL for reward lexical statistics.

Run after generating anchored QAR data:

    python sft_rl_pipeline/analyze_reward_vocabulary.py \
      --data-dir sft_answer_pipeline/data/QAR \
      --out sft_rl_pipeline/reports/s1k_reward_vocab_stats.json

The output is read by reward.py through reward.lexicon_stats_path.  This keeps
reward cues grounded in the actual S1K split you prepared, rather than only a
hand-written list.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from reward import (
    DEFAULT_ANSWER_CUES,
    DEFAULT_REASONING_CUES,
    split_qar_sections,
    punctuation_correctness_score,
    section_edge_quality_score,
    reasoning_keyword_density_score,
    reasoning_math_signal_score,
    answer_result_quality_score,
    has_bad_leading_punctuation,
    reasoning_terminal_ok,
    answer_terminal_ok,
)

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9']*|\d+(?:\.\d+)?|\\[A-Za-z]+|[=<>+\-*/^]")


def normalize(text):
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def ngrams(tokens, n):
    for i in range(0, max(0, len(tokens) - n + 1)):
        yield " ".join(tokens[i:i + n])


def ordered_segments(sample):
    segments = sample.get("segments", {})
    order = sample.get("segment_order") or list(segments.keys())
    used = set()
    for name in order:
        if name in segments:
            used.add(name)
            yield name, segments[name]
    for name, seg in segments.items():
        if name not in used:
            yield name, seg


def assistant_completion(sample):
    parts = []
    seen_assistant = False
    for name, seg in ordered_segments(sample):
        if seen_assistant:
            parts.append(seg.get("text", ""))
        if name == "assistant_label":
            seen_assistant = True
    if parts:
        return "".join(parts)
    # fallback: all train/anchor text
    return "".join(seg.get("text", "") for _, seg in ordered_segments(sample))


def phrase_count(text, phrase):
    return normalize(text).count(normalize(phrase))


def load_rows(data_dir, splits):
    for split in splits:
        path = Path(data_dir) / f"{split}.jsonl"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if line.strip():
                    yield split, line_idx, json.loads(line)


def top_items(counter, n):
    return [{"text": k, "count": v} for k, v in counter.most_common(n)]


def main():
    parser = argparse.ArgumentParser(description="Compute S1K anchored-QAR reward vocabulary statistics.")
    parser.add_argument("--data-dir", default="sft_answer_pipeline/data/QAR")
    parser.add_argument("--splits", default="train,validation,test")
    parser.add_argument("--out", default="sft_rl_pipeline/reports/s1k_reward_vocab_stats.json")
    parser.add_argument("--top-n", type=int, default=80)
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    counts = defaultdict(int)
    tok_reason = Counter()
    tok_answer = Counter()
    bigram_reason = Counter()
    trigram_reason = Counter()
    bigram_answer = Counter()
    trigram_answer = Counter()
    cue_reason = Counter()
    cue_answer = Counter()
    punct_scores = []
    edge_scores = []
    reasoning_keyword_scores = []
    reasoning_math_scores = []
    answer_result_scores = []
    bad_reasoning_starts = 0
    bad_answer_starts = 0
    bad_reasoning_ends = 0
    bad_answer_ends = 0
    scalar_answer_like = 0
    length_reason = []
    length_answer = []
    examples_missing = []

    for split, line_idx, sample in load_rows(args.data_dir, splits):
        completion = assistant_completion(sample)
        info = split_qar_sections(completion)
        reasoning = str(info["reasoning_text"] or "")
        answer = str(info["answer_text"] or "")
        counts[f"rows_{split}"] += 1
        counts["rows_total"] += 1
        if not reasoning or not answer:
            counts["rows_missing_section"] += 1
            if len(examples_missing) < 10:
                examples_missing.append({"split": split, "line_idx": line_idx, "id": sample.get("id")})
        rtoks = [t.lower() for t in TOKEN_RE.findall(reasoning)]
        atoks = [t.lower() for t in TOKEN_RE.findall(answer)]
        length_reason.append(len(rtoks))
        length_answer.append(len(atoks))
        tok_reason.update(rtoks)
        tok_answer.update(atoks)
        bigram_reason.update(ngrams(rtoks, 2))
        trigram_reason.update(ngrams(rtoks, 3))
        bigram_answer.update(ngrams(atoks, 2))
        trigram_answer.update(ngrams(atoks, 3))
        for cue in DEFAULT_REASONING_CUES:
            c = phrase_count(reasoning, cue)
            if c:
                cue_reason[cue] += c
        for cue in DEFAULT_ANSWER_CUES:
            c = phrase_count(answer, cue)
            if c:
                cue_answer[cue] += c
        joined_text = reasoning + "\n" + answer
        punct_scores.append(punctuation_correctness_score(joined_text))
        edge_scores.append(section_edge_quality_score(reasoning, answer))
        reasoning_keyword_scores.append(reasoning_keyword_density_score(reasoning))
        reasoning_math_scores.append(reasoning_math_signal_score(reasoning))
        answer_result_scores.append(answer_result_quality_score(answer))
        bad_reasoning_starts += int(has_bad_leading_punctuation(reasoning))
        bad_answer_starts += int(has_bad_leading_punctuation(answer))
        bad_reasoning_ends += int(not reasoning_terminal_ok(reasoning))
        bad_answer_ends += int(not answer_terminal_ok(answer))
        scalar_answer_like += int(len(atoks) <= 5 and bool(answer.strip()))

    def summary(vals):
        vals = sorted(vals)
        if not vals:
            return {"count": 0}
        def pct(q):
            idx = min(len(vals)-1, max(0, round((len(vals)-1)*q)))
            return vals[int(idx)]
        return {
            "count": len(vals),
            "min": vals[0],
            "p50": pct(0.5),
            "p90": pct(0.9),
            "p95": pct(0.95),
            "max": vals[-1],
            "avg": round(sum(vals)/len(vals), 4),
        }

    # Recommended cues: keep hand-written cue set but order by actual S1K frequency.
    recommended_reasoning = [k for k, _ in cue_reason.most_common()] + [k for k in DEFAULT_REASONING_CUES if k not in cue_reason]
    recommended_answer = [k for k, _ in cue_answer.most_common()] + [k for k in DEFAULT_ANSWER_CUES if k not in cue_answer]

    out = {
        "data_dir": str(args.data_dir),
        "splits": splits,
        "counts": dict(counts),
        "length_reason_tokens": summary(length_reason),
        "length_answer_tokens": summary(length_answer),
        "punctuation_correctness": summary(punct_scores),
        "section_edge_quality": summary(edge_scores),
        "reasoning_keyword_density": summary(reasoning_keyword_scores),
        "reasoning_math_signal": summary(reasoning_math_scores),
        "answer_result_quality": summary(answer_result_scores),
        "section_edge_counts": {
            "bad_reasoning_starts": bad_reasoning_starts,
            "bad_answer_starts": bad_answer_starts,
            "bad_reasoning_ends": bad_reasoning_ends,
            "bad_answer_ends": bad_answer_ends,
            "scalar_answer_like_le5_tokens": scalar_answer_like,
        },
        "reasoning_cue_hits": top_items(cue_reason, args.top_n),
        "answer_cue_hits": top_items(cue_answer, args.top_n),
        "reasoning_top_unigrams": top_items(tok_reason, args.top_n),
        "answer_top_unigrams": top_items(tok_answer, args.top_n),
        "reasoning_top_bigrams": top_items(bigram_reason, args.top_n),
        "reasoning_top_trigrams": top_items(trigram_reason, args.top_n),
        "answer_top_bigrams": top_items(bigram_answer, args.top_n),
        "answer_top_trigrams": top_items(trigram_answer, args.top_n),
        "recommended_reasoning_cues": recommended_reasoning[:80],
        "recommended_answer_cues": recommended_answer[:40],
        "examples_missing_section": examples_missing,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps({
        "rows_total": counts.get("rows_total", 0),
        "rows_missing_section": counts.get("rows_missing_section", 0),
        "reasoning_length": out["length_reason_tokens"],
        "answer_length": out["length_answer_tokens"],
        "top_reasoning_cues": out["reasoning_cue_hits"][:10],
        "top_answer_cues": out["answer_cue_hits"][:10],
        "section_edge_counts": out["section_edge_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
