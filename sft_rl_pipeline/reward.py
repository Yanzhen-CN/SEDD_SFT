"""Rule reward for anchored QAR SEDD outputs, v4.

Current anchored QAR data/generation format:

    User: <question>
    Assistant:
    Reasoning:      # fixed anchor, train=False
    <generated reasoning content>

    Answer:         # fixed anchor, train=False
    <generated answer content>

Because `Reasoning:` and `Answer:` are anchors, this reward does NOT reward
whether the model generated those labels.  It scores only the contents inside
the two anchored sections.

v3 changes relative to v2:
  * adds explicit deduction multipliers for malformed section edges, e.g. a
    section starting with '.', ',', ':' or other bad punctuation;
  * treats mathematical writing specially: equations / LaTeX may end with
    '}', ']', '$', or a numeric expression, but ordinary prose reasoning should
    end with sentence punctuation;
  * does not require the Answer section to contain fixed final-answer phrases:
    pure numbers, single words, short formulas, and \boxed{} answers are also
    rewarded;
  * expands reasoning cues with documented transition / signal words such as
    according to, based on, because, therefore, thus, hence, consequently, etc.;
  * keeps reference_overlap disabled by default because exact restoration can
    over-reward memorization and under-reward valid paraphrases.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple


MOJIBAKE_PATTERNS = (
    "\ufffd", "\u951f", "\u65a4", "\u62f7", "\u00c3", "\u00c2", "\u00e2", "\u20ac", "\u2122",
)

# Seed lexicons.  Run analyze_reward_vocabulary.py after preparing S1K/QAR data
# to get frequency-sorted recommended cues from your actual dataset.
DEFAULT_REASONING_CUES = (
    # planning / decomposition
    "we need", "need to", "let", "first", "second", "third", "next", "then", "now", "finally",
    "consider", "case", "suppose", "assume", "given", "condition", "constraint", "goal", "objective",
    "we want", "we are asked", "the problem asks", "start by", "begin by",
    # evidence / reference / grounding
    "according to", "based on", "from", "by", "using", "given that", "note that", "notice that",
    "observe that", "recall that", "definition", "theorem", "property", "fact", "formula",
    # causal / logical connectors
    "because", "since", "as", "therefore", "thus", "hence", "so", "consequently", "accordingly",
    "as a result", "for this reason", "which means", "this means", "implies", "follows", "leads to",
    "result in", "if", "then", "otherwise", "however", "but", "although", "nevertheless", "also",
    # computation / algebra / verification
    "solve", "compute", "calculate", "derive", "simplify", "substitute", "equation", "expression",
    "value", "check", "verify", "satisfy", "plug", "compare", "rearrange", "factor", "expand",
    "evaluate", "cancel", "divide", "multiply", "add", "subtract", "sum", "product", "ratio",
    "probability", "integer", "positive", "negative", "mod", "gcd", "lcm", "prime", "root",
    # self-correction / exploration common in long reasoning traces
    "wait", "maybe", "actually", "alternatively", "instead", "let's", "try", "another way", "on the other hand",
    "conclude", "we get", "we have", "we can", "this gives", "this is", "it remains", "it suffices",
)

DEFAULT_ANSWER_CUES = (
    "the final answer is", "final answer", "the answer is", "answer is", "therefore, the answer",
    "thus, the answer", "so the answer", "we conclude", "the result is", "boxed", "\\boxed",
)

DEFAULT_ENABLED_REWARDS = (
    "section_edge_quality",
    "punctuation_correctness",
    "reasoning_keyword_density",
    "reasoning_math_signal",
    "answer_result_quality",
    "section_content_quality",
    "no_repeat",
    "no_mojibake",
    "low_symbol_noise",
    "readability",
    "length_guardrail",
    "reference_overlap",
)

DEFAULT_REWARD_WEIGHTS = {
    # Integer-point reward.  These are NOT probabilities.
    # Positive subtotal is about 100 points; explicit penalties can make the
    # final score negative.  This is easier to explain than fractional weights.
    "section_edge_quality": 12.0,
    "punctuation_correctness": 10.0,
    "reasoning_keyword_density": 20.0,
    "reasoning_math_signal": 12.0,
    "answer_result_quality": 20.0,
    "section_content_quality": 12.0,
    "no_repeat": 6.0,
    "no_mojibake": 3.0,
    "low_symbol_noise": 3.0,
    "readability": 2.0,
    "length_guardrail": 2.0,
    # Kept off by default; exact overlap is not the main goal of RL reward.
    "reference_overlap": 0.0,
}

REASONING_MARKER_RE = re.compile(r"(?im)(^|\n)\s*(?:#+\s*)?reasoning\s*:")
ANSWER_MARKER_RE = re.compile(r"(?im)(^|\n)\s*(?:#+\s*)?(?:final\s+)?answer\s*:")
ASSISTANT_RE = re.compile(r"(?im)(^|\n)\s*assistant\s*:")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9']*|\d+(?:\.\d+)?|\\[A-Za-z]+|[=<>+\-*/^]")

BAD_LEADING_PUNCT = set(".,;:!?)]}>")
PROSE_TERMINAL = set(".!?")
MATH_TERMINAL_RE = re.compile(r"(?:\\\]|\\\)|\$|\}|\]|\)|\d|[A-Za-z]|\\boxed\s*\{[^{}]*\})\s*$", re.I)
SCALAR_ANSWER_RE = re.compile(
    r"^\s*(?:[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?|[-+]?\$?\d[\d,]*(?:\.\d+)?%?|[A-Za-z]+|[A-Za-z]\s*=\s*[-+]?\d+(?:\.\d+)?|\\?\(?[-+]?\d+(?:\.\d+)?\\?\)?|\\boxed\s*\{[^{}]+\})\s*[.!?]?\s*$",
    re.I,
)
FORMULA_RE = re.compile(
    r"(\\[a-zA-Z]+|\\\(|\\\)|\$[^$]+\$|[A-Za-z0-9]\s*[=<>]\s*[A-Za-z0-9\\$\-]|\d+\s*[+\-*/^]\s*\d+|\\boxed\s*\{)",
    re.I,
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def token_set(text: str) -> set:
    return set(re.findall(r"[a-zA-Z0-9]+", normalize(text)))


def assistant_completion_view(text: str) -> str:
    """Drop User/Assistant prefix when present, keeping Reasoning/Answer sections."""
    raw = str(text or "")
    matches = list(ASSISTANT_RE.finditer(raw))
    if matches:
        match = matches[-1]
        colon = match.group(0).rfind(":")
        return raw[match.start() + colon + 1:].strip()
    return raw.strip()


def _marker_content_start(match: re.Match) -> int:
    marker = match.group(0)
    colon = marker.rfind(":")
    return match.start() + colon + 1


def _first_nonspace_index(text: str) -> int:
    match = re.search(r"\S", str(text or ""))
    return match.start() if match else -1


def _first_nonspace_char(text: str) -> str:
    match = re.search(r"\S", str(text or ""))
    return match.group(0) if match else ""


def _last_nonspace_char(text: str) -> str:
    match = re.search(r"\S\s*$", str(text or ""))
    return match.group(0) if match else ""


def split_qar_sections(text: str) -> Dict[str, object]:
    """Parse QAR completion into reasoning and answer sections.

    Accepts full text (`User... Assistant...`) or assistant completion
    (`Reasoning... Answer...`).  When anchors are absent, it returns a fallback
    answer_text so old files still get guardrail rewards, but marker diagnostics
    remain zero.
    """
    raw = assistant_completion_view(text)
    reasoning_matches = list(REASONING_MARKER_RE.finditer(raw))
    answer_matches = list(ANSWER_MARKER_RE.finditer(raw))
    r_match = reasoning_matches[0] if reasoning_matches else None
    a_match = answer_matches[0] if answer_matches else None

    has_reasoning = r_match is not None
    has_answer = a_match is not None
    order_ok = bool(r_match is not None and a_match is not None and r_match.start() < a_match.start())

    starts_with_reasoning = False
    if r_match is not None:
        first_nonspace = _first_nonspace_index(raw)
        starts_with_reasoning = first_nonspace >= 0 and r_match.start() <= first_nonspace + 3

    reasoning_text = ""
    answer_text = ""
    if order_ok:
        reasoning_text = raw[_marker_content_start(r_match): a_match.start()].strip()
        answer_text = raw[_marker_content_start(a_match):].strip()
    elif has_reasoning:
        reasoning_text = raw[_marker_content_start(r_match):].strip()
    elif has_answer:
        answer_text = raw[_marker_content_start(a_match):].strip()
    else:
        answer_text = raw.strip()

    return {
        "raw_completion": raw,
        "has_reasoning": has_reasoning,
        "has_answer": has_answer,
        "order_ok": order_ok,
        "starts_with_reasoning": starts_with_reasoning,
        "reasoning_text": reasoning_text,
        "answer_text": answer_text,
        "reasoning_marker_count": len(reasoning_matches),
        "answer_marker_count": len(answer_matches),
        "reasoning_pos": -1 if r_match is None else r_match.start(),
        "answer_pos": -1 if a_match is None else a_match.start(),
    }


def _as_tuple(value, default):
    if value is None:
        return tuple(default)
    if isinstance(value, (list, tuple)):
        return tuple(str(x) for x in value if str(x).strip())
    return tuple(str(value).split("|"))


def _load_lexicon_file(path: str | None) -> Dict[str, List[str]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    if isinstance(data, dict):
        if "reasoning_cues" in data:
            out["reasoning_cues"] = list(data["reasoning_cues"])
        if "answer_cues" in data:
            out["answer_cues"] = list(data["answer_cues"])
        if "recommended_reasoning_cues" in data:
            out["reasoning_cues"] = list(data["recommended_reasoning_cues"])
        if "recommended_answer_cues" in data:
            out["answer_cues"] = list(data["recommended_answer_cues"])
    return out


def get_reasoning_cues(config: Mapping | None = None) -> Tuple[str, ...]:
    cfg = config or {}
    lex = _load_lexicon_file(cfg.get("lexicon_stats_path") or cfg.get("reward_vocab_path"))
    return _as_tuple(cfg.get("reasoning_cues") or lex.get("reasoning_cues"), DEFAULT_REASONING_CUES)


def get_answer_cues(config: Mapping | None = None) -> Tuple[str, ...]:
    cfg = config or {}
    lex = _load_lexicon_file(cfg.get("lexicon_stats_path") or cfg.get("reward_vocab_path"))
    return _as_tuple(cfg.get("answer_cues") or lex.get("answer_cues"), DEFAULT_ANSWER_CUES)


def count_phrase_hits(text: str, cues: Iterable[str]) -> Tuple[int, int, List[str]]:
    lowered = normalize(text)
    total = 0
    hit_phrases = []
    for cue in cues:
        cue_l = normalize(cue)
        if not cue_l:
            continue
        hits = lowered.count(cue_l)
        if hits:
            total += hits
            hit_phrases.append(cue)
    return total, len(set(hit_phrases)), hit_phrases


def reasoning_keyword_density_score(reasoning_text: str, config: Mapping | None = None) -> float:
    """Reward reasoning-like connective words, not raw length."""
    cfg = config or {}
    text = str(reasoning_text or "")
    if not text.strip():
        return 0.0
    cues = get_reasoning_cues(cfg)
    total_hits, unique_hits, _ = count_phrase_hits(text, cues)
    tokens = max(1, len(WORD_RE.findall(text)))
    expected_hits = max(3.0, min(16.0, tokens / 42.0))
    total_score = min(1.0, total_hits / expected_hits)
    unique_score = min(1.0, unique_hits / 8.0)
    sequence_score = 1.0 if re.search(r"\b(first|second|next|then|finally|therefore|thus|hence|because|since|according to|based on)\b", normalize(text)) else 0.0
    return max(0.0, min(1.0, 0.55 * total_score + 0.30 * unique_score + 0.15 * sequence_score))


def reasoning_math_signal_score(reasoning_text: str) -> float:
    """Reward actual math / symbolic reasoning signals in the reasoning section."""
    text = str(reasoning_text or "")
    if not text.strip():
        return 0.0
    has_formula = 1.0 if FORMULA_RE.search(text) else 0.0
    has_number = 1.0 if re.search(r"\d", text) else 0.0
    has_operation_word = 1.0 if re.search(r"\b(solve|compute|calculate|equation|substitute|simplify|factor|expand|sum|product|probability|integer|gcd|prime|root)\b", normalize(text)) else 0.0
    line_or_step = 1.0 if re.search(r"(?:^|\n)\s*(?:\d+[.)]|[-*]|[A-Za-z]+\s*=|\\\[|\$)", text) else 0.0
    return min(1.0, 0.35 * has_formula + 0.20 * has_number + 0.25 * has_operation_word + 0.20 * line_or_step)


def answer_result_quality_score(answer_text: str, config: Mapping | None = None) -> float:
    """Reward final-answer-like content without requiring stock phrases.

    A bare number, a single word/choice, a short formula, and \boxed{} are all
    valid final-answer forms.  Stock phrases are a bonus, not a requirement.
    """
    cfg = config or {}
    text = str(answer_text or "").strip()
    if not text:
        return 0.0
    cues = get_answer_cues(cfg)
    cue_total, cue_unique, _ = count_phrase_hits(text, cues)
    cue_score = min(1.0, 0.65 * min(1.0, cue_total / 1.0) + 0.35 * min(1.0, cue_unique / 2.0))
    has_boxed = 1.0 if re.search(r"\\boxed\s*\{|boxed", text, re.I) else 0.0
    has_formula = 1.0 if FORMULA_RE.search(text) else 0.0
    has_number = 1.0 if re.search(r"\d", text) else 0.0
    scalar = 1.0 if SCALAR_ANSWER_RE.match(text) else 0.0
    tokens = re.findall(r"\S+", text)
    concise = 1.0 if 1 <= len(tokens) <= int(cfg.get("answer_concise_max_tokens", 80)) else 0.35
    terminal_ok = 1.0 if answer_terminal_ok(text) else 0.55
    return max(0.0, min(1.0,
        0.25 * cue_score
        + 0.30 * max(scalar, has_boxed, has_formula, has_number)
        + 0.25 * concise
        + 0.20 * terminal_ok
    ))


def math_format_score(text: str) -> float:
    if not str(text or "").strip():
        return 0.0
    has_equation = 1.0 if re.search(r"[A-Za-z0-9]\s*[=<>]\s*[A-Za-z0-9\\$-]", text) else 0.0
    has_latex = 1.0 if re.search(r"\\[a-zA-Z]+|\\\(|\\\)|\$[^$]+\$", text) else 0.0
    has_number = 1.0 if re.search(r"\d", text) else 0.0
    has_operation = 1.0 if re.search(r"\d\s*[+\-*/^]\s*\d", text) else 0.0
    return min(1.0, 0.35 * has_equation + 0.30 * has_latex + 0.20 * has_number + 0.15 * has_operation)


def has_bad_leading_punctuation(text: str) -> bool:
    stripped = str(text or "").lstrip()
    if not stripped:
        return False
    if stripped.startswith(".") and re.match(r"^\.\d", stripped):
        return False  # decimal such as .5
    return stripped[0] in BAD_LEADING_PUNCT


def reasoning_terminal_ok(text: str) -> bool:
    stripped = str(text or "").rstrip()
    if not stripped:
        return False
    if stripped[-1] in PROSE_TERMINAL:
        return True
    # Equations and display math can end with }, ], ), or $, but only if the
    # section contains math-like content.  This avoids rewarding arbitrary words
    # with no final punctuation.
    if FORMULA_RE.search(stripped) and MATH_TERMINAL_RE.search(stripped):
        return True
    return False


def answer_terminal_ok(text: str) -> bool:
    stripped = str(text or "").rstrip()
    if not stripped:
        return False
    if stripped[-1] in PROSE_TERMINAL:
        return True
    # Short scalar answers are valid without a period.
    if SCALAR_ANSWER_RE.match(stripped):
        return True
    if FORMULA_RE.search(stripped) and MATH_TERMINAL_RE.search(stripped):
        return True
    return False


def section_edge_quality_score(reasoning_text: str, answer_text: str, config: Mapping | None = None) -> float:
    """Score section starts/ends.

    This implements the explicit format guardrail discussed in the project:
    sections should not start with malformed punctuation such as '.', and prose
    reasoning should end as a sentence.  Answer is allowed to be a single number,
    word, option, or formula without a final period.
    """
    reasoning = str(reasoning_text or "")
    answer = str(answer_text or "")
    if not reasoning.strip() or not answer.strip():
        return 0.0
    score = 1.0
    if has_bad_leading_punctuation(reasoning):
        score -= 0.25
    if has_bad_leading_punctuation(answer):
        score -= 0.25
    if not reasoning_terminal_ok(reasoning):
        score -= 0.20
    if not answer_terminal_ok(answer):
        score -= 0.10
    # Avoid content that begins with repeated delimiters or ends in an obvious
    # dangling operator.
    if re.match(r"^\s*[-_=*]{3,}", reasoning) or re.match(r"^\s*[-_=*]{3,}", answer):
        score -= 0.15
    if re.search(r"[+\-*/=<>]\s*$", reasoning) or re.search(r"[+\-*/=<>]\s*$", answer):
        score -= 0.20
    return max(0.0, min(1.0, score))


def section_format_penalty_multiplier(reasoning_text: str, answer_text: str, completion_text: str, config: Mapping | None = None) -> float:
    """Explicit multiplicative deductions for malformed section content."""
    cfg = config or {}
    multiplier = 1.0
    if not str(reasoning_text or "").strip() or not str(answer_text or "").strip():
        return 0.0
    if has_bad_leading_punctuation(reasoning_text):
        multiplier *= float(cfg.get("penalty_bad_reasoning_start", 0.80))
    if has_bad_leading_punctuation(answer_text):
        multiplier *= float(cfg.get("penalty_bad_answer_start", 0.80))
    if not reasoning_terminal_ok(reasoning_text):
        multiplier *= float(cfg.get("penalty_bad_reasoning_end", 0.88))
    if not answer_terminal_ok(answer_text):
        multiplier *= float(cfg.get("penalty_bad_answer_end", 0.92))
    if re.search(r"(?i)\breasoning\s*:", reasoning_text) or re.search(r"(?i)\banswer\s*:", answer_text):
        multiplier *= float(cfg.get("penalty_repeated_anchor_in_content", 0.85))
    # If answer becomes a second reasoning trace, reduce reward.  This does not
    # punish multi-word final statements, just long verbose answers.
    answer_tokens = len(re.findall(r"\S+", str(answer_text or "")))
    if answer_tokens > int(cfg.get("answer_soft_max_tokens", 120)):
        multiplier *= float(cfg.get("penalty_verbose_answer", 0.90))
    return max(0.0, min(1.0, multiplier))


def section_content_quality_score(reasoning_text: str, answer_text: str, config: Mapping | None = None) -> float:
    cfg = config or {}
    r_tokens = re.findall(r"\S+", str(reasoning_text or ""))
    a_tokens = re.findall(r"\S+", str(answer_text or ""))
    score = 0.0
    if r_tokens:
        score += 0.22
    if a_tokens:
        score += 0.22
    r_min = int(cfg.get("reasoning_min_tokens", 12))
    if len(r_tokens) >= r_min:
        score += 0.18
    elif len(r_tokens) >= max(4, r_min // 3):
        score += 0.09
    a_max = int(cfg.get("answer_max_tokens", 160))
    if 1 <= len(a_tokens) <= a_max:
        score += 0.18
    elif len(a_tokens) > 0:
        score += 0.06
    if reasoning_text and answer_text and normalize(reasoning_text) != normalize(answer_text):
        score += 0.10
    if section_edge_quality_score(reasoning_text, answer_text, cfg) >= 0.75:
        score += 0.10
    return max(0.0, min(1.0, score))


def ngram_repeat_score(text: str, n: int = 4) -> float:
    toks = re.findall(r"\S+", normalize(text))
    if len(toks) < n * 2:
        return 1.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return max(0.0, 1.0 - repeated / max(1, len(grams)))


def reference_overlap(answer: str, reference: str) -> float:
    a = token_set(answer)
    r = token_set(reference)
    if not a or not r:
        return 0.0
    return len(a & r) / max(1, len(r))


def repeated_char_score(text: str, max_run: int = 5) -> float:
    if not text:
        return 0.0
    runs = [len(match.group(0)) for match in re.finditer(r"(.)\1{2,}", text)]
    if not runs:
        return 1.0
    excess = sum(max(0, run - max_run) for run in runs)
    return max(0.0, 1.0 - excess / max(1, len(text)))


def mojibake_score(text: str) -> float:
    if not text:
        return 0.0
    hits = sum(text.count(pattern) for pattern in MOJIBAKE_PATTERNS)
    control_hits = sum(1 for char in text if ord(char) < 32 and char not in "\n\t\r")
    penalty = hits * 8 + control_hits * 4
    return max(0.0, 1.0 - penalty / max(1, len(text)))


def balanced_symbol_score(text: str) -> float:
    if not text:
        return 0.0
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = {value: key for key, value in pairs.items()}
    stack = []
    errors = 0
    for char in text:
        if char in pairs:
            stack.append(char)
        elif char in closing:
            if stack and stack[-1] == closing[char]:
                stack.pop()
            else:
                errors += 1
    errors += len(stack)
    errors += text.count("$") % 2
    return max(0.0, 1.0 - errors / max(1.0, len(text) / 40))


def punctuation_correctness_score(text: str) -> float:
    raw = str(text or "")
    if not raw.strip():
        return 0.0
    balanced = balanced_symbol_score(raw)
    punct_count = len(re.findall(r"[.,;:!?]", raw))
    punct_density = punct_count / max(1, len(raw))
    density_score = 1.0 - min(1.0, max(0.0, punct_density - 0.18) / 0.25)
    if len(raw) > 160 and punct_count == 0 and not re.search(r"[=<>]", raw):
        density_score *= 0.5
    bad_runs = len(re.findall(r"[.,;:!?]{4,}", raw)) + len(re.findall(r"[-_=*]{8,}", raw))
    run_score = max(0.0, 1.0 - bad_runs / 3.0)
    separator_score = 1.0 if re.search(r"[.;!?\n=]", raw) else 0.5
    return max(0.0, min(1.0, 0.45 * balanced + 0.25 * density_score + 0.20 * run_score + 0.10 * separator_score))


def readability_score(text: str) -> float:
    if not text:
        return 0.0
    words = re.findall(r"[A-Za-z][A-Za-z0-9']*", text)
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return 0.0
    word_ratio = min(1.0, len(words) / max(1, len(tokens)) / 0.55)
    avg_word_len = sum(len(word) for word in words) / max(1, len(words))
    word_len_score = 1.0 if 2.0 <= avg_word_len <= 14.0 else 0.5
    punct_ratio = len(re.findall(r"[^\w\s]", text)) / max(1, len(text))
    punct_score = max(0.0, 1.0 - max(0.0, punct_ratio - 0.24) / 0.35)
    return max(0.0, min(1.0, 0.5 * word_ratio + 0.25 * word_len_score + 0.25 * punct_score))


def symbol_noise_score(text: str) -> float:
    if not text:
        return 0.0
    noisy = len(re.findall(r"[^A-Za-z0-9\s.,;:!?(){}\[\]<>+=\-*/\\$%^_'\"|]", text))
    slash_runs = sum(len(match.group(0)) - 2 for match in re.finditer(r"\\{3,}", text))
    penalty = noisy + slash_runs * 2
    return max(0.0, 1.0 - penalty / max(1.0, len(text) / 25))


def length_guardrail_score(reasoning_text: str, answer_text: str, config: Mapping | None = None) -> float:
    cfg = config or {}
    r_tokens = len(re.findall(r"\S+", str(reasoning_text or "")))
    a_tokens = len(re.findall(r"\S+", str(answer_text or "")))
    if r_tokens == 0 or a_tokens == 0:
        return 0.0
    r_max = int(cfg.get("reasoning_guardrail_max_tokens", 1600))
    a_max = int(cfg.get("answer_guardrail_max_tokens", 220))
    score = 1.0
    if r_tokens > r_max:
        score *= max(0.1, 1.0 - (r_tokens - r_max) / max(r_max, 1))
    if a_tokens > a_max:
        score *= max(0.1, 1.0 - (a_tokens - a_max) / max(a_max, 1))
    return max(0.0, min(1.0, score))


def fatal_error_multiplier(text: str, config: Mapping | None = None) -> float:
    cfg = config or {}
    text = str(text or "")
    if not text.strip():
        return 0.0
    if "\ufffd" in text:
        return 0.0
    control_hits = sum(1 for char in text if ord(char) < 32 and char not in "\n\t\r")
    if control_hits:
        return 0.0
    if ngram_repeat_score(text, n=int(cfg.get("repeat_ngram", 4))) < float(cfg.get("fatal_repeat_threshold", 0.25)):
        return 0.0
    if repeated_char_score(text, max_run=int(cfg.get("max_char_run", 5))) < float(cfg.get("fatal_char_repeat_threshold", 0.5)):
        return 0.0
    toks = re.findall(r"\S+", normalize(text))
    if len(toks) >= int(cfg.get("fatal_diversity_min_tokens", 12)):
        diversity = len(set(toks)) / max(1, len(toks))
        if diversity < float(cfg.get("fatal_diversity_threshold", 0.2)):
            return 0.0
    if balanced_symbol_score(text) < float(cfg.get("fatal_balance_threshold", 0.15)):
        return 0.2
    return 1.0


def reward_weights(config: Mapping | None = None) -> Dict[str, float]:
    cfg = config or {}
    rules = cfg.get("rules")
    if rules is not None:
        weights = {
            str(rule["name"]): float(rule.get("weight", 1.0))
            for rule in rules
            if rule.get("enabled", True) and float(rule.get("weight", 1.0)) > 0
        }
    else:
        enabled = cfg.get("enabled")
        enabled = set(DEFAULT_ENABLED_REWARDS) if enabled is None else {str(name) for name in enabled}
        raw_weights = {**DEFAULT_REWARD_WEIGHTS, **cfg.get("weights", {})}
        weights = {name: float(weight) for name, weight in raw_weights.items() if name in enabled and float(weight) > 0}
    if cfg.get("normalize_weights", False):
        total = sum(weights.values())
        if total > 0:
            weights = {name: weight / total for name, weight in weights.items()}
    return weights


def _score_texts_for_qar(answer: str, reference: str):
    info = split_qar_sections(answer)
    ref_info = split_qar_sections(reference)
    reasoning_text = str(info["reasoning_text"])
    answer_text = str(info["answer_text"])
    ref_answer_text = str(ref_info["answer_text"] or reference or "")
    math_text = "\n".join(part for part in [reasoning_text, answer_text] if part).strip() or str(answer or "")
    return info, reasoning_text, answer_text, math_text, ref_answer_text



def format_penalty_points(reasoning_text: str, answer_text: str, completion_text: str, config: Mapping | None = None) -> Dict[str, float]:
    """Integer-style penalties.  Negative points are easier to audit than a
    silent multiplicative penalty.  Fatal errors can drive the total reward below
    zero, which is useful for best-of-k rejection.
    """
    cfg = config or {}
    penalties: Dict[str, float] = {}

    reasoning = str(reasoning_text or "")
    answer = str(answer_text or "")
    completion = str(completion_text or "")

    if not reasoning.strip():
        penalties["penalty_empty_reasoning"] = -float(cfg.get("penalty_empty_reasoning_points", 100))
    if not answer.strip():
        penalties["penalty_empty_answer"] = -float(cfg.get("penalty_empty_answer_points", 100))
    if has_bad_leading_punctuation(reasoning):
        penalties["penalty_bad_reasoning_start"] = -float(cfg.get("penalty_bad_reasoning_start_points", 15))
    if has_bad_leading_punctuation(answer):
        penalties["penalty_bad_answer_start"] = -float(cfg.get("penalty_bad_answer_start_points", 15))
    if reasoning.strip() and not reasoning_terminal_ok(reasoning):
        penalties["penalty_bad_reasoning_end"] = -float(cfg.get("penalty_bad_reasoning_end_points", 8))
    if answer.strip() and not answer_terminal_ok(answer):
        penalties["penalty_bad_answer_end"] = -float(cfg.get("penalty_bad_answer_end_points", 5))
    if re.search(r"(?i)\breasoning\s*:", reasoning) or re.search(r"(?i)\banswer\s*:", answer):
        penalties["penalty_repeated_anchor_in_content"] = -float(cfg.get("penalty_repeated_anchor_points", 12))
    if re.search(r"[+\-*/=<>]\s*$", reasoning) or re.search(r"[+\-*/=<>]\s*$", answer):
        penalties["penalty_dangling_operator"] = -float(cfg.get("penalty_dangling_operator_points", 10))

    answer_tokens = len(re.findall(r"\S+", answer))
    if answer_tokens > int(cfg.get("answer_soft_max_tokens", 120)):
        penalties["penalty_verbose_answer"] = -float(cfg.get("penalty_verbose_answer_points", 8))

    # Structural/noise fatal-ish penalties.  These are not softened into a tiny
    # multiplier because malformed text should be visibly punished.
    if "\ufffd" in completion:
        penalties["penalty_mojibake_fatal"] = -float(cfg.get("penalty_fatal_points", 120))
    control_hits = sum(1 for char in completion if ord(char) < 32 and char not in "\n\t\r")
    if control_hits:
        penalties["penalty_control_char_fatal"] = -float(cfg.get("penalty_fatal_points", 120))
    if ngram_repeat_score(completion, n=int(cfg.get("repeat_ngram", 4))) < float(cfg.get("fatal_repeat_threshold", 0.25)):
        penalties["penalty_repetition_fatal"] = -float(cfg.get("penalty_fatal_points", 120))
    if repeated_char_score(completion, max_run=int(cfg.get("max_char_run", 5))) < float(cfg.get("fatal_char_repeat_threshold", 0.5)):
        penalties["penalty_char_repeat_fatal"] = -float(cfg.get("penalty_fatal_points", 120))
    if balanced_symbol_score(completion) < float(cfg.get("fatal_balance_threshold", 0.15)):
        penalties["penalty_unbalanced_symbols"] = -float(cfg.get("penalty_unbalanced_symbols_points", 25))

    return penalties


def score_answer(answer: str, reference: str = "", config: Mapping | None = None) -> Dict[str, float]:
    cfg = config or {}
    weights = reward_weights(cfg)
    text = str(answer or "").strip()
    reference = str(reference or "").strip()

    info, reasoning_text, answer_text, math_text, ref_answer_text = _score_texts_for_qar(text, reference)
    completion_text = str(info["raw_completion"])

    r_total, r_unique, r_hits = count_phrase_hits(reasoning_text, get_reasoning_cues(cfg))
    a_total, a_unique, a_hits = count_phrase_hits(answer_text, get_answer_cues(cfg))

    components = {
        # Anchor diagnostics: returned for reports, not weighted by default.
        "has_reasoning_marker": 1.0 if info["has_reasoning"] else 0.0,
        "has_answer_marker": 1.0 if info["has_answer"] else 0.0,
        "reasoning_before_answer": 1.0 if info["order_ok"] else 0.0,
        "starts_with_reasoning": 1.0 if info["starts_with_reasoning"] else 0.0,
        # Actual anchored-QAR reward components.
        "section_edge_quality": section_edge_quality_score(reasoning_text, answer_text, cfg),
        "punctuation_correctness": punctuation_correctness_score(math_text),
        "reasoning_keyword_density": reasoning_keyword_density_score(reasoning_text, cfg),
        "reasoning_math_signal": reasoning_math_signal_score(reasoning_text),
        "answer_result_quality": answer_result_quality_score(answer_text, cfg),
        "answer_finality": answer_result_quality_score(answer_text, cfg),  # backward-compatible key
        "section_content_quality": section_content_quality_score(reasoning_text, answer_text, cfg),
        "math_format": math_format_score(math_text),
        "length_guardrail": length_guardrail_score(reasoning_text, answer_text, cfg),
        "reference_overlap": reference_overlap(answer_text, ref_answer_text),
        "no_repeat": ngram_repeat_score(completion_text, n=int(cfg.get("repeat_ngram", 4))),
        "no_char_repeat": repeated_char_score(completion_text, max_run=int(cfg.get("max_char_run", 5))),
        "no_mojibake": mojibake_score(completion_text),
        "balanced_symbols": balanced_symbol_score(completion_text),
        "readability": readability_score(completion_text),
        "low_symbol_noise": symbol_noise_score(completion_text),
    }

    positive_points_float = sum(float(weights.get(name, 0.0)) * float(components.get(name, 0.0)) for name in weights)
    max_positive_points = max(1.0, sum(max(0.0, float(v)) for v in weights.values()))
    penalties = format_penalty_points(reasoning_text, answer_text, completion_text, cfg)
    penalty_points_float = sum(float(v) for v in penalties.values())
    score_points_float = positive_points_float + penalty_points_float

    # Normalized score for selection/training.  It can be negative when the text
    # is malformed.  Reward-weighted SFT maps negative scores to the minimum
    # sample weight; best-of-k simply chooses the highest score.
    score = max(-1.0, min(1.0, score_points_float / max_positive_points))

    # Backward-compatible diagnostics.  Multipliers are not used to compute the
    # final score in v4, but keeping them avoids breaking old reports.
    fatal_multiplier = fatal_error_multiplier(completion_text, cfg)
    format_penalty_multiplier = section_format_penalty_multiplier(reasoning_text, answer_text, completion_text, cfg)

    out = {
        "score": score,
        "score_points": int(round(score_points_float)),
        "positive_points": int(round(positive_points_float)),
        "penalty_points": int(round(penalty_points_float)),
        "max_positive_points": int(round(max_positive_points)),
        "base_score": positive_points_float / max_positive_points,
        "fatal_multiplier": fatal_multiplier,
        "format_penalty_multiplier": format_penalty_multiplier,
        "active_weight_sum": sum(weights.values()),
        **components,
        "chars": len(completion_text),
        "reasoning_chars": len(reasoning_text),
        "answer_chars": len(answer_text),
        "reasoning_tokens": len(re.findall(r"\S+", reasoning_text)),
        "answer_tokens": len(re.findall(r"\S+", answer_text)),
        "reasoning_bad_start": 1.0 if has_bad_leading_punctuation(reasoning_text) else 0.0,
        "answer_bad_start": 1.0 if has_bad_leading_punctuation(answer_text) else 0.0,
        "reasoning_terminal_ok": 1.0 if reasoning_terminal_ok(reasoning_text) else 0.0,
        "answer_terminal_ok": 1.0 if answer_terminal_ok(answer_text) else 0.0,
        "answer_scalar_form": 1.0 if SCALAR_ANSWER_RE.match(answer_text.strip()) else 0.0,
        "reasoning_cue_total": r_total,
        "reasoning_cue_unique": r_unique,
        "answer_cue_total": a_total,
        "answer_cue_unique": a_unique,
        "reasoning_cue_hits": r_hits[:20],
        "answer_cue_hits": a_hits[:20],
    }
    out.update(penalties)
    return out
