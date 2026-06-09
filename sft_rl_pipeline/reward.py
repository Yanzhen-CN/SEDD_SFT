import re
from collections import Counter

MOJIBAKE_PATTERNS = (
    "\ufffd",
    "\u951f",
    "\u65a4",
    "\u62f7",
    "\u00c3",
    "\u00c2",
    "\u00e2",
    "\u20ac",
    "\u2122",
)

DEFAULT_ENABLED_REWARDS = (
    "style_structure",
    "math_format",
    "final_answer_style",
    "length",
)

DEFAULT_REWARD_WEIGHTS = {
    "style_structure": 0.4,
    "math_format": 0.3,
    "final_answer_style": 0.2,
    "length": 0.1,
}


def normalize(text):
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def token_set(text):
    return set(re.findall(r"[a-zA-Z0-9]+", normalize(text)))


def ngram_repeat_score(text, n=4):
    toks = re.findall(r"\S+", normalize(text))
    if len(toks) < n * 2:
        return 1.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return max(0.0, 1.0 - repeated / max(1, len(grams)))


def reference_overlap(answer, reference):
    a = token_set(answer)
    r = token_set(reference)
    if not a or not r:
        return 0.0
    return len(a & r) / max(1, len(r))


def style_structure_score(text):
    markers = (
        "let",
        "first",
        "next",
        "because",
        "therefore",
        "thus",
        "since",
        "we have",
        "we get",
        "then",
        "hence",
        "wait",
        "check",
        "substitute",
        "simplify",
        "solve",
    )
    lowered = normalize(text)
    hits = sum(1 for marker in markers if marker in lowered)
    paragraph_bonus = 1.0 if "\n\n" in str(text or "") else 0.0
    return min(1.0, 0.8 * min(1.0, hits / 3) + 0.2 * paragraph_bonus)


def math_format_score(text):
    if not text:
        return 0.0
    has_equation = 1.0 if re.search(r"[A-Za-z0-9]\s*[=<>]\s*[A-Za-z0-9\\$-]", text) else 0.0
    has_latex = 1.0 if re.search(r"\\[a-zA-Z]+|\\\(|\\\)|\$[^$]+\$", text) else 0.0
    has_number = 1.0 if re.search(r"\d", text) else 0.0
    return min(1.0, 0.5 * has_equation + 0.3 * has_latex + 0.2 * has_number)


def final_answer_style_score(text):
    cues = (
        "final answer",
        "boxed",
        "answer",
        "final",
        "therefore",
        "thus",
        "so the",
        "we conclude",
        "the result is",
    )
    lowered = normalize(text)
    return 1.0 if any(cue in lowered for cue in cues) else 0.0


def repeated_char_score(text, max_run=5):
    if not text:
        return 0.0
    runs = [len(match.group(0)) for match in re.finditer(r"(.)\1{2,}", text)]
    if not runs:
        return 1.0
    excess = sum(max(0, run - max_run) for run in runs)
    return max(0.0, 1.0 - excess / max(1, len(text)))


def mojibake_score(text):
    if not text:
        return 0.0
    hits = sum(text.count(pattern) for pattern in MOJIBAKE_PATTERNS)
    control_hits = sum(1 for char in text if ord(char) < 32 and char not in "\n\t\r")
    penalty = hits * 8 + control_hits * 4
    return max(0.0, 1.0 - penalty / max(1, len(text)))


def balanced_symbol_score(text):
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


def readability_score(text):
    if not text:
        return 0.0
    words = re.findall(r"[A-Za-z][A-Za-z0-9']*", text)
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return 0.0
    word_ratio = min(1.0, len(words) / max(1, len(tokens)) / 0.55)
    avg_word_len = sum(len(word) for word in words) / max(1, len(words))
    word_len_score = 1.0 if 2.0 <= avg_word_len <= 12.0 else 0.5
    punct_ratio = len(re.findall(r"[^\w\s]", text)) / max(1, len(text))
    punct_score = max(0.0, 1.0 - max(0.0, punct_ratio - 0.22) / 0.35)
    return max(0.0, min(1.0, 0.5 * word_ratio + 0.25 * word_len_score + 0.25 * punct_score))


def symbol_noise_score(text):
    if not text:
        return 0.0
    noisy = len(re.findall(r"[^A-Za-z0-9\s.,;:!?(){}\[\]<>+=\-*/\\$%^_'\"|]", text))
    slash_runs = sum(len(match.group(0)) - 2 for match in re.finditer(r"\\{3,}", text))
    penalty = noisy + slash_runs * 2
    return max(0.0, 1.0 - penalty / max(1.0, len(text) / 25))


def fatal_error_multiplier(text, config):
    text = str(text or "")
    if not text.strip():
        return 0.0
    if "\ufffd" in text:
        return 0.0
    control_hits = sum(1 for char in text if ord(char) < 32 and char not in "\n\t\r")
    if control_hits:
        return 0.0
    if ngram_repeat_score(text, n=int(config.get("repeat_ngram", 4))) < float(config.get("fatal_repeat_threshold", 0.25)):
        return 0.0
    if repeated_char_score(text, max_run=int(config.get("max_char_run", 5))) < float(config.get("fatal_char_repeat_threshold", 0.5)):
        return 0.0
    toks = re.findall(r"\S+", normalize(text))
    if len(toks) >= int(config.get("fatal_diversity_min_tokens", 12)):
        diversity = len(set(toks)) / max(1, len(toks))
        if diversity < float(config.get("fatal_diversity_threshold", 0.2)):
            return 0.0
    if balanced_symbol_score(text) < float(config.get("fatal_balance_threshold", 0.15)):
        return 0.2
    return 1.0


def reward_weights(config):
    rules = config.get("rules")
    if rules is not None:
        weights = {
            str(rule["name"]): float(rule.get("weight", 1.0))
            for rule in rules
            if rule.get("enabled", True) and float(rule.get("weight", 1.0)) > 0
        }
    else:
        enabled = config.get("enabled")
        enabled = set(DEFAULT_ENABLED_REWARDS) if enabled is None else {str(name) for name in enabled}
        raw_weights = {**DEFAULT_REWARD_WEIGHTS, **config.get("weights", {})}
        weights = {
            name: float(weight)
            for name, weight in raw_weights.items()
            if name in enabled and float(weight) > 0
        }

    if config.get("normalize_weights", True):
        total = sum(weights.values())
        if total > 0:
            weights = {name: weight / total for name, weight in weights.items()}
    return weights


def score_answer(answer, reference="", config=None):
    cfg = config or {}
    weights = reward_weights(cfg)
    text = str(answer or "").strip()
    length_min = int(cfg.get("length_min", 64))
    length_max = int(cfg.get("length_max", 900))
    repeat_ngram = int(cfg.get("repeat_ngram", 4))

    length = len(text)
    if length_min <= length <= length_max:
        length_score = 1.0
    else:
        length_score = max(0.0, 1.0 - abs(length - min(max(length, length_min), length_max)) / max(length_max, 1))

    components = {
        "style_structure": style_structure_score(text),
        "math_format": math_format_score(text),
        "final_answer_style": final_answer_style_score(text),
        "length": length_score,
        "reference_overlap": reference_overlap(text, reference),
        "no_repeat": ngram_repeat_score(text, n=repeat_ngram),
        "no_char_repeat": repeated_char_score(text, max_run=int(cfg.get("max_char_run", 5))),
        "no_mojibake": mojibake_score(text),
        "balanced_symbols": balanced_symbol_score(text),
        "readability": readability_score(text),
        "low_symbol_noise": symbol_noise_score(text),
    }

    base_score = sum(float(weights.get(name, 0.0)) * value for name, value in components.items())
    fatal_multiplier = fatal_error_multiplier(text, cfg)
    score = base_score * fatal_multiplier
    return {
        "score": score,
        "base_score": base_score,
        "fatal_multiplier": fatal_multiplier,
        "active_weight_sum": sum(weights.values()),
        **components,
        "chars": length,
    }
