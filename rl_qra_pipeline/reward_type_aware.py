from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from answer_specs import (
    AnswerSpec,
    answer_potential,
    extract_answer_section,
    extract_reasoning_section,
    final_answer_score,
    parse_answer_spec,
    same_answer_type,
    exact_or_numeric_match,
)

REASON_KEYWORDS = re.compile(
    r"\b(because|therefore|thus|so|hence|since|first|then|next|finally|calculate|solve|subtract|add|multiply|divide|equals|result)\b",
    re.I,
)
MATH_SIGNAL = re.compile(r"[=+\-*/^<>≤≥√]|\d")


@dataclass
class RewardOutput:
    final_reward: float
    reason_reward: float
    answer_reward: float
    answer_type: str
    pred_answer: str
    gt_answer: str
    pred_spec: AnswerSpec
    gt_spec: AnswerSpec
    same_type: bool
    exact: bool
    legacy_score: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


def simple_reason_reward(completion: str, cfg: Dict[str, Any]) -> float:
    """Small region-level reward for Reasoning.

    This should not dominate answer correctness. It prevents the RL loop from
    rewarding empty / repeated / malformed reasoning while keeping the answer
    region responsible for exact correctness.
    """
    reason = extract_reasoning_section(completion)
    toks = reason.split()
    if not reason.strip():
        return -0.5
    score = 0.0
    min_tokens = int(cfg.get("reasoning_min_tokens", 8))
    if len(toks) >= min_tokens:
        score += 0.25
    else:
        score -= 0.15
    if REASON_KEYWORDS.search(reason):
        score += 0.25
    if MATH_SIGNAL.search(reason):
        score += 0.20
    # Penalize obvious repeated fragments.
    joined = " ".join(toks[:80]).lower()
    if len(toks) >= 12:
        unique_ratio = len(set(toks)) / max(1, len(toks))
        if unique_ratio < 0.25:
            score -= 0.35
        else:
            score += 0.10
    if len(toks) > int(cfg.get("reasoning_guardrail_max_tokens", 1600)):
        score -= 0.40
    return float(max(-1.0, min(1.0, score)))


def score_qra_type_aware(
    completion: str,
    reference_completion: str,
    cfg: Dict[str, Any],
    legacy_reward: Optional[Dict[str, Any]] = None,
) -> RewardOutput:
    pred_answer = extract_answer_section(completion)
    gt_answer = extract_answer_section(reference_completion)
    pred_spec = parse_answer_spec(pred_answer)
    gt_spec = parse_answer_spec(gt_answer)
    same_type = same_answer_type(pred_spec, gt_spec)
    exact = exact_or_numeric_match(pred_spec, gt_spec)

    answer_reward = final_answer_score(pred_spec, gt_spec)
    reason_reward = simple_reason_reward(completion, cfg)
    legacy_score = None
    if legacy_reward is not None:
        try:
            legacy_score = float(legacy_reward.get("score", 0.0))
        except Exception:
            legacy_score = None

    w_answer = float(cfg.get("answer_weight", 0.75))
    w_reason = float(cfg.get("reason_weight", 0.20))
    w_legacy = float(cfg.get("legacy_weight", 0.05 if legacy_score is not None else 0.0))
    denom = max(1e-8, w_answer + w_reason + w_legacy)
    final_reward = (w_answer * answer_reward + w_reason * reason_reward + w_legacy * (legacy_score or 0.0)) / denom
    final_reward = float(max(-1.0, min(1.0, final_reward)))

    return RewardOutput(
        final_reward=final_reward,
        reason_reward=reason_reward,
        answer_reward=answer_reward,
        answer_type=gt_spec.type,
        pred_answer=pred_answer,
        gt_answer=gt_answer,
        pred_spec=pred_spec,
        gt_spec=gt_spec,
        same_type=same_type,
        exact=exact,
        legacy_score=legacy_score,
        details={
            "pred_answer": pred_answer,
            "gt_answer": gt_answer,
            "pred_type": pred_spec.type,
            "gt_type": gt_spec.type,
            "same_type": same_type,
            "exact": exact,
            "answer_reward": answer_reward,
            "reason_reward": reason_reward,
            "legacy_score": legacy_score,
            "final_reward": final_reward,
        },
    )


def answer_state_potential(answer_text: str, reward: RewardOutput) -> float:
    return answer_potential(answer_text, reward.gt_spec)


def is_structured_answer(reward: RewardOutput) -> bool:
    return reward.gt_spec.structured


def is_single_answer(reward: RewardOutput) -> bool:
    return reward.gt_spec.single
