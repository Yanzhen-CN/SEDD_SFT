from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import torch

from answer_specs import parse_answer_spec, same_answer_type, normalize_answer
from reward_type_aware import score_qra_type_aware, is_single_answer, is_structured_answer
from state_builder import EncodedSample, encode_sample, mask_id_from_graph, project_fixed_, transition_probs


@dataclass
class GuidedRatioState:
    x_state: torch.Tensor              # [L], CPU long tensor
    t: float
    step_size: float
    target_positions: List[int]
    target_ids: List[int]
    state_kind: str
    state_weight: float = 1.0
    meta: Dict[str, str | int | float] = field(default_factory=dict)


def _decode_one(tokenizer, token_id: int) -> str:
    try:
        return normalize_answer(tokenizer.decode([int(token_id)]))
    except Exception:
        return ""


def _wrong_token_weight(tokenizer, token_id: int, gt_type: str, cfg: Dict) -> float:
    """How strongly to suppress a non-target token.

    For single integer/letter answers, wrong type is catastrophic, while same type
    wrong is only a small negative. For structured answers, wrong brackets/signs/
    digits are local errors, so the negative is moderate.
    """
    tok = _decode_one(tokenizer, token_id)
    spec = parse_answer_spec(tok)
    if gt_type in {"single_letter", "single_integer"}:
        dummy = parse_answer_spec("A" if gt_type == "single_letter" else "1")
        dummy.type = gt_type
        if same_answer_type(spec, dummy):
            return float(cfg.get("same_type_negative_weight", 0.10))
        return float(cfg.get("wrong_type_negative_weight", 0.80))
    if gt_type in {"signed_decimal", "interval"}:
        return float(cfg.get("structured_negative_weight", 0.25))
    return float(cfg.get("short_text_negative_weight", 0.15))


def _choose_t(cfg: Dict) -> float:
    vals = cfg.get("t_values")
    if isinstance(vals, list) and vals:
        return float(random.choice(vals))
    return float(random.uniform(float(cfg.get("t_min", 0.05)), float(cfg.get("t_max", 0.95))))


def _corrupt_positions_(x: torch.Tensor, positions: Sequence[int], graph, cfg: Dict, device: torch.device) -> torch.Tensor:
    if not positions:
        return x
    mode = str(cfg.get("corrupt_mode", "mask"))
    if mode == "limit":
        base = graph.sample_limit(1, x.shape[1]).to(device)
        x[:, list(positions)] = base[:, list(positions)]
    else:
        x[:, list(positions)] = int(mask_id_from_graph(graph))
    return x


def _base_state(encoded: EncodedSample, graph, cfg: Dict, device: torch.device) -> torch.Tensor:
    ids_t = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    mode = str(cfg.get("base_state", "reference_answer_masked"))
    if mode == "limit_with_fixed_prompt":
        x = graph.sample_limit(1, len(encoded.ids)).to(device)
        project_fixed_(x, encoded.layout.fixed_locs, encoded.layout.fixed_ids.to(device))
        return x
    # Default: reference context + reference target, then selected answer tokens are corrupted.
    return ids_t.clone()


def build_guided_ratio_states(sample: Dict, tokenizer, graph, cfg: Dict, device: torch.device) -> Tuple[List[GuidedRatioState], Dict]:
    """Construct local policy-improvement states.

    This is not pretending that the target token was sampled. Instead, the verifier
    supplies a reward-improved local action: the GT/component token is positive,
    and high-probability non-GT alternatives are negative. This uses SEDD's full
    ratio field at a noisy state.
    """
    model_cfg = cfg.get("model", {})
    gcfg = cfg.get("guided", {})
    max_length = int(model_cfg.get("max_length", 1024))
    encoded = encode_sample(sample, tokenizer, max_length, tokenizer.eos_token_id)
    answer_pos = [p for p in encoded.layout.answer_positions if p < encoded.layout.real_len]
    if not answer_pos:
        return [], {"reason": "no_answer_positions"}

    rw = score_qra_type_aware(encoded.reference_completion, encoded.reference_completion, cfg.get("reward", {}), legacy_reward=None)
    gt_type = rw.gt_spec.type
    ans_n = len(answer_pos)

    n_states = int(gcfg.get("states_per_sample", 3))
    max_targets = int(gcfg.get("max_targets_per_state", 16))
    reveal_mode = str(gcfg.get("reveal_mode", "prefix_grid"))
    step_size = float(gcfg.get("step_size", 1e-5))

    states: List[GuidedRatioState] = []
    for sidx in range(max(1, n_states)):
        x = _base_state(encoded, graph, gcfg, device)

        # Single-token answers: train the whole answer region from mask.
        if is_single_answer(rw):
            reveal = 0
            corrupt = answer_pos[:]
            targets = answer_pos[:]
            kind = f"{gt_type}:single_mask"
        else:
            if reveal_mode == "random_prefix":
                reveal = random.randint(0, max(0, ans_n - 1))
            elif reveal_mode == "random_subset":
                reveal = 0
            else:
                # prefix grid: all mask, partial prefix, near-complete states.
                reveal = int(round((sidx / max(1, n_states - 1)) * max(0, ans_n - 1))) if n_states > 1 else 0
            reveal = max(0, min(reveal, ans_n))
            if reveal_mode == "random_subset" and ans_n > 1:
                revealed = set(random.sample(answer_pos, k=random.randint(0, ans_n - 1)))
                corrupt = [p for p in answer_pos if p not in revealed]
                targets = corrupt[:]
                kind = f"{gt_type}:random_subset"
            else:
                corrupt = answer_pos[reveal:]
                targets = corrupt[:]
                kind = f"{gt_type}:prefix_reveal_{reveal}"

        _corrupt_positions_(x, corrupt, graph, gcfg, device)
        project_fixed_(x, encoded.layout.fixed_locs, encoded.layout.fixed_ids.to(device))

        if max_targets > 0 and len(targets) > max_targets:
            # Structured answers are short and ordered, so prefix is okay; long text targets are sampled.
            if is_structured_answer(rw):
                targets = targets[:max_targets]
            else:
                targets = sorted(random.sample(targets, max_targets))
        if not targets:
            continue
        target_ids = [int(encoded.ids[p]) for p in targets]
        states.append(
            GuidedRatioState(
                x_state=x[0].detach().cpu(),
                t=_choose_t(gcfg),
                step_size=step_size,
                target_positions=targets,
                target_ids=target_ids,
                state_kind=kind,
                state_weight=1.0,
                meta={"answer_type": gt_type, "answer_len": ans_n, "reveal": reveal},
            )
        )
    return states, {"answer_type": gt_type, "num_states": len(states), "answer_len": ans_n}


def guided_ratio_loss(model, graph, noise, tokenizer, sample: Dict, cfg: Dict, device: torch.device) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Verifier-guided local ratio/policy-improvement loss.

    For a constructed noisy answer state s, SEDD gives pi_theta(.|s) over all
    candidate token transitions. We directly encourage the reward-approved target
    action and suppress high-probability wrong actions:

        L = -w+ log pi(y+|s) + sum_y- w-(y-) [-log(1 - pi(y-|s))]

    This updates the SEDD ratio field without needing many on-policy rollouts.
    """
    gcfg = cfg.get("guided", {})
    states, meta = build_guided_ratio_states(sample, tokenizer, graph, cfg, device)
    if not states:
        z = torch.zeros((), device=device, requires_grad=True)
        return z, {"guided_states": 0, "guided_targets": 0, "pos_logp": 0.0, "neg_prob": 0.0, "answer_type_code": 0.0}

    pos_weight = float(gcfg.get("positive_weight", 1.0))
    neg_weight = float(gcfg.get("negative_weight", 0.30))
    topk_neg = int(gcfg.get("topk_negative", 4))
    eps = float(gcfg.get("eps", 1e-8))
    transition_kind = str(gcfg.get("transition_kind", "analytic"))

    losses: List[torch.Tensor] = []
    pos_logps: List[float] = []
    neg_probs: List[float] = []
    target_count = 0

    for st in states:
        x = st.x_state.to(device).unsqueeze(0)
        t = torch.tensor([[st.t]], dtype=torch.float32, device=device)
        probs = transition_probs(model, graph, noise, x, t, st.step_size, transition_kind, train=True)

        pos_idx = torch.tensor(st.target_positions, dtype=torch.long, device=device)
        tgt = torch.tensor(st.target_ids, dtype=torch.long, device=device)
        p_pos = probs[0, pos_idx, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1).clamp_min(eps)
        state_loss = -pos_weight * p_pos.log().mean() * float(st.state_weight)
        pos_logps.append(float(p_pos.detach().log().mean().item()))
        target_count += len(st.target_positions)

        if topk_neg > 0 and neg_weight > 0:
            local_probs = probs[0, pos_idx, :]
            local_probs_no_gt = local_probs.clone()
            local_probs_no_gt.scatter_(1, tgt.unsqueeze(1), -1.0)
            k = min(topk_neg, local_probs_no_gt.shape[-1] - 1)
            top_vals, top_ids = torch.topk(local_probs_no_gt, k=k, dim=-1)
            # Suppress only actual probability mass; -log(1-p) is stable and bounded below by 0.
            penalties = []
            for row in range(top_ids.shape[0]):
                weights = []
                for tok in top_ids[row].detach().cpu().tolist():
                    weights.append(_wrong_token_weight(tokenizer, int(tok), str(st.meta.get("answer_type", "short_text")), gcfg))
                w = torch.tensor(weights, dtype=top_vals.dtype, device=device)
                penalties.append((w * (-(1.0 - top_vals[row].clamp(max=1 - eps)).clamp_min(eps).log())).mean())
                neg_probs.append(float(top_vals[row].detach().mean().item()))
            if penalties:
                state_loss = state_loss + neg_weight * torch.stack(penalties).mean() * float(st.state_weight)

        losses.append(state_loss)

    loss = torch.stack(losses).mean()
    return loss, {
        "guided_states": float(len(states)),
        "guided_targets": float(target_count),
        "pos_logp": float(sum(pos_logps) / max(1, len(pos_logps))),
        "neg_prob": float(sum(neg_probs) / max(1, len(neg_probs))) if neg_probs else 0.0,
    }
