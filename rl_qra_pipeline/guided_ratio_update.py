from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple, Any

import torch
import torch.nn.functional as F

from answer_specs import (
    AnswerSpec,
    extract_answer_section,
    final_answer_score,
    normalize_answer,
    parse_answer_spec,
)
from state_builder import (
    EncodedSample,
    decode_positions,
    encode_sample,
    mask_id_from_graph,
    project_fixed_,
    safe_decode_ids,
    transition_probs,
)


@dataclass
class GuidedRatioState:
    """One local RRPI training state.

    x_state is the actual SEDD input state.  The reward, however, is computed by
    enumerating candidate answer actions in the reference answer context.  This
    makes the reward->target-policy link explicit and avoids high-variance full
    reverse-diffusion rollout.
    """

    x_state: torch.Tensor  # [L], CPU long tensor
    t: float
    step_size: float
    target_positions: List[int]
    target_ids: List[int]
    answer_positions: List[int]
    reference_answer_ids: List[int]
    gt_answer: str
    gt_spec: AnswerSpec
    state_kind: str
    state_weight: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateRow:
    token_id: int
    token_text: str
    candidate_answer: str
    reward: float
    q_target: float
    pi_model: float
    is_gt: bool
    source: str


_DIGITS = list("0123456789")
_LETTERS = list("ABCDE")
_INTERVAL_SYMBOLS = ["(", "[", ")", "]", ","]
_NUMBER_SYMBOLS = ["-", "+", "."]
_CLEAN_SYMBOLS = ["\n", ".", " ", ""]
_BAD_CONTINUATIONS = [" The", " Therefore", " Answer", " because"]


def _decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return ""


def _single_token_ids(tokenizer, texts: Sequence[str]) -> List[int]:
    ids: List[int] = []
    seen = set()
    for text in texts:
        try:
            toks = tokenizer(str(text), add_special_tokens=False).input_ids
        except Exception:
            continue
        if len(toks) == 1 and toks[0] not in seen:
            ids.append(int(toks[0]))
            seen.add(int(toks[0]))
    return ids


def _candidate_texts_for_type(gt_type: str, target_text: str, cfg: Dict[str, Any]) -> List[str]:
    """Small semantically meaningful candidate action set.

    We include leading-space variants because GPT-2 BPE often encodes short
    answers as tokens like ' 4' or ' A' depending on the Answer: anchor.
    """
    texts: List[str] = []
    raw = target_text
    stripped = normalize_answer(raw)
    has_leading_space = bool(raw.startswith(" "))

    def add_variants(items: Sequence[str]) -> None:
        for item in items:
            texts.append(item)
            texts.append(" " + item)

    if gt_type == "single_letter":
        add_variants(_LETTERS + [x.lower() for x in _LETTERS])
    elif gt_type == "single_integer":
        add_variants(_DIGITS + _NUMBER_SYMBOLS)
    elif gt_type == "signed_decimal":
        add_variants(_DIGITS + _NUMBER_SYMBOLS)
    elif gt_type == "interval":
        add_variants(_DIGITS + _NUMBER_SYMBOLS + _INTERVAL_SYMBOLS)
    else:
        # Short text: keep this conservative and let GT/current/model-topk carry it.
        add_variants([stripped] if stripped else [])

    # Clean-output candidates help positions after the short answer learn to stop.
    if bool(cfg.get("include_clean_candidates", True)):
        texts.extend(_CLEAN_SYMBOLS)
        texts.extend(_BAD_CONTINUATIONS if bool(cfg.get("include_bad_continuations", False)) else [])

    # Preserve the exact target token spelling.
    if raw:
        texts.insert(0, raw)
    if stripped and stripped != raw:
        texts.insert(1, stripped)
    if has_leading_space and stripped:
        texts.insert(2, " " + stripped)

    # Deduplicate in order.
    out: List[str] = []
    seen = set()
    for t in texts:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _choose_t(cfg: Dict[str, Any]) -> float:
    vals = cfg.get("t_values")
    if isinstance(vals, list) and vals:
        return float(random.choice(vals))
    return float(random.uniform(float(cfg.get("t_min", 0.05)), float(cfg.get("t_max", 0.95))))


def _corrupt_positions_(x: torch.Tensor, positions: Sequence[int], graph, cfg: Dict[str, Any], device: torch.device) -> torch.Tensor:
    if not positions:
        return x
    mode = str(cfg.get("corrupt_mode", "mask"))
    if mode == "limit":
        base = graph.sample_limit(1, x.shape[1]).to(device)
        x[:, list(positions)] = base[:, list(positions)]
    else:
        x[:, list(positions)] = int(mask_id_from_graph(graph))
    return x


def _base_state(encoded: EncodedSample, graph, cfg: Dict[str, Any], device: torch.device) -> torch.Tensor:
    ids_t = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    mode = str(cfg.get("base_state", "reference_answer_masked"))
    if mode == "limit_with_fixed_prompt":
        x = graph.sample_limit(1, len(encoded.ids)).to(device)
        project_fixed_(x, encoded.layout.fixed_locs, encoded.layout.fixed_ids.to(device))
        return x
    return ids_t.clone()


def _reference_answer_ids(encoded: EncodedSample, answer_pos: Sequence[int]) -> List[int]:
    return [int(encoded.ids[p]) for p in answer_pos]


def build_guided_ratio_states(sample: Dict, tokenizer, graph, cfg: Dict, device: torch.device) -> Tuple[List[GuidedRatioState], Dict]:
    """Build local states for RRPI.

    Difference from the old guided update:
    - old: GT token is hard positive, full-vocab top-k are negatives;
    - new: each candidate token is turned into a candidate answer and scored by
      the verifier, then reward softmax defines the target policy q(a).
    """
    model_cfg = cfg.get("model", {})
    rrpi_cfg = cfg.get("rrpi", cfg.get("guided", {}))
    max_length = int(model_cfg.get("max_length", 1024))
    encoded = encode_sample(sample, tokenizer, max_length, tokenizer.eos_token_id)
    answer_pos = [p for p in encoded.layout.answer_positions if p < encoded.layout.real_len]
    if not answer_pos:
        return [], {"reason": "no_answer_positions"}

    gt_answer = extract_answer_section(encoded.reference_completion)
    gt_spec = parse_answer_spec(gt_answer)
    ans_n = len(answer_pos)

    n_states = int(rrpi_cfg.get("states_per_sample", 2))
    max_targets = int(rrpi_cfg.get("max_targets_per_state", 8))
    reveal_mode = str(rrpi_cfg.get("reveal_mode", "prefix_grid"))
    step_size = float(rrpi_cfg.get("step_size", 1e-5))
    ref_answer_ids = _reference_answer_ids(encoded, answer_pos)

    states: List[GuidedRatioState] = []
    for sidx in range(max(1, n_states)):
        x = _base_state(encoded, graph, rrpi_cfg, device)
        if reveal_mode == "random_prefix":
            reveal = random.randint(0, max(0, ans_n - 1))
        elif reveal_mode == "random_subset":
            reveal = 0
        else:
            reveal = int(round((sidx / max(1, n_states - 1)) * max(0, ans_n - 1))) if n_states > 1 else 0
        reveal = max(0, min(reveal, ans_n))

        if reveal_mode == "random_subset" and ans_n > 1:
            revealed = set(random.sample(answer_pos, k=random.randint(0, ans_n - 1)))
            corrupt = [p for p in answer_pos if p not in revealed]
            targets = corrupt[:]
            kind = f"{gt_spec.type}:random_subset"
        else:
            corrupt = answer_pos[reveal:]
            targets = corrupt[:]
            kind = f"{gt_spec.type}:prefix_reveal_{reveal}"

        _corrupt_positions_(x, corrupt, graph, rrpi_cfg, device)
        project_fixed_(x, encoded.layout.fixed_locs, encoded.layout.fixed_ids.to(device))

        if max_targets > 0 and len(targets) > max_targets:
            # Prefer non-whitespace target tokens, then fill with sampled positions.
            non_ws = []
            for p in targets:
                tok = _decode_token(tokenizer, encoded.ids[p])
                if tok.strip():
                    non_ws.append(p)
            if len(non_ws) >= max_targets:
                targets = sorted(random.sample(non_ws, max_targets))
            else:
                rest = [p for p in targets if p not in set(non_ws)]
                need = max_targets - len(non_ws)
                targets = sorted(non_ws + random.sample(rest, min(need, len(rest))))
        if not targets:
            continue

        target_ids = [int(encoded.ids[p]) for p in targets]
        states.append(
            GuidedRatioState(
                x_state=x[0].detach().cpu(),
                t=_choose_t(rrpi_cfg),
                step_size=step_size,
                target_positions=targets,
                target_ids=target_ids,
                answer_positions=answer_pos,
                reference_answer_ids=ref_answer_ids,
                gt_answer=gt_answer,
                gt_spec=gt_spec,
                state_kind=kind,
                state_weight=1.0,
                meta={"answer_type": gt_spec.type, "answer_len": ans_n, "reveal": reveal},
            )
        )
    return states, {"answer_type": gt_spec.type, "num_states": len(states), "answer_len": ans_n}


def _candidate_ids_for_position(
    tokenizer,
    local_probs: torch.Tensor,
    target_id: int,
    gt_type: str,
    cfg: Dict[str, Any],
) -> Tuple[List[int], Dict[int, str]]:
    sources: Dict[int, str] = {}

    def add_ids(ids: Sequence[int], source: str) -> None:
        vocab_dim = int(local_probs.shape[-1])
        for tid in ids:
            tid = int(tid)
            if 0 <= tid < vocab_dim and tid not in sources:
                sources[tid] = source

    add_ids([target_id], "gt")
    target_text = _decode_token(tokenizer, int(target_id))
    add_ids(_single_token_ids(tokenizer, _candidate_texts_for_type(gt_type, target_text, cfg)), "type")

    topk = int(cfg.get("include_model_topk", 4))
    if topk > 0:
        vals, ids = torch.topk(local_probs.detach(), k=min(topk, local_probs.shape[-1]), dim=-1)
        add_ids([int(x) for x in ids.detach().cpu().tolist()], "model_topk")

    max_cands = int(cfg.get("max_candidates", 32))
    ordered = list(sources.keys())
    if max_cands > 0 and len(ordered) > max_cands:
        # Always keep GT; fill the rest by model probability among existing cands.
        rest = [x for x in ordered if x != int(target_id)]
        rest = sorted(rest, key=lambda z: float(local_probs[int(z)].detach().item()), reverse=True)
        ordered = [int(target_id)] + rest[: max_cands - 1]
    return ordered, sources


def _candidate_answer_for_action(tokenizer, answer_ids: Sequence[int], answer_index: int, token_id: int) -> str:
    arr = [int(x) for x in answer_ids]
    if 0 <= answer_index < len(arr):
        arr[answer_index] = int(token_id)
    return normalize_answer(safe_decode_ids(tokenizer, arr))


def _answer_reward(candidate_answer: str, gt_spec: AnswerSpec) -> float:
    pred_spec = parse_answer_spec(candidate_answer)
    return float(final_answer_score(pred_spec, gt_spec))


def guided_ratio_loss(model, graph, noise, tokenizer, sample: Dict, cfg: Dict, device: torch.device) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """RRPI: Ratio-Reward Policy Improvement.

    SEDD gives a ratio-derived local policy pi_theta(a | x_t, t, position).
    We enumerate type-aware candidate actions a, score each candidate answer via
    reward R(a), construct q(a)=softmax(R(a)/tau), and optimize

        L = - sum_a q(a) log pi_theta(a | x_t, t, position)

    This keeps the user's desired no-real-sampling path, while making the
    reward-to-gradient route explicit.
    """
    rrpi_cfg = cfg.get("rrpi", cfg.get("guided", {}))
    states, meta = build_guided_ratio_states(sample, tokenizer, graph, cfg, device)
    if not states:
        z = torch.zeros((), device=device, requires_grad=True)
        return z, {
            "guided_states": 0.0,
            "guided_targets": 0.0,
            "rrpi_loss": 0.0,
            "target_logp": 0.0,
            "target_prob": 0.0,
            "model_reward": 0.0,
            "best_reward": 0.0,
            "reward_gap": 0.0,
            "candidate_entropy": 0.0,
            "debug_records": [],
        }

    tau = float(rrpi_cfg.get("reward_temperature", 0.10))
    tau = max(1e-6, tau)
    eps = float(rrpi_cfg.get("eps", 1e-8))
    min_gap = float(rrpi_cfg.get("min_reward_gap", 0.0))
    sft_anchor_weight = float(rrpi_cfg.get("sft_anchor_weight", 0.05))
    transition_kind = str(rrpi_cfg.get("transition_kind", "analytic"))
    site_weight = float(rrpi_cfg.get("rrpi_weight", 1.0))
    max_debug_candidates = int(rrpi_cfg.get("max_debug_candidates", 10))

    losses: List[torch.Tensor] = []
    target_logps: List[float] = []
    target_probs: List[float] = []
    model_rewards: List[float] = []
    best_rewards: List[float] = []
    reward_gaps: List[float] = []
    entropies: List[float] = []
    target_count = 0
    debug_records: List[Dict[str, Any]] = []

    for st in states:
        x = st.x_state.to(device).unsqueeze(0)
        t = torch.tensor([[st.t]], dtype=torch.float32, device=device)
        probs = transition_probs(model, graph, noise, x, t, st.step_size, transition_kind, train=True)
        current_answer = normalize_answer(decode_positions(tokenizer, x[0].detach().cpu(), st.answer_positions))

        pos_to_answer_index = {p: i for i, p in enumerate(st.answer_positions)}
        for pos, target_id in zip(st.target_positions, st.target_ids):
            if pos not in pos_to_answer_index:
                continue
            ans_idx = pos_to_answer_index[pos]
            local_probs = probs[0, int(pos), :]
            cand_ids, cand_sources = _candidate_ids_for_position(
                tokenizer, local_probs, int(target_id), st.gt_spec.type, rrpi_cfg
            )
            if not cand_ids:
                continue

            rewards: List[float] = []
            answers: List[str] = []
            token_texts: List[str] = []
            for cid in cand_ids:
                cand_answer = _candidate_answer_for_action(tokenizer, st.reference_answer_ids, ans_idx, int(cid))
                answers.append(cand_answer)
                rewards.append(_answer_reward(cand_answer, st.gt_spec))
                token_texts.append(_decode_token(tokenizer, int(cid)))

            reward_t = torch.tensor(rewards, dtype=local_probs.dtype, device=device)
            gap = float((reward_t.max() - reward_t.min()).detach().item()) if reward_t.numel() else 0.0
            if gap < min_gap and int(target_id) not in cand_ids:
                continue

            q = F.softmax(reward_t / tau, dim=0).detach()
            cand_index = torch.tensor(cand_ids, dtype=torch.long, device=device)
            cand_probs = local_probs.index_select(0, cand_index).clamp_min(eps)
            # Main reward-improved target-policy loss over actual full-vocab probabilities.
            ce = -(q * cand_probs.log()).sum()

            # Small QRA/SFT anchor: always keep the reference target from collapsing.
            p_tgt = local_probs[int(target_id)].clamp_min(eps)
            anchor = -p_tgt.log() * sft_anchor_weight
            loss_site = (site_weight * ce + anchor) * float(st.state_weight)
            losses.append(loss_site)
            target_count += 1

            target_logps.append(float(p_tgt.detach().log().item()))
            target_probs.append(float(p_tgt.detach().item()))
            entropies.append(float((-(q * q.clamp_min(eps).log()).sum()).detach().item()))
            best_rewards.append(float(max(rewards)))
            reward_gaps.append(gap)

            with torch.no_grad():
                cand_probs_det = cand_probs.detach()
                model_choice_local = int(torch.argmax(cand_probs_det).item())
                model_rewards.append(float(rewards[model_choice_local]))

            if len(debug_records) < int(rrpi_cfg.get("debug_records_per_step", 2)):
                rows: List[CandidateRow] = []
                q_cpu = q.detach().cpu().tolist()
                p_cpu = cand_probs.detach().cpu().tolist()
                for j, cid in enumerate(cand_ids):
                    rows.append(
                        CandidateRow(
                            token_id=int(cid),
                            token_text=token_texts[j],
                            candidate_answer=answers[j],
                            reward=float(rewards[j]),
                            q_target=float(q_cpu[j]),
                            pi_model=float(p_cpu[j]),
                            is_gt=(int(cid) == int(target_id)),
                            source=str(cand_sources.get(int(cid), "candidate")),
                        )
                    )
                rows = sorted(rows, key=lambda r: (r.q_target, r.pi_model), reverse=True)[:max_debug_candidates]
                debug_records.append(
                    {
                        "sample_id": sample.get("id", ""),
                        "state_kind": st.state_kind,
                        "answer_type": st.gt_spec.type,
                        "gt_answer": st.gt_answer,
                        "current_state_answer": current_answer,
                        "position": int(pos),
                        "answer_index": int(ans_idx),
                        "target_token_id": int(target_id),
                        "target_token_text": _decode_token(tokenizer, int(target_id)),
                        "target_prob": float(p_tgt.detach().item()),
                        "target_logp": float(p_tgt.detach().log().item()),
                        "best_reward": float(max(rewards)),
                        "model_choice_reward": float(rewards[int(torch.argmax(cand_probs.detach()).item())]),
                        "reward_gap": gap,
                        "candidate_table": [r.__dict__ for r in rows],
                    }
                )

    if not losses:
        z = torch.zeros((), device=device, requires_grad=True)
        return z, {
            "guided_states": float(len(states)),
            "guided_targets": 0.0,
            "rrpi_loss": 0.0,
            "target_logp": 0.0,
            "target_prob": 0.0,
            "model_reward": 0.0,
            "best_reward": 0.0,
            "reward_gap": 0.0,
            "candidate_entropy": 0.0,
            "debug_records": debug_records,
        }

    loss = torch.stack(losses).mean()
    return loss, {
        "guided_states": float(len(states)),
        "guided_targets": float(target_count),
        "rrpi_loss": float(loss.detach().item()),
        "target_logp": float(sum(target_logps) / max(1, len(target_logps))),
        "target_prob": float(sum(target_probs) / max(1, len(target_probs))),
        "model_reward": float(sum(model_rewards) / max(1, len(model_rewards))),
        "best_reward": float(sum(best_rewards) / max(1, len(best_rewards))),
        "reward_gap": float(sum(reward_gaps) / max(1, len(reward_gaps))),
        "candidate_entropy": float(sum(entropies) / max(1, len(entropies))),
        # Backward-compatible names so old plotting scripts do not crash.
        "pos_logp": float(sum(target_logps) / max(1, len(target_logps))),
        "neg_prob": float(sum(model_rewards) / max(1, len(model_rewards))),
        "debug_records": debug_records,
    }
