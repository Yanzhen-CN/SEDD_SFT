from __future__ import annotations

"""True rollout-chain RL for QRA SEDD fine-tuning.

This module performs real reverse-generation inside the answer block.  Each
sampled action is a joint (answer position, token) decision whose log-probability
comes from the SEDD ratio-induced transition probabilities.  Rewards are
stage-aware and dominated by strict slot/type alignment.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch

from answer_specs import extract_answer_section, normalize_answer, parse_answer_spec
from slot_alignment_reward import normalize_advantages, reward_to_go, step_alignment_reward, token_kind
from state_builder import encode_sample, mask_id_from_graph, project_fixed_, transition_probs


_DIGITS = list("0123456789")
_INTERVAL_SYMBOLS = ["(", "[", ")", "]", ","]
_NUMBER_SYMBOLS = ["-", "+", "."]
_UNITS = ["m", "mm", "cm", "kg", "g", "s", "ms", "N", "J", "W", "V", "A", "Hz", "m/s", "m/s^2"]
_EQUATION_SYMBOLS = ["=", "+", "-", "*", "/", "^", "(", ")", "x", "y", "z", "t", "r", "v", "u", "a", "F", "m", "k", "L", "A", "pi", "sqrt"]
_OPTION_LETTERS = list("ABCDE") + list("abcde")


@dataclass
class RolloutAction:
    answer_index: int
    position: int
    token_id: int
    token_text: str
    source: str


@dataclass
class RolloutRecord:
    step: int
    t: float
    before_ids: List[int]
    after_ids: List[int]
    before_answer: str
    after_answer: str
    action: RolloutAction
    logprob: torch.Tensor
    entropy: torch.Tensor
    reward: float
    reward_parts: Dict[str, Any]
    anchor_loss: torch.Tensor | None = None


def _decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return ""


def _single_token_ids(tokenizer, texts: Sequence[str]) -> List[int]:
    out: List[int] = []
    seen = set()
    for text in texts:
        try:
            ids = tokenizer(str(text), add_special_tokens=False).input_ids
        except Exception:
            continue
        if len(ids) == 1 and int(ids[0]) not in seen:
            out.append(int(ids[0]))
            seen.add(int(ids[0]))
    return out


def _candidate_texts(answer_kind: str, target_text: str = "") -> List[str]:
    kind = str(answer_kind or "")
    texts: List[str] = []

    def add(items: Sequence[str]) -> None:
        for item in items:
            texts.append(item)
            texts.append(" " + item)

    if kind == "single_letter":
        add(_OPTION_LETTERS)
    elif kind == "single_integer":
        add(_DIGITS + ["-", "+"])
    elif kind in {"signed_decimal", "unit_decimal"}:
        add(_DIGITS + _NUMBER_SYMBOLS + _UNITS)
    elif kind == "interval":
        add(_DIGITS + _NUMBER_SYMBOLS + _INTERVAL_SYMBOLS)
    elif kind in {"equation", "symbolic_expression", "short_text"}:
        add(_DIGITS + _NUMBER_SYMBOLS + _EQUATION_SYMBOLS + _UNITS)
    else:
        add(_DIGITS + _NUMBER_SYMBOLS + _INTERVAL_SYMBOLS + _EQUATION_SYMBOLS + _UNITS + _OPTION_LETTERS)

    if target_text:
        texts.insert(0, target_text)
        stripped = normalize_answer(target_text)
        if stripped and stripped != target_text:
            texts.insert(1, stripped)
            texts.insert(2, " " + stripped)

    dedup: List[str] = []
    seen = set()
    for t in texts:
        if t not in seen:
            dedup.append(t)
            seen.add(t)
    return dedup


def _answer_ids_from_state(x_1d: torch.Tensor, answer_positions: Sequence[int]) -> List[int]:
    arr = x_1d.detach().cpu().tolist()
    return [int(arr[p]) for p in answer_positions if p < len(arr)]


def _decode_answer_ids(tokenizer, ids: Sequence[int], mask_id: int, mask_char: str = "□") -> str:
    parts: List[str] = []
    for tid in ids:
        if int(tid) == int(mask_id):
            parts.append(mask_char)
        else:
            parts.append(_decode_token(tokenizer, int(tid)))
    return "".join(parts)


def _choose_t_schedule(cfg: Dict[str, Any]) -> List[float]:
    r_cfg = cfg.get("rollout", {})
    steps = int(r_cfg.get("steps", 32))
    if steps <= 0:
        return []
    custom = r_cfg.get("t_values")
    if isinstance(custom, list) and custom:
        vals = [float(v) for v in custom]
        if len(vals) >= steps:
            return vals[:steps]
        out: List[float] = []
        for k in range(steps):
            idx = round(k * (len(vals) - 1) / max(1, steps - 1))
            out.append(float(vals[idx]))
        return out
    t_start = float(r_cfg.get("t_start", 0.95))
    t_end = float(r_cfg.get("t_end", 0.01))
    if steps == 1:
        return [t_start]
    return [t_start + (t_end - t_start) * k / max(1, steps - 1) for k in range(steps)]


def _build_candidates_for_position(
    tokenizer,
    local_probs: torch.Tensor,
    current_id: int,
    gt_id: int,
    answer_kind: str,
    cfg: Dict[str, Any],
) -> Tuple[List[int], Dict[int, str]]:
    r_cfg = cfg.get("rollout", {})
    sources: Dict[int, str] = {}
    vocab = int(local_probs.shape[-1])
    include_current = bool(r_cfg.get("include_current_candidate", False))
    allow_noop = bool(r_cfg.get("allow_noop_action", False))

    def add(ids: Sequence[int], source: str) -> None:
        for tid in ids:
            tid = int(tid)
            if not (0 <= tid < vocab):
                continue
            if not allow_noop and tid == int(current_id):
                continue
            if tid not in sources:
                sources[tid] = source

    if include_current:
        add([int(current_id)], "keep_current")
    if bool(r_cfg.get("include_gt_candidates", True)):
        add([int(gt_id)], "gt")
    target_text = _decode_token(tokenizer, int(gt_id))
    add(_single_token_ids(tokenizer, _candidate_texts(answer_kind, target_text)), "type")

    topk = int(r_cfg.get("action_topk", 32))
    if topk > 0:
        _, top_ids = torch.topk(local_probs.detach(), k=min(topk, vocab), dim=-1)
        add([int(x) for x in top_ids.detach().cpu().tolist()], "model_topk")

    # If every candidate was filtered as no-op, force GT if it changes state.
    if not sources and int(gt_id) != int(current_id):
        sources[int(gt_id)] = "gt_forced"
    if not sources:
        sources[int(current_id)] = "noop_forced"

    max_cands = int(r_cfg.get("max_candidates_per_position", 48))
    ordered = list(sources.keys())
    if max_cands > 0 and len(ordered) > max_cands:
        protected: List[int] = []
        for x in [int(gt_id)]:
            if x in sources and x not in protected:
                protected.append(x)
        if include_current and int(current_id) in sources and int(current_id) not in protected:
            protected.append(int(current_id))
        rest = [x for x in ordered if x not in set(protected)]
        rest = sorted(rest, key=lambda z: float(local_probs[int(z)].detach().item()), reverse=True)
        ordered = protected + rest[: max(0, max_cands - len(protected))]
    return ordered, sources


def _make_all_mask_answer_state(encoded, graph, device: torch.device) -> torch.Tensor:
    x = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    mask_id = int(mask_id_from_graph(graph))
    answer_positions = [p for p in encoded.layout.answer_positions if p < encoded.layout.real_len]
    if answer_positions:
        x[:, answer_positions] = mask_id
    project_fixed_(x, encoded.layout.fixed_locs, encoded.layout.fixed_ids.to(device))
    return x


def _anchor_loss_from_probs(
    probs: torch.Tensor,
    answer_positions: Sequence[int],
    gt_answer_ids: Sequence[int],
    t_value: float,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    r_cfg = cfg.get("rollout", {})
    eps = float(r_cfg.get("eps", 1e-12))
    power = float(r_cfg.get("anchor_late_power", 1.0))
    weight_t = max(0.0, min(1.0, 1.0 - float(t_value))) ** max(0.0, power)
    terms: List[torch.Tensor] = []
    for ans_idx, pos in enumerate(answer_positions):
        if ans_idx >= len(gt_answer_ids):
            continue
        gid = int(gt_answer_ids[ans_idx])
        terms.append(-probs[0, int(pos), gid].clamp_min(eps).log() * float(weight_t))
    if not terms:
        return torch.zeros((), device=probs.device, dtype=probs.dtype)
    return torch.stack(terms).mean()


def _build_joint_action_distribution(
    tokenizer,
    probs: torch.Tensor,
    x: torch.Tensor,
    answer_positions: Sequence[int],
    gt_answer_ids: Sequence[int],
    answer_kind: str,
    cfg: Dict[str, Any],
) -> Tuple[torch.distributions.Categorical, List[RolloutAction], torch.Tensor]:
    r_cfg = cfg.get("rollout", {})
    temperature = max(1e-6, float(r_cfg.get("sample_temperature", 1.0)))
    eps = float(r_cfg.get("eps", 1e-12))

    logits: List[torch.Tensor] = []
    actions: List[RolloutAction] = []
    x1 = x[0]

    for ans_idx, pos in enumerate(answer_positions):
        if ans_idx >= len(gt_answer_ids):
            continue
        local_probs = probs[0, int(pos), :]
        current_id = int(x1[int(pos)].detach().item())
        gt_id = int(gt_answer_ids[ans_idx])
        cand_ids, sources = _build_candidates_for_position(tokenizer, local_probs, current_id, gt_id, answer_kind, cfg)
        for tid in cand_ids:
            tid = int(tid)
            logit = local_probs[tid].clamp_min(eps).log() / temperature
            logits.append(logit)
            actions.append(
                RolloutAction(
                    answer_index=int(ans_idx),
                    position=int(pos),
                    token_id=int(tid),
                    token_text=_decode_token(tokenizer, tid),
                    source=str(sources.get(tid, "candidate")),
                )
            )

    if not logits:
        raise RuntimeError("No rollout candidate actions were built.")
    logits_t = torch.stack(logits)
    dist = torch.distributions.Categorical(logits=logits_t)
    return dist, actions, logits_t


def rollout_answer_chain_with_logprob(
    model,
    graph,
    noise,
    tokenizer,
    sample: Dict[str, Any],
    cfg: Dict[str, Any],
    device: torch.device,
    train: bool = True,
) -> Tuple[List[RolloutRecord], Dict[str, Any]]:
    model_cfg = cfg.get("model", {})
    r_cfg = cfg.get("rollout", {})
    max_length = int(model_cfg.get("max_length", 1024))
    encoded = encode_sample(sample, tokenizer, max_length, tokenizer.eos_token_id)
    answer_positions = [p for p in encoded.layout.answer_positions if p < encoded.layout.real_len]
    if not answer_positions:
        return [], {"reason": "no_answer_positions"}

    gt_answer = extract_answer_section(encoded.reference_completion)
    gt_spec = parse_answer_spec(gt_answer)
    gt_answer_ids = [int(encoded.ids[p]) for p in answer_positions]
    mask_id = int(mask_id_from_graph(graph))
    x = _make_all_mask_answer_state(encoded, graph, device)

    records: List[RolloutRecord] = []
    t_values = _choose_t_schedule(cfg)
    step_size = float(r_cfg.get("step_size", 1e-5))
    transition_kind = str(r_cfg.get("transition_kind", "analytic"))
    mode = str(r_cfg.get("mode", "sample"))
    reward_clip = float(r_cfg.get("reward_clip", 1.0))
    freeze_filled = bool(r_cfg.get("freeze_filled", False))

    for step, t_value in enumerate(t_values):
        t = torch.tensor([[float(t_value)]], dtype=torch.float32, device=device)
        before_ids = _answer_ids_from_state(x[0], answer_positions)
        before_answer = _decode_answer_ids(tokenizer, before_ids, mask_id)

        probs = transition_probs(
            model,
            graph,
            noise,
            x,
            t,
            step_size,
            transition_kind,
            train=train,
            fixed_locs=encoded.layout.fixed_locs,
            fixed_ids=encoded.layout.fixed_ids.to(device),
        )

        if freeze_filled:
            fixed_now = list(encoded.layout.fixed_locs)
            fixed_ids_now = [int(encoded.ids[p]) for p in encoded.layout.fixed_locs]
            for pos in answer_positions:
                if int(x[0, pos].detach().item()) != mask_id:
                    fixed_now.append(int(pos))
                    fixed_ids_now.append(int(x[0, pos].detach().item()))
            if fixed_now:
                fid = torch.tensor([fixed_ids_now], dtype=torch.long, device=device)
                probs[:, fixed_now, :] = 0.0
                probs[0, fixed_now, fid[0]] = 1.0

        anchor_loss = _anchor_loss_from_probs(probs, answer_positions, gt_answer_ids, float(t_value), cfg)
        dist, actions, _ = _build_joint_action_distribution(
            tokenizer=tokenizer,
            probs=probs,
            x=x,
            answer_positions=answer_positions,
            gt_answer_ids=gt_answer_ids,
            answer_kind=gt_spec.type,
            cfg=cfg,
        )
        if mode == "greedy":
            action_idx = torch.argmax(dist.logits.detach())
        else:
            action_idx = dist.sample()
        logprob = dist.log_prob(action_idx)
        entropy = dist.entropy()
        action = actions[int(action_idx.detach().cpu().item())]

        x_next = x.clone()
        x_next[0, action.position] = int(action.token_id)
        project_fixed_(x_next, encoded.layout.fixed_locs, encoded.layout.fixed_ids.to(device))

        after_ids = _answer_ids_from_state(x_next[0], answer_positions)
        after_answer = _decode_answer_ids(tokenizer, after_ids, mask_id)
        reward, parts = step_alignment_reward(
            before_ids=before_ids,
            after_ids=after_ids,
            gt_ids=gt_answer_ids,
            tokenizer=tokenizer,
            mask_id=mask_id,
            t=float(t_value),
            answer_kind=gt_spec.type,
            reward_clip=reward_clip,
            action_answer_index=action.answer_index,
            action_token_id=action.token_id,
        )

        records.append(
            RolloutRecord(
                step=int(step),
                t=float(t_value),
                before_ids=before_ids,
                after_ids=after_ids,
                before_answer=before_answer,
                after_answer=after_answer,
                action=action,
                logprob=logprob,
                entropy=entropy,
                reward=float(reward),
                reward_parts=parts,
                anchor_loss=anchor_loss,
            )
        )
        # State is discrete.  Policy-gradient carries gradient through logprob only.
        x = x_next.detach()

    return records, {
        "sample_id": sample.get("id", ""),
        "gt_answer": gt_answer,
        "answer_type": gt_spec.type,
        "answer_len": len(answer_positions),
    }


def _compute_returns_advantages(rewards: Sequence[float], cfg: Dict[str, Any], device: torch.device, dtype=torch.float32) -> torch.Tensor:
    r_cfg = cfg.get("rollout", {})
    gamma = float(r_cfg.get("gamma", 0.95))
    baseline = str(r_cfg.get("baseline", "reward_to_go_norm"))
    clip = float(r_cfg.get("advantage_clip", 0.25))
    if baseline in {"step_norm", "step"}:
        values = [float(r) for r in rewards]
    else:
        values = reward_to_go(rewards, gamma=gamma)
    return normalize_advantages(values, clip=clip, device=device, dtype=dtype)


def rollout_chain_loss(
    model,
    graph,
    noise,
    tokenizer,
    sample: Dict[str, Any],
    cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    r_cfg = cfg.get("rollout", {})
    num_rollouts = int(r_cfg.get("num_rollouts", 1))
    policy_scale = float(r_cfg.get("policy_loss_scale", 0.02))
    entropy_weight = float(r_cfg.get("entropy_weight", 0.0005))
    anchor_scale = float(r_cfg.get("sft_anchor_scale", 0.005))

    all_losses: List[torch.Tensor] = []
    all_rewards: List[float] = []
    all_logprobs: List[float] = []
    all_entropies: List[float] = []
    all_anchor_losses: List[float] = []
    debug_records: List[Dict[str, Any]] = []
    meta_last: Dict[str, Any] = {}

    for ridx in range(max(1, num_rollouts)):
        records, meta = rollout_answer_chain_with_logprob(
            model=model,
            graph=graph,
            noise=noise,
            tokenizer=tokenizer,
            sample=sample,
            cfg=cfg,
            device=device,
            train=torch.is_grad_enabled(),
        )
        meta_last = meta
        if not records:
            continue
        rewards = [float(r.reward) for r in records]
        adv = _compute_returns_advantages(rewards, cfg, device=device, dtype=records[0].logprob.dtype)
        if adv.numel() != len(records):
            continue

        pg_terms: List[torch.Tensor] = []
        entropy_terms: List[torch.Tensor] = []
        anchor_terms: List[torch.Tensor] = []
        for rec, a in zip(records, adv):
            pg_terms.append(-a.detach() * rec.logprob)
            entropy_terms.append(rec.entropy)
            if rec.anchor_loss is not None:
                anchor_terms.append(rec.anchor_loss)
            all_rewards.append(float(rec.reward))
            all_logprobs.append(float(rec.logprob.detach().item()))
            all_entropies.append(float(rec.entropy.detach().item()))
            if rec.anchor_loss is not None:
                all_anchor_losses.append(float(rec.anchor_loss.detach().item()))

        pg_loss = torch.stack(pg_terms).mean()
        entropy = torch.stack(entropy_terms).mean()
        anchor_loss = torch.stack(anchor_terms).mean() if anchor_terms else torch.zeros((), device=device, dtype=pg_loss.dtype)
        loss = policy_scale * pg_loss + anchor_scale * anchor_loss - entropy_weight * entropy
        all_losses.append(loss)

        if len(debug_records) < int(r_cfg.get("debug_records_per_step", 2)):
            max_steps = int(r_cfg.get("max_debug_steps", 12))
            rows = []
            for rec, a in zip(records[:max_steps], adv[:max_steps]):
                rows.append(
                    {
                        "step": rec.step,
                        "t": rec.t,
                        "before": rec.before_answer,
                        "after": rec.after_answer,
                        "action_pos": rec.action.answer_index,
                        "token": rec.action.token_text,
                        "token_id": rec.action.token_id,
                        "source": rec.action.source,
                        "reward": rec.reward,
                        "advantage": float(a.detach().item()),
                        "logprob": float(rec.logprob.detach().item()),
                        "anchor_loss": float(rec.anchor_loss.detach().item()) if rec.anchor_loss is not None else 0.0,
                        **{f"r_{k}": v for k, v in rec.reward_parts.items()},
                    }
                )
            debug_records.append({**meta, "rollout_index": ridx, "chain": rows})

    if not all_losses:
        z = torch.zeros((), device=device, requires_grad=True)
        return z, {
            "guided_states": 0.0,
            "guided_targets": 0.0,
            "rrpi_loss": 0.0,
            "rollout_loss": 0.0,
            "rollout_reward": 0.0,
            "rollout_entropy": 0.0,
            "rollout_logprob": 0.0,
            "rollout_anchor_loss": 0.0,
            "debug_records": [],
        }

    loss_out = torch.stack(all_losses).mean()
    rewards_t = torch.tensor(all_rewards, dtype=torch.float32) if all_rewards else torch.zeros(1)
    return loss_out, {
        "guided_states": float(num_rollouts),
        "guided_targets": float(len(all_rewards)),
        "rrpi_loss": float(loss_out.detach().item()),
        "rollout_loss": float(loss_out.detach().item()),
        "rollout_reward": float(sum(all_rewards) / max(1, len(all_rewards))),
        "rollout_reward_min": float(min(all_rewards) if all_rewards else 0.0),
        "rollout_reward_max": float(max(all_rewards) if all_rewards else 0.0),
        "rollout_reward_std": float(rewards_t.std(unbiased=False).item()) if rewards_t.numel() > 1 else 0.0,
        "rollout_entropy": float(sum(all_entropies) / max(1, len(all_entropies))),
        "rollout_logprob": float(sum(all_logprobs) / max(1, len(all_logprobs))),
        "rollout_anchor_loss": float(sum(all_anchor_losses) / max(1, len(all_anchor_losses))) if all_anchor_losses else 0.0,
        # Keep old plotting keys alive.
        "target_logp": float(sum(all_logprobs) / max(1, len(all_logprobs))),
        "target_prob": 0.0,
        "model_reward": float(sum(all_rewards) / max(1, len(all_rewards))),
        "best_reward": float(max(all_rewards) if all_rewards else 0.0),
        "reward_gap": float((max(all_rewards) - min(all_rewards)) if all_rewards else 0.0),
        "candidate_entropy": float(sum(all_entropies) / max(1, len(all_entropies))),
        "pos_logp": float(sum(all_logprobs) / max(1, len(all_logprobs))),
        "neg_prob": 0.0,
        "debug_records": debug_records,
    }
