from __future__ import annotations

"""Sample-chain aligned rollout RL for QRA SEDD fine-tuning.

Important design point:
  This trainer now matches ``generate_sample_chain.py`` and the SEDD paper's
  tau-leaping style sampler.  At each fixed reverse-time transition t_k ->
  t_{k+1}, it builds a transition distribution for every answer position and
  samples all answer positions in parallel.  A single t step can therefore
  change multiple answer slots, exactly as the sample-chain CSV shows.

The old one-action rollout was too different from SEDD sampling and could get
stuck in mask/no-op behavior.  The critical fix is using the true delta_t from
our fixed t grid, not a tiny constant step_size=1e-5.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

import importlib
import torch

from answer_specs import extract_answer_section, parse_answer_spec
from slot_alignment_reward import normalize_advantages, reward_to_go
from state_builder import encode_sample, mask_id_from_graph, project_fixed_, transition_probs
from t_schedule import rollout_t_grid, transition_times_from_grid


@dataclass
class RolloutAction:
    answer_index: int
    position: int
    token_id: int
    token_text: str
    source: str = "per_position"


@dataclass
class RolloutRecord:
    step: int
    t: float
    t_next: float
    delta_t: float
    x_before_ids: List[int]
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
    apply_anchor: bool = False
    changed_count: int = 1




def resolve_reward_function(cfg: Dict[str, Any]) -> Callable[..., Tuple[float, Dict[str, Any]]]:
    """Resolve the rollout reward from config without overwriting reward files.

    Config options under ``rollout``:
      reward_name: slot_alignment | qra_refine | <python_module>
      reward_module: optional explicit python module name

    Examples:
      reward_name: slot_alignment  -> rl_qra_pipeline/slot_alignment_reward.py
      reward_name: qra_refine      -> rl_qra_pipeline/qra_refine_reward.py
      reward_module: my_reward     -> import my_reward.step_alignment_reward
    """
    r_cfg = cfg.get("rollout", {}) or {}
    name = str(
        r_cfg.get("reward_module")
        or r_cfg.get("reward_name")
        or r_cfg.get("reward_type")
        or "slot_alignment"
    ).strip()
    aliases = {
        "": "slot_alignment_reward",
        "slot": "slot_alignment_reward",
        "slot_alignment": "slot_alignment_reward",
        "pretrain": "slot_alignment_reward",
        "pretrain_slot_alignment": "slot_alignment_reward",
        "qra": "qra_refine_reward",
        "qra_refine": "qra_refine_reward",
        "qra_refinement": "qra_refine_reward",
    }
    module_name = aliases.get(name, name)
    if module_name.endswith(".py"):
        module_name = module_name[:-3]
    mod = importlib.import_module(module_name)
    fn = getattr(mod, "step_alignment_reward", None)
    if fn is None:
        raise AttributeError(f"Reward module {module_name!r} does not define step_alignment_reward")
    return fn


def selected_reward_name(cfg: Dict[str, Any]) -> str:
    r_cfg = cfg.get("rollout", {}) or {}
    return str(r_cfg.get("reward_name") or r_cfg.get("reward_module") or r_cfg.get("reward_type") or "slot_alignment")


def _decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return ""


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


def _local_entropy(local_probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = local_probs.float().clamp_min(0.0)
    p = p / p.sum().clamp_min(eps)
    return -(p * p.clamp_min(eps).log()).sum()


def _sample_next_state_per_position(
    probs: torch.Tensor,
    x: torch.Tensor,
    answer_positions: Sequence[int],
    mode: str,
    freeze_filled: bool,
    mask_id: int,
) -> torch.Tensor:
    """Sample next full sequence exactly like sample-chain tracing, restricted only by optional freeze_filled."""
    if str(mode) == "greedy":
        next_x = probs.argmax(dim=-1)
    else:
        flat = probs.view(-1, probs.shape[-1])
        next_x = torch.multinomial(flat.float().clamp_min(0.0), num_samples=1).view_as(x)

    if freeze_filled:
        keep = torch.zeros_like(x, dtype=torch.bool)
        for p in answer_positions:
            keep[:, int(p)] = x[:, int(p)] != int(mask_id)
        next_x = torch.where(keep, x, next_x)
    return next_x


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
    t_grid = rollout_t_grid(cfg)
    transitions = transition_times_from_grid(t_grid)
    transition_kind = str(r_cfg.get("transition_kind", "analytic"))
    mode = str(r_cfg.get("mode", "sample"))
    reward_fn = resolve_reward_function(cfg)
    reward_clip = float(r_cfg.get("reward_clip", 1.0))
    freeze_filled = bool(r_cfg.get("freeze_filled", False))
    eps = float(r_cfg.get("eps", 1e-12))

    for step, (t_now, t_next, delta_t) in enumerate(transitions):
        t = torch.tensor([[float(t_now)]], dtype=torch.float32, device=device)
        x_before_full = [int(v) for v in x[0].detach().cpu().tolist()]
        before_ids = _answer_ids_from_state(x[0], answer_positions)
        before_answer = _decode_answer_ids(tokenizer, before_ids, mask_id)

        # Key fix: use the actual grid delta_t.  This matches generate_sample_chain.
        probs = transition_probs(
            model,
            graph,
            noise,
            x,
            t,
            float(delta_t),
            transition_kind,
            train=train,
            fixed_locs=encoded.layout.fixed_locs,
            fixed_ids=encoded.layout.fixed_ids.to(device),
        )
        anchor_loss = _anchor_loss_from_probs(probs, answer_positions, gt_answer_ids, float(t_now), cfg)

        x_next = _sample_next_state_per_position(
            probs=probs,
            x=x,
            answer_positions=answer_positions,
            mode=mode,
            freeze_filled=freeze_filled,
            mask_id=mask_id,
        )
        project_fixed_(x_next, encoded.layout.fixed_locs, encoded.layout.fixed_ids.to(device))

        after_ids = _answer_ids_from_state(x_next[0], answer_positions)
        after_answer = _decode_answer_ids(tokenizer, after_ids, mask_id)
        changed = [i for i, (a, b) in enumerate(zip(before_ids, after_ids)) if int(a) != int(b)]
        changed_count = max(1, len(changed))

        # One t transition can create many position-token actions.
        for n, ans_idx in enumerate(changed):
            pos = int(answer_positions[ans_idx])
            tok = int(after_ids[ans_idx])
            local_probs = probs[0, pos, :]
            logprob = local_probs[tok].clamp_min(eps).log()
            entropy = _local_entropy(local_probs, eps=eps)
            reward, parts = reward_fn(
                before_ids=before_ids,
                after_ids=after_ids,
                gt_ids=gt_answer_ids,
                tokenizer=tokenizer,
                mask_id=mask_id,
                t=float(t_now),
                answer_kind=gt_spec.type,
                reward_clip=reward_clip,
                action_answer_index=int(ans_idx),
                action_token_id=int(tok),
            )
            # Avoid multiplying state-level skeleton/exact deltas too much when several positions changed.
            if changed_count > 1:
                reward = float(reward) / float(changed_count ** 0.5)
                parts = dict(parts)
                parts["reward_scaled_for_changed_count"] = float(reward)
                parts["changed_count"] = int(changed_count)

            records.append(
                RolloutRecord(
                    step=int(step),
                    t=float(t_now),
                    t_next=float(t_next),
                    delta_t=float(delta_t),
                    x_before_ids=x_before_full,
                    before_ids=before_ids,
                    after_ids=after_ids,
                    before_answer=before_answer,
                    after_answer=after_answer,
                    action=RolloutAction(
                        answer_index=int(ans_idx),
                        position=int(pos),
                        token_id=int(tok),
                        token_text=_decode_token(tokenizer, tok),
                        source="per_position",
                    ),
                    logprob=logprob,
                    entropy=entropy,
                    reward=float(reward),
                    reward_parts=parts,
                    anchor_loss=anchor_loss if n == 0 else torch.zeros((), device=device, dtype=probs.dtype),
                    apply_anchor=(n == 0),
                    changed_count=int(changed_count),
                )
            )

        # Discrete state transition.  Policy gradient goes through logprob, not through x_next.
        x = x_next.detach()

    return records, {
        "sample_id": sample.get("id", ""),
        "gt_answer": gt_answer,
        "answer_type": gt_spec.type,
        "answer_len": len(answer_positions),
        "t_grid": [float(x) for x in t_grid],
        "num_transitions": len(transitions),
        "reward_name": selected_reward_name(cfg),
    }


def _compute_returns_advantages(rewards: Sequence[float], cfg: Dict[str, Any], device: torch.device, dtype=torch.float32) -> torch.Tensor:
    r_cfg = cfg.get("rollout", {})
    gamma = float(r_cfg.get("gamma", 0.95))
    baseline = str(r_cfg.get("baseline", "raw_clip"))
    clip = float(r_cfg.get("advantage_clip", 0.25))

    # QRA refinement: keep the sign of local rewards. With reward_to_go_norm,
    # a good action can become negative after within-rollout normalization,
    # which is bad when we only want small corrections to an already-good QRA.
    if baseline in {"raw_clip", "raw", "step_raw"}:
        vals = torch.tensor([float(r) for r in rewards], device=device, dtype=dtype)
        return vals.clamp(-clip, clip) if clip and clip > 0 else vals
    if baseline in {"reward_to_go_clip", "rtg_clip"}:
        vals = torch.tensor(reward_to_go(rewards, gamma=gamma), device=device, dtype=dtype)
        return vals.clamp(-clip, clip) if clip and clip > 0 else vals
    if baseline in {"step_norm", "step"}:
        values = [float(r) for r in rewards]
    else:
        values = reward_to_go(rewards, gamma=gamma)
    return normalize_advantages(values, clip=clip, device=device, dtype=dtype)

def _recompute_action_terms_from_state(
    model,
    graph,
    noise,
    tokenizer,
    sample: Dict[str, Any],
    cfg: Dict[str, Any],
    device: torch.device,
    rec: RolloutRecord,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replay one recorded per-position action and rebuild differentiable logprob."""
    model_cfg = cfg.get("model", {})
    r_cfg = cfg.get("rollout", {})
    max_length = int(model_cfg.get("max_length", 1024))
    encoded = encode_sample(sample, tokenizer, max_length, tokenizer.eos_token_id)
    answer_positions = [p for p in encoded.layout.answer_positions if p < encoded.layout.real_len]
    gt_answer = extract_answer_section(encoded.reference_completion)
    gt_spec = parse_answer_spec(gt_answer)
    gt_answer_ids = [int(encoded.ids[p]) for p in answer_positions]

    x = torch.tensor([rec.x_before_ids], dtype=torch.long, device=device)
    t = torch.tensor([[float(rec.t)]], dtype=torch.float32, device=device)
    probs = transition_probs(
        model,
        graph,
        noise,
        x,
        t,
        float(rec.delta_t),
        str(r_cfg.get("transition_kind", "analytic")),
        train=True,
        fixed_locs=encoded.layout.fixed_locs,
        fixed_ids=encoded.layout.fixed_ids.to(device),
    )
    eps = float(r_cfg.get("eps", 1e-12))
    local_probs = probs[0, int(rec.action.position), :]
    logprob = local_probs[int(rec.action.token_id)].clamp_min(eps).log()
    entropy = _local_entropy(local_probs, eps=eps)
    if rec.apply_anchor:
        anchor_loss = _anchor_loss_from_probs(probs, answer_positions, gt_answer_ids, float(rec.t), cfg)
    else:
        anchor_loss = torch.zeros((), device=device, dtype=probs.dtype)
    return logprob, entropy, anchor_loss


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
    memory_safe_replay = bool(r_cfg.get("memory_safe_replay", True))

    all_losses: List[torch.Tensor] = []
    all_rewards: List[float] = []
    all_logprobs: List[float] = []
    all_entropies: List[float] = []
    all_anchor_losses: List[float] = []
    debug_records: List[Dict[str, Any]] = []

    for ridx in range(max(1, num_rollouts)):
        if memory_safe_replay and torch.is_grad_enabled():
            with torch.no_grad():
                records, meta = rollout_answer_chain_with_logprob(
                    model=model,
                    graph=graph,
                    noise=noise,
                    tokenizer=tokenizer,
                    sample=sample,
                    cfg=cfg,
                    device=device,
                    train=False,
                )
        else:
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

        if not records:
            # No changed action in this rollout.  This can happen rarely if the transition sampled all stay-current.
            # We skip it rather than inventing an off-policy action.
            continue

        rewards = [float(r.reward) for r in records]
        adv = _compute_returns_advantages(rewards, cfg, device=device, dtype=torch.float32)
        if adv.numel() != len(records):
            continue

        if memory_safe_replay and torch.is_grad_enabled():
            # When train_rl_qra.py accumulates multiple different samples before one
            # optimizer.step(), memory_safe_replay still calls backward inside this
            # function.  loss_normalizer keeps the effective update size invariant:
            # sample 8/16/32 rollouts => average their gradients, not multiply LR.
            loss_normalizer = float(r_cfg.get("loss_normalizer", 1.0) or 1.0)
            denom = max(1, len(records)) * max(1, num_rollouts) * max(1.0, loss_normalizer)
            rollout_loss_value = 0.0
            for rec, a in zip(records, adv):
                logprob, entropy, anchor_loss = _recompute_action_terms_from_state(
                    model=model,
                    graph=graph,
                    noise=noise,
                    tokenizer=tokenizer,
                    sample=sample,
                    cfg=cfg,
                    device=device,
                    rec=rec,
                )
                loss_step = (
                    policy_scale * (-a.detach() * logprob)
                    + anchor_scale * anchor_loss
                    - entropy_weight * entropy
                ) / float(denom)
                loss_step.backward()
                rollout_loss_value += float(loss_step.detach().item())
                all_rewards.append(float(rec.reward))
                all_logprobs.append(float(logprob.detach().item()))
                all_entropies.append(float(entropy.detach().item()))
                all_anchor_losses.append(float(anchor_loss.detach().item()))
            all_losses.append(torch.tensor(float(rollout_loss_value), device=device))
        else:
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
            max_steps = int(r_cfg.get("max_debug_steps", 32))
            rows = []
            for rec, a in zip(records[:max_steps], adv[:max_steps]):
                rows.append(
                    {
                        "step": rec.step,
                        "t": rec.t,
                        "t_next": rec.t_next,
                        "delta_t": rec.delta_t,
                        "before": rec.before_answer,
                        "after": rec.after_answer,
                        "action_pos": rec.action.answer_index,
                        "token": rec.action.token_text,
                        "token_id": rec.action.token_id,
                        "source": rec.action.source,
                        "changed_count": rec.changed_count,
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
    mean_logp = float(sum(all_logprobs) / max(1, len(all_logprobs)))
    mean_reward = float(sum(all_rewards) / max(1, len(all_rewards)))
    mean_entropy = float(sum(all_entropies) / max(1, len(all_entropies)))
    return loss_out, {
        "_already_backward": bool(memory_safe_replay) and torch.is_grad_enabled(),
        "guided_states": float(max(1, num_rollouts)),
        "guided_targets": float(len(all_rewards)),
        "rollout_num_rollouts": float(max(1, num_rollouts)),
        "rollout_loss_normalizer": float(r_cfg.get("loss_normalizer", 1.0) or 1.0),
        "rollout_samples_per_update": float(r_cfg.get("samples_per_update", 1.0) or 1.0),
        "rrpi_loss": float(loss_out.detach().item()),
        "rollout_loss": float(loss_out.detach().item()),
        "rollout_reward": mean_reward,
        "rollout_reward_min": float(min(all_rewards) if all_rewards else 0.0),
        "rollout_reward_max": float(max(all_rewards) if all_rewards else 0.0),
        "rollout_reward_std": float(rewards_t.std(unbiased=False).item()) if rewards_t.numel() > 1 else 0.0,
        "rollout_entropy": mean_entropy,
        "rollout_logprob": mean_logp,
        "rollout_anchor_loss": float(sum(all_anchor_losses) / max(1, len(all_anchor_losses))) if all_anchor_losses else 0.0,
        # Old plotting keys.
        "target_logp": mean_logp,
        "target_prob": 0.0,
        "model_reward": mean_reward,
        "best_reward": float(max(all_rewards) if all_rewards else 0.0),
        "reward_gap": float((max(all_rewards) - min(all_rewards)) if all_rewards else 0.0),
        "candidate_entropy": mean_entropy,
        "pos_logp": mean_logp,
        "neg_prob": 0.0,
        "debug_records": debug_records,
    }
