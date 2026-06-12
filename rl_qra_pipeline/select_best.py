from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import GPT2TokenizerFast

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
for p in (REPO_DIR, REPO_DIR / "sft_answer_pipeline", REPO_DIR / "sft_rl_pipeline", SCRIPT_DIR):
    p_str = str(p)
    if p.exists() and p_str not in sys.path:
        sys.path.insert(0, p_str)

# Reuse the exact train/eval implementation so the selection pass measures the
# same rollout objective as training.
from train_rl_qra import (  # noqa: E402
    load_config,
    select_start_config,
    resolve_repo_path,
    choose_device,
    load_policy,
    load_samples,
    evaluate_loss,
    parse_eval_limit,
    eval_limit_label,
    metric_better,
    initial_best_value,
    get_metric_value,
    dump_json,
)
from model.ema import ExponentialMovingAverage  # noqa: E402

DEFAULT_CONFIG = SCRIPT_DIR / "rl_qra_config.yaml"


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def maybe_disable_missing_pretrain_checkpoint(cfg: Dict[str, Any], start_name: str) -> Dict[str, Any]:
    ckpt_value = str(cfg.get("model", {}).get("init_checkpoint", "") or "").strip()
    if str(start_name).lower() == "pretrain" and ckpt_value:
        ckpt_path = resolve_repo_path(ckpt_value)
        if not ckpt_path.exists() or ckpt_value.lower() in {"pretrained", "hf", "cache", "none", "null"}:
            cfg.setdefault("model", {})["init_checkpoint"] = ""
            print(f"[info] --start pretrain uses pretrained/cache weights; ignored init_checkpoint={ckpt_value!r}", flush=True)
    return cfg


def default_full_eval_split(cfg: Dict[str, Any]) -> str:
    data_cfg = cfg.get("data", {})
    if data_cfg.get("full_eval_split"):
        return str(data_cfg["full_eval_split"])
    split = str(data_cfg.get("eval_split", "val") or "val")
    if split.lower() in {"val-mini", "val_mini", "validation-mini", "validation_mini"}:
        return "val"
    return split


def load_eval_samples_for_split(cfg: Dict[str, Any], tokenizer, split: str) -> List[Dict[str, Any]]:
    samples = load_samples(cfg, split, tokenizer)
    if not samples:
        raise RuntimeError(f"No eval samples loaded for split={split!r}.")
    return samples


def load_state_into_model(model, checkpoint_path: Path, device: torch.device, ema_decay: float = 0.9999, use_ema: bool = True) -> Dict[str, Any]:
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"], strict=True)
        if use_ema and "ema" in state:
            try:
                ema = ExponentialMovingAverage(model.parameters(), decay=float(ema_decay))
                ema.load_state_dict(state["ema"])
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
            except Exception as exc:
                print(f"[warn] failed to apply EMA for {checkpoint_path}: {exc}", flush=True)
        return state
    model.load_state_dict(state, strict=True)
    return {"model": state}


def run_dirs(run_root: Path) -> List[Path]:
    roots = [run_root]
    legacy = run_root / "runs"
    if legacy.exists():
        roots.append(legacy)
    dirs: List[Path] = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for d in sorted([p for p in root.iterdir() if p.is_dir()]):
            if d.name.startswith("."):
                continue
            rp = d.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            dirs.append(d)
    return dirs


def candidate_record(run_dir: Path, ckpt: Path, kind: str, source: str = "") -> Dict[str, Any]:
    return {
        "checkpoint": ckpt,
        "checkpoint_name": ckpt.name,
        "candidate_kind": kind,
        "candidate_source": source or kind,
        "run_dir": run_dir,
        "run_id": run_dir.name,
        "metrics_json": run_dir / "metrics.json",
        "metrics_csv": run_dir / "metrics.csv",
        "run_info_json": run_dir / "run_info.json",
    }


def first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def recover_file(target: Path, sources: List[Path], label: str) -> Tuple[Optional[Path], str, bool]:
    """Return target if present; otherwise copy the first existing legacy/source file to target."""
    if target.exists():
        return target, label, False
    src = first_existing(sources)
    if src is None:
        return None, "missing", False
    try:
        shutil.copy2(src, target)
        print(f"[recover] wrote {target} from {src}", flush=True)
        return target, f"recovered_from_{src.name}", True
    except Exception as exc:
        print(f"[warn] failed to recover {target} from {src}: {exc}", flush=True)
        return src, f"fallback_{src.name}", False


def find_loss_candidates(run_root: Path, include_last: bool = False, include_mini: bool = True) -> List[Dict[str, Any]]:
    """Find per-run loss-best checkpoints.

    New layout uses <run>/best.pth and <run>/best_mini.pth.  For old runs, this
    function can recover from <run>/best_run.pth and <run>/best_mini_run.pth by
    copying them to the new names.  Root best.pth is never treated as a run.
    """
    out: List[Dict[str, Any]] = []
    seen = set()
    for d in run_dirs(run_root):
        candidates: List[Tuple[str, Path, List[Path]]] = [
            ("loss_best", d / "best.pth", [d / "best_run.pth"]),
        ]
        if include_mini:
            candidates.append(("mini_best", d / "best_mini.pth", [d / "best_mini_run.pth"]))
        if include_last:
            candidates.append(("last", d / "last.pth", [d / "last_run.pth"]))
        for kind, target, legacy_sources in candidates:
            ckpt, source, recovered = recover_file(target, legacy_sources, kind)
            if ckpt is None:
                continue
            rp = ckpt.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            rec = candidate_record(d, ckpt, "loss_candidate", source=source)
            rec["recovered_checkpoint"] = bool(recovered)
            out.append(rec)
    return out


def find_reward_candidates(run_root: Path, recover_missing: bool = True) -> List[Dict[str, Any]]:
    """One reward candidate per run: best_reward.pth if present, else last.pth.

    Recovery rule for old or earlier runs:
      - if <run>/best_reward.pth exists, use it;
      - else if old <run>/best_reward_run.pth exists, copy it to best_reward.pth;
      - else if <run>/last.pth or old <run>/last_run.pth exists, copy it to
        best_reward.pth and use that copy.

    This physically backfills per-run best_reward.pth.  Root best.pth is still
    chosen only from loss candidates; this reward path only competes for root
    best_reward.pth.
    """
    out: List[Dict[str, Any]] = []
    seen = set()
    for d in run_dirs(run_root):
        preferred = d / "best_reward.pth"
        sources = [d / "best_reward_run.pth", d / "last.pth", d / "last_run.pth"]
        recovered = False
        if preferred.exists():
            ckpt = preferred
            source = "best_reward"
        else:
            if not recover_missing:
                src = first_existing(sources)
                if src is None:
                    continue
                ckpt = src
                source = f"fallback_{src.name}"
            else:
                ckpt, source, recovered = recover_file(preferred, sources, "best_reward")
                if ckpt is None:
                    continue
        rp = ckpt.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        rec = candidate_record(d, ckpt, "reward_candidate", source=source)
        rec["recovered_best_reward"] = bool(recovered)
        rec["fallback_last"] = str(first_existing([d / "last.pth", d / "last_run.pth"]) or "") if source != "best_reward" else ""
        out.append(rec)
    return out

def metric_from_eval(eval_info: Dict[str, Any], name: str, mode: str) -> float:
    if name == "full_eval_loss":
        return get_metric_value(eval_info, "eval_loss", initial_best_value(mode))
    if name.startswith("full_eval_"):
        return get_metric_value(eval_info, name[len("full_eval_"):], initial_best_value(mode))
    return get_metric_value(eval_info, name, get_metric_value(eval_info, "eval_loss", initial_best_value(mode)))


def delta(new: Any, old: Any) -> Optional[float]:
    try:
        new_f, old_f = float(new), float(old)
        if not np.isfinite(new_f) or not np.isfinite(old_f):
            return None
        return new_f - old_f
    except Exception:
        return None


def relative_loss_improvement(before: Any, after: Any) -> Optional[float]:
    try:
        before_f, after_f = float(before), float(after)
        if not np.isfinite(before_f) or not np.isfinite(after_f) or abs(before_f) < 1e-12:
            return None
        return (before_f - after_f) / abs(before_f)
    except Exception:
        return None


def before_after_summary(baseline: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "before_loss": baseline.get("eval_loss"),
        "after_loss": after.get("eval_loss"),
        "delta_loss": delta(after.get("eval_loss"), baseline.get("eval_loss")),
        "relative_loss_improvement": relative_loss_improvement(baseline.get("eval_loss"), after.get("eval_loss")),
        "before_rollout_reward": baseline.get("rollout_reward"),
        "after_rollout_reward": after.get("rollout_reward"),
        "delta_rollout_reward": delta(after.get("rollout_reward"), baseline.get("rollout_reward")),
        "before_rollout_loss": baseline.get("rollout_loss"),
        "after_rollout_loss": after.get("rollout_loss"),
        "delta_rollout_loss": delta(after.get("rollout_loss"), baseline.get("rollout_loss")),
        "before_entropy": baseline.get("rollout_entropy"),
        "after_entropy": after.get("rollout_entropy"),
        "delta_entropy": delta(after.get("rollout_entropy"), baseline.get("rollout_entropy")),
        "before_anchor_loss": baseline.get("rollout_anchor_loss"),
        "after_anchor_loss": after.get("rollout_anchor_loss"),
        "delta_anchor_loss": delta(after.get("rollout_anchor_loss"), baseline.get("rollout_anchor_loss")),
    }


def eval_candidate(model, graph, noise, tokenizer, cand: Dict[str, Any], cfg: Dict[str, Any], device: torch.device, samples: List[Dict[str, Any]], limit, ema_decay: float, use_ema: bool) -> Dict[str, Any]:
    ckpt = Path(cand["checkpoint"])
    load_state_into_model(model, ckpt, device, ema_decay=ema_decay, use_ema=use_ema)
    info = evaluate_loss(model, graph, noise, tokenizer, samples, cfg, device, limit=limit)
    row = {
        "run_id": cand.get("run_id"),
        "run_dir": str(cand.get("run_dir", "")),
        "checkpoint": str(ckpt),
        "checkpoint_name": cand.get("checkpoint_name"),
        "candidate_kind": cand.get("candidate_kind"),
        "candidate_source": cand.get("candidate_source"),
        "metrics_csv": str(cand.get("metrics_csv", "")),
        "metrics_json": str(cand.get("metrics_json", "")),
        "run_info_json": str(cand.get("run_info_json", "")),
        "eval_loss": info.get("eval_loss"),
        "eval_count": info.get("eval_count"),
        "rollout_loss": info.get("rollout_loss"),
        "rollout_reward": info.get("rollout_reward"),
        "rollout_reward_std": info.get("rollout_reward_std"),
        "rollout_entropy": info.get("rollout_entropy"),
        "rollout_logprob": info.get("rollout_logprob"),
        "rollout_anchor_loss": info.get("rollout_anchor_loss"),
        "eval_metrics": info,
    }
    return row


def copy_root_outputs(
    out_root: Path,
    loss_best: Dict[str, Any],
    reward_best: Optional[Dict[str, Any]],
    baseline_eval: Dict[str, Any],
    cfg: Dict[str, Any],
    start_name: str,
    eval_split: str,
    eval_limit_label_value: str | int,
    best_metric_name: str,
    best_mode: str,
    reward_metric_name: str,
    reward_mode: str,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    # Clean old verbose files and old naming.
    for extra in ["best_selection_report.csv", "best_selection_report.json", "best_reward.pth"]:
        try:
            (out_root / extra).unlink()
        except FileNotFoundError:
            pass

    shutil.copy2(Path(loss_best["checkpoint"]), out_root / "best.pth")
    if Path(loss_best.get("metrics_csv", "")).exists():
        shutil.copy2(Path(loss_best["metrics_csv"]), out_root / "best_metrics.csv")
    else:
        # Keep filename convention even if no per-run metrics.csv was found.
        with open(out_root / "best_metrics.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(loss_best.keys()))
            writer.writeheader(); writer.writerow(loss_best)
    if Path(loss_best.get("run_info_json", "")).exists():
        shutil.copy2(Path(loss_best["run_info_json"]), out_root / "best_run_info.json")
    else:
        dump_json(out_root / "best_run_info.json", {"start": start_name, "run_dir": loss_best.get("run_dir")})

    loss_eval = dict(loss_best.get("eval_metrics", {}))
    loss_eval.update({
        "selected_kind": "loss_best",
        "root_checkpoint": str(out_root / "best.pth"),
        "checkpoint": loss_best.get("checkpoint"),
        "checkpoint_name": loss_best.get("checkpoint_name"),
        "run_id": loss_best.get("run_id"),
        "run_dir": loss_best.get("run_dir"),
        "best_metric_name": best_metric_name,
        "best_mode": best_mode,
        "best_metric_value": loss_best.get("metric_value"),
        "eval_split": eval_split,
        "eval_limit": eval_limit_label_value,
        "before_after": before_after_summary(baseline_eval, loss_best),
    })

    metrics_obj: Dict[str, Any] = {
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "start": start_name,
        "best_checkpoint": str(out_root / "best.pth"),
        "best_metric_name": best_metric_name,
        "best_mode": best_mode,
        "best_metric_value": loss_best.get("metric_value"),
        "eval_loss": loss_best.get("eval_loss"),
        "eval_count": loss_best.get("eval_count"),
        "eval_split": eval_split,
        "eval_limit": eval_limit_label_value,
        "loss_best": loss_eval,
        "loss_best_before_after": loss_eval["before_after"],
        "baseline_eval_metrics": baseline_eval,
        "config": cfg,
    }

    if reward_best is not None:
        shutil.copy2(Path(reward_best["checkpoint"]), out_root / "best_reward.pth")
        reward_eval = dict(reward_best.get("eval_metrics", {}))
        reward_payload = {
            "selected_kind": "reward_best",
            "root_checkpoint": str(out_root / "best_reward.pth"),
            "checkpoint": reward_best.get("checkpoint"),
            "checkpoint_name": reward_best.get("checkpoint_name"),
            "candidate_source": reward_best.get("candidate_source"),
            "run_id": reward_best.get("run_id"),
            "run_dir": reward_best.get("run_dir"),
            "best_metric_name": reward_metric_name,
            "best_mode": reward_mode,
            "best_metric_value": reward_best.get("reward_metric_value"),
            "eval_loss": reward_best.get("eval_loss"),
            "eval_count": reward_best.get("eval_count"),
            "rollout_reward": reward_best.get("rollout_reward"),
            "rollout_loss": reward_best.get("rollout_loss"),
            "rollout_entropy": reward_best.get("rollout_entropy"),
            "rollout_anchor_loss": reward_best.get("rollout_anchor_loss"),
            "eval_metrics": reward_eval,
            "before_after": before_after_summary(baseline_eval, reward_best),
        }
        metrics_obj["best_reward_checkpoint"] = str(out_root / "best_reward.pth")
        metrics_obj["reward_best"] = reward_payload
        loss_eval["reward_best"] = reward_payload

    dump_json(out_root / "best_eval.json", loss_eval)
    dump_json(out_root / "best_metrics.json", metrics_obj)
    append_jsonl(out_root / "improvement_log.jsonl", {
        "time": metrics_obj["time"],
        "selected_kind": "select_best_rebuild",
        "start": start_name,
        "loss_best_checkpoint": loss_best.get("checkpoint"),
        "loss_best_metric": loss_best.get("metric_value"),
        "reward_best_checkpoint": reward_best.get("checkpoint") if reward_best else "",
        "reward_best_metric": reward_best.get("reward_metric_value") if reward_best else "",
        "eval_split": eval_split,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild root best.pth and best_reward.pth from per-run RL checkpoints.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--start", type=str, default=None, help="Start key: pretrain, QA, QRA")
    parser.add_argument("--run-root", type=str, default="", help="Override root directory, e.g. rl_qra_pipeline/modelparameter/rl_QRA")
    parser.add_argument("--split", type=str, default="", help="Full-eval split. Default: data.full_eval_split or val.")
    parser.add_argument("--limit", type=str, default="", help="all/full/-1 or integer. Default: training.full_eval_limit/all.")
    parser.add_argument("--best-metric", type=str, default="", help="Metric for root best.pth. Default training.best_metric/full_eval_loss.")
    parser.add_argument("--best-mode", type=str, default="", choices=["min", "max"])
    parser.add_argument("--best-reward-metric", type=str, default="", help="Metric for root best_reward.pth. Default training.best_reward_metric/full_eval_rollout_reward.")
    parser.add_argument("--best-reward-mode", type=str, default="", choices=["min", "max"])
    parser.add_argument("--include-last-for-loss", action="store_true", help="Also allow last.pth to compete for conservative best.pth. Default false.")
    parser.add_argument("--no-mini", action="store_true", help="Do not include best_mini.pth for conservative best.pth.")
    parser.add_argument("--no-ema", action="store_true", help="Evaluate raw model weights instead of EMA weights in checkpoint.")
    parser.add_argument("--no-recover-best-reward", action="store_true", help="Do not backfill per-run best_reward.pth from last.pth.")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    raw_cfg = load_config(args.config)
    cfg, start_name = select_start_config(raw_cfg, args.start)
    cfg = maybe_disable_missing_pretrain_checkpoint(cfg, start_name)
    device = choose_device(cfg, cli_gpu=args.gpu, cli_cpu=args.cpu)

    out_root = resolve_repo_path(args.run_root) if args.run_root else resolve_repo_path(
        cfg.get("output", {}).get("dir", f"rl_qra_pipeline/modelparameter/rl_{start_name}")
    )
    eval_split = args.split or default_full_eval_split(cfg)
    limit_value = args.limit if args.limit != "" else cfg.get("training", {}).get("full_eval_limit", "all")
    eval_limit = parse_eval_limit(limit_value, default=int(cfg.get("data", {}).get("eval_limit", 64) or 64))
    eval_limit_for_logs = eval_limit_label(eval_limit)
    best_metric_name = args.best_metric or str(cfg.get("training", {}).get("best_metric", "full_eval_loss"))
    best_mode = args.best_mode or str(cfg.get("training", {}).get("best_mode", "min")).lower()
    reward_metric_name = args.best_reward_metric or str(cfg.get("training", {}).get("best_reward_metric", cfg.get("training", {}).get("reward_best_metric", "full_eval_rollout_reward")))
    reward_mode = args.best_reward_mode or str(cfg.get("training", {}).get("best_reward_mode", cfg.get("training", {}).get("reward_best_mode", "max"))).lower()

    tokenizer = GPT2TokenizerFast.from_pretrained(cfg["model"].get("tokenizer", "gpt2"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[select-best] start={start_name} device={device} root={out_root}", flush=True)
    print(f"[select-best] eval_split={eval_split} limit={eval_limit_for_logs} best={best_metric_name}/{best_mode} reward={reward_metric_name}/{reward_mode}", flush=True)
    model, graph, noise, ema, loaded_from = load_policy(cfg, device)
    eval_samples = load_eval_samples_for_split(cfg, tokenizer, eval_split)

    baseline_eval = evaluate_loss(model, graph, noise, tokenizer, eval_samples, cfg, device, limit=eval_limit)
    baseline_eval.update({"eval_split": eval_split, "eval_limit": eval_limit_for_logs, "loaded_from": loaded_from, "start": start_name})
    print(f"[baseline] eval_loss={baseline_eval.get('eval_loss')} R={baseline_eval.get('rollout_reward', 0.0):+.4f} count={baseline_eval.get('eval_count')}", flush=True)

    ema_decay = float(cfg.get("training", {}).get("ema", 0.9999))
    eval_cache: Dict[str, Dict[str, Any]] = {}

    def evaluate(cand: Dict[str, Any]) -> Dict[str, Any]:
        key = str(Path(cand["checkpoint"]).resolve())
        if key not in eval_cache:
            print(f"[eval] {cand['candidate_kind']} {cand['checkpoint']} source={cand.get('candidate_source')}", flush=True)
            try:
                row = eval_candidate(model, graph, noise, tokenizer, cand, cfg, device, eval_samples, eval_limit, ema_decay, use_ema=not args.no_ema)
            except Exception as exc:
                row = {"checkpoint": str(cand.get("checkpoint")), "candidate_kind": cand.get("candidate_kind"), "error": str(exc)}
                print(f"[warn] failed {cand.get('checkpoint')}: {exc}", flush=True)
            eval_cache[key] = row
        return eval_cache[key]

    loss_candidates = find_loss_candidates(out_root, include_last=args.include_last_for_loss, include_mini=not args.no_mini)
    if not loss_candidates:
        raise RuntimeError(f"No loss candidates found under {out_root}; expected <run_id>/best.pth.")
    reward_candidates = find_reward_candidates(out_root, recover_missing=not args.no_recover_best_reward)
    if not reward_candidates:
        print(f"[warn] No reward candidates found under {out_root}; expected <run_id>/best_reward.pth or last.pth.", flush=True)

    loss_best: Optional[Dict[str, Any]] = None
    loss_best_value = initial_best_value(best_mode)
    for cand in loss_candidates:
        row = evaluate(cand)
        if row.get("error"):
            continue
        metric_value = metric_from_eval(row, best_metric_name, best_mode)
        row["metric_value"] = metric_value
        if metric_better(metric_value, loss_best_value, best_mode):
            loss_best_value = metric_value
            loss_best = row
        print(f"    loss-candidate {Path(row['checkpoint']).name}: {best_metric_name}={metric_value:.6g} loss={float(row.get('eval_loss', float('inf'))):.6g} R={float(row.get('rollout_reward', 0.0)):+.4f}", flush=True)

    reward_best: Optional[Dict[str, Any]] = None
    reward_best_value = initial_best_value(reward_mode)
    for cand in reward_candidates:
        row = evaluate(cand)
        if row.get("error"):
            continue
        metric_value = metric_from_eval(row, reward_metric_name, reward_mode)
        row["reward_metric_value"] = metric_value
        # Backfill/update the per-run reward-best report too.  This is under the
        # run directory, not the root output directory.
        try:
            run_dir = Path(str(row.get("run_dir", "")))
            if run_dir.exists():
                dump_json(run_dir / "best_reward_eval.json", {
                    "selected_kind": "per_run_reward_candidate",
                    "checkpoint": row.get("checkpoint"),
                    "checkpoint_name": row.get("checkpoint_name"),
                    "candidate_source": row.get("candidate_source"),
                    "recovered_best_reward": bool(cand.get("recovered_best_reward", False)),
                    "best_metric_name": reward_metric_name,
                    "best_mode": reward_mode,
                    "best_metric_value": metric_value,
                    "eval_split": eval_split,
                    "eval_limit": eval_limit_for_logs,
                    "eval_loss": row.get("eval_loss"),
                    "eval_count": row.get("eval_count"),
                    "rollout_reward": row.get("rollout_reward"),
                    "rollout_loss": row.get("rollout_loss"),
                    "rollout_entropy": row.get("rollout_entropy"),
                    "rollout_anchor_loss": row.get("rollout_anchor_loss"),
                    "eval_metrics": row.get("eval_metrics", {}),
                    "before_after": before_after_summary(baseline_eval, row),
                })
        except Exception as exc:
            print(f"[warn] failed to write per-run best_reward_eval.json for {row.get('run_dir')}: {exc}", flush=True)
        if metric_better(metric_value, reward_best_value, reward_mode):
            reward_best_value = metric_value
            reward_best = row
        print(f"    reward-candidate {Path(row['checkpoint']).name}: {reward_metric_name}={metric_value:.6g} loss={float(row.get('eval_loss', float('inf'))):.6g} R={float(row.get('rollout_reward', 0.0)):+.4f} source={row.get('candidate_source')}", flush=True)

    if loss_best is None:
        raise RuntimeError("No loss candidate evaluated successfully.")

    copy_root_outputs(
        out_root=out_root,
        loss_best=loss_best,
        reward_best=reward_best,
        baseline_eval=baseline_eval,
        cfg=cfg,
        start_name=start_name,
        eval_split=eval_split,
        eval_limit_label_value=eval_limit_for_logs,
        best_metric_name=best_metric_name,
        best_mode=best_mode,
        reward_metric_name=reward_metric_name,
        reward_mode=reward_mode,
    )

    print("\nSelected conservative best.pth:", flush=True)
    print(f"  {loss_best['checkpoint']}", flush=True)
    print(f"  {best_metric_name}={loss_best.get('metric_value')} eval_loss={loss_best.get('eval_loss')} R={loss_best.get('rollout_reward')}", flush=True)
    if reward_best is not None:
        print("Selected reward best_reward.pth:", flush=True)
        print(f"  {reward_best['checkpoint']}", flush=True)
        print(f"  {reward_metric_name}={reward_best.get('reward_metric_value')} eval_loss={reward_best.get('eval_loss')} R={reward_best.get('rollout_reward')} source={reward_best.get('candidate_source')}", flush=True)
    print(f"Wrote root best files under: {out_root}", flush=True)


if __name__ == "__main__":
    main()
