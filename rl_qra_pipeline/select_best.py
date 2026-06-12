from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
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
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

# Reuse the exact training/eval implementation so this script evaluates candidates
# with the same rollout objective used during RL training.
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


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def maybe_disable_missing_pretrain_checkpoint(cfg: Dict[str, Any], start_name: str) -> Dict[str, Any]:
    """Allow --start pretrain to use SEDD.from_pretrained/cache without a local .pth.

    Older configs may still contain starts.pretrain.init_checkpoint pointing to a
    nonexistent pretrain.pth.  That should not block baseline/pretrain RL eval.
    """
    ckpt_value = str(cfg.get("model", {}).get("init_checkpoint", "") or "").strip()
    if not ckpt_value:
        return cfg
    if str(start_name).lower() == "pretrain":
        ckpt_path = resolve_repo_path(ckpt_value)
        if not ckpt_path.exists() or ckpt_value.lower() in {"pretrained", "hf", "cache", "none", "null"}:
            cfg.setdefault("model", {})["init_checkpoint"] = ""
            print(f"[info] --start pretrain uses pretrained/cache weights; ignored missing init_checkpoint={ckpt_value!r}", flush=True)
    return cfg


def default_full_eval_split(cfg: Dict[str, Any]) -> str:
    data_cfg = cfg.get("data", {})
    if data_cfg.get("full_eval_split"):
        return str(data_cfg.get("full_eval_split"))
    eval_split = str(data_cfg.get("eval_split", "val") or "val")
    # If training uses val-mini for mini eval, the full selection pass should use val.
    if eval_split.lower() in {"val-mini", "val_mini", "validation-mini", "validation_mini"}:
        return "val"
    return eval_split


def load_eval_samples_for_split(cfg: Dict[str, Any], tokenizer, split: str):
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


def find_candidate_checkpoints(run_root: Path, include_last: bool = False, include_mini: bool = True) -> List[Dict[str, Any]]:
    """Find per-run checkpoints under the new flat layout and old runs/ layout.

    New layout:
      rl_QRA/<run_id>/best_run.pth
    Old layout:
      rl_QRA/runs/<run_id>/best_run.pth
    """
    names = ["best_run.pth"]
    if include_mini:
        names.append("best_mini_run.pth")
    if include_last:
        names.append("last_run.pth")

    candidates: List[Dict[str, Any]] = []
    seen: set[Path] = set()
    search_roots = [run_root]
    old_runs_root = run_root / "runs"
    if old_runs_root.exists():
        search_roots.append(old_runs_root)

    for root in search_roots:
        if not root.exists():
            continue
        for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            # Skip helper dirs that are not training runs.
            if run_dir.name.startswith("."):
                continue
            for name in names:
                ckpt = run_dir / name
                if not ckpt.exists():
                    continue
                rp = ckpt.resolve()
                if rp in seen:
                    continue
                seen.add(rp)
                candidates.append({
                    "checkpoint": ckpt,
                    "checkpoint_name": name,
                    "run_dir": run_dir,
                    "run_id": run_dir.name,
                    "metrics_json": run_dir / "metrics.json",
                    "eval_json": run_dir / "eval.json",
                    "best_eval_json": run_dir / "best_eval.json",
                    "metrics_csv": run_dir / "metrics.csv",
                    "run_info_json": run_dir / "run_info.json",
                })
    return candidates


def metric_from_eval(eval_info: Dict[str, Any], name: str, mode: str) -> float:
    # Script evaluates full split and stores standard eval_loss.  If the config's
    # best_metric is full_eval_loss, map it to eval_loss for this selection pass.
    if name == "full_eval_loss":
        return get_metric_value(eval_info, "eval_loss", initial_best_value(mode))
    if name.startswith("full_eval_"):
        return get_metric_value(eval_info, name[len("full_eval_"):], initial_best_value(mode))
    return get_metric_value(eval_info, name, get_metric_value(eval_info, "eval_loss", initial_best_value(mode)))


def delta(new: Any, old: Any) -> Optional[float]:
    try:
        new_f = float(new)
        old_f = float(old)
        if not np.isfinite(new_f) or not np.isfinite(old_f):
            return None
        return new_f - old_f
    except Exception:
        return None


def relative_loss_improvement(before: Any, after: Any) -> Optional[float]:
    try:
        before_f = float(before)
        after_f = float(after)
        if not np.isfinite(before_f) or not np.isfinite(after_f) or abs(before_f) < 1e-12:
            return None
        # Positive means loss decreased.
        return (before_f - after_f) / abs(before_f)
    except Exception:
        return None


def add_before_after_fields(row: Dict[str, Any], baseline: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Attach before/after metrics showing RL effect against the start checkpoint."""
    keys = [
        ("eval_loss", "loss"),
        ("rollout_loss", "rollout_loss"),
        ("rollout_reward", "rollout_reward"),
        ("rollout_reward_std", "rollout_reward_std"),
        ("rollout_entropy", "rollout_entropy"),
        ("rollout_logprob", "rollout_logprob"),
        ("rollout_anchor_loss", "rollout_anchor_loss"),
    ]
    for key, short in keys:
        before = baseline.get(key)
        after = row.get(key)
        row[f"{prefix}before_{short}"] = before
        row[f"{prefix}after_{short}"] = after
        row[f"{prefix}delta_{short}"] = delta(after, before)
    row[f"{prefix}relative_loss_improvement"] = relative_loss_improvement(baseline.get("eval_loss"), row.get("eval_loss"))
    return row


def copy_best_outputs(
    out_root: Path,
    best: Dict[str, Any],
    baseline_eval: Dict[str, Any],
    all_rows: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    start_name: str,
    eval_split: str,
    eval_limit_label_value: str | int,
    best_metric_name: str,
    best_mode: str,
) -> None:
    """Write the same compact root layout as the other RL/SFT pipelines.

    Root directory output is intentionally limited to best.pth plus five files:
      - best_metrics.csv
      - best_metrics.json
      - best_eval.json
      - best_run_info.json
      - improvement_log.jsonl

    Per-run details remain inside each run directory.  Candidate-level selection
    details are summarized inside best_eval.json / best_metrics.json instead of
    creating extra root report files.
    """
    out_root.mkdir(parents=True, exist_ok=True)

    # Remove verbose report files produced by older versions of this helper.
    # Keep root aligned with the compact layout used by the other pipelines.
    for extra_name in ["best_selection_report.csv", "best_selection_report.json"]:
        try:
            (out_root / extra_name).unlink()
        except FileNotFoundError:
            pass

    ckpt = Path(best["checkpoint"])
    shutil.copy2(ckpt, out_root / "best.pth")

    # Keep the conventional root metrics.csv copy from the selected run.  This is
    # useful for visual_rl_qra.py and matches the existing best_metrics.csv pattern.
    if Path(best.get("metrics_csv", "")).exists():
        shutil.copy2(Path(best["metrics_csv"]), out_root / "best_metrics.csv")
    else:
        write_csv(out_root / "best_metrics.csv", [best])

    if Path(best.get("run_info_json", "")).exists():
        shutil.copy2(Path(best["run_info_json"]), out_root / "best_run_info.json")
    else:
        dump_json(out_root / "best_run_info.json", {"run_dir": best.get("run_dir"), "run_id": best.get("run_id")})

    before_after = {
        "before_loss": baseline_eval.get("eval_loss"),
        "after_loss": best.get("eval_loss"),
        "delta_loss": delta(best.get("eval_loss"), baseline_eval.get("eval_loss")),
        "relative_loss_improvement": relative_loss_improvement(baseline_eval.get("eval_loss"), best.get("eval_loss")),
        "before_rollout_reward": baseline_eval.get("rollout_reward"),
        "after_rollout_reward": best.get("rollout_reward"),
        "delta_rollout_reward": delta(best.get("rollout_reward"), baseline_eval.get("rollout_reward")),
        "before_entropy": baseline_eval.get("rollout_entropy"),
        "after_entropy": best.get("rollout_entropy"),
        "delta_entropy": delta(best.get("rollout_entropy"), baseline_eval.get("rollout_entropy")),
        "before_anchor_loss": baseline_eval.get("rollout_anchor_loss"),
        "after_anchor_loss": best.get("rollout_anchor_loss"),
        "delta_anchor_loss": delta(best.get("rollout_anchor_loss"), baseline_eval.get("rollout_anchor_loss")),
    }
    top_candidates = []
    for row in all_rows[: min(10, len(all_rows))]:
        top_candidates.append({
            "rank_metric": row.get("rank_metric"),
            "metric_value": row.get("metric_value"),
            "eval_loss": row.get("eval_loss"),
            "rollout_reward": row.get("rollout_reward"),
            "run_id": row.get("run_id"),
            "checkpoint_name": row.get("checkpoint_name"),
            "checkpoint": row.get("checkpoint"),
            "delta_loss": row.get("delta_loss"),
            "relative_loss_improvement": row.get("relative_loss_improvement"),
            "error": row.get("error", ""),
        })

    best_eval = {
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "selected_by_script": True,
        "algorithm": cfg.get("rl", {}).get("mode", ""),
        "start": start_name,
        "eval_split": eval_split,
        "eval_limit": eval_limit_label_value,
        "best_metric_name": best_metric_name,
        "best_mode": best_mode,
        "best_metric_value": best.get("metric_value"),
        "checkpoint": str(ckpt),
        "checkpoint_name": best.get("checkpoint_name"),
        "run_dir": str(best.get("run_dir", "")),
        "run_id": best.get("run_id"),
        "baseline_eval_metrics": baseline_eval,
        "selected_eval_metrics": {k: best.get(k) for k in best.keys() if k.startswith("eval_") or k.startswith("rollout_")},
        "before_after": before_after,
        "num_candidates": len(all_rows),
        "top_candidates": top_candidates,
    }
    dump_json(out_root / "best_eval.json", best_eval)

    dump_json(out_root / "best_metrics.json", {
        "time": best_eval["time"],
        "selected_by_script": True,
        "start": start_name,
        "run_id": best.get("run_id"),
        "run_dir": str(best.get("run_dir", "")),
        "checkpoint": str(ckpt),
        "checkpoint_name": best.get("checkpoint_name"),
        "best_metric_name": best_metric_name,
        "best_mode": best_mode,
        "best_metric_value": best.get("metric_value"),
        "eval_loss": best.get("eval_loss"),
        "eval_count": best.get("eval_count"),
        "eval_split": eval_split,
        "eval_limit": eval_limit_label_value,
        "baseline_eval_loss": baseline_eval.get("eval_loss"),
        "before_after": before_after,
        "num_candidates": len(all_rows),
        "top_candidates": top_candidates,
    })

    append_jsonl(out_root / "improvement_log.jsonl", {
        "time": best_eval["time"],
        "selected_by_script": True,
        "run_dir": str(best.get("run_dir", "")),
        "checkpoint": str(ckpt),
        "best_metric_name": best_metric_name,
        "best_mode": best_mode,
        "new_best_metric": best.get("metric_value"),
        "new_eval_loss": best.get("eval_loss"),
        "baseline_eval_loss": baseline_eval.get("eval_loss"),
        "delta_loss": before_after["delta_loss"],
        "relative_loss_improvement": before_after["relative_loss_improvement"],
    })

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all RL run checkpoints and rebuild root best.pth / reports.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--start", type=str, default=None, help="Start key under config.starts, e.g. pretrain, QA, QRA.")
    parser.add_argument("--run-root", type=str, default="", help="Override output root, e.g. rl_qra_pipeline/modelparameter/rl_QRA.")
    parser.add_argument("--split", type=str, default="", help="Eval split. Default: data.full_eval_split or val.")
    parser.add_argument("--limit", type=str, default="", help="Eval limit: all, full, -1, or an integer. Default: training.full_eval_limit/all.")
    parser.add_argument("--best-metric", type=str, default="", help="Metric used to select best. Default: training.best_metric/full_eval_loss.")
    parser.add_argument("--best-mode", type=str, default="", choices=["min", "max"], help="min for loss, max for reward.")
    parser.add_argument("--include-last", action="store_true", help="Also evaluate last_run.pth from each run.")
    parser.add_argument("--no-mini", action="store_true", help="Do not evaluate best_mini_run.pth candidates.")
    parser.add_argument("--no-ema", action="store_true", help="Evaluate raw model weights instead of EMA weights saved in checkpoints.")
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

    tokenizer = GPT2TokenizerFast.from_pretrained(cfg["model"].get("tokenizer", "gpt2"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[select-best] start={start_name} device={device} run_root={out_root}", flush=True)
    print(f"[select-best] eval_split={eval_split} eval_limit={eval_limit_for_logs} metric={best_metric_name} mode={best_mode}", flush=True)

    model, graph, noise, ema, loaded_from = load_policy(cfg, device)
    eval_samples = load_eval_samples_for_split(cfg, tokenizer, eval_split)

    # Baseline before-RL eval: the selected start checkpoint/cache.
    baseline_eval = evaluate_loss(model, graph, noise, tokenizer, eval_samples, cfg, device, limit=eval_limit)
    baseline_eval.update({
        "eval_split": eval_split,
        "eval_limit": eval_limit_for_logs,
        "loaded_from": loaded_from,
        "start": start_name,
    })
    print(
        f"[baseline] eval_loss={baseline_eval.get('eval_loss')} "
        f"R={baseline_eval.get('rollout_reward', 0.0):+.4f} count={baseline_eval.get('eval_count')}",
        flush=True,
    )

    candidates = find_candidate_checkpoints(out_root, include_last=args.include_last, include_mini=not args.no_mini)
    if not candidates:
        raise RuntimeError(f"No candidate checkpoints found under {out_root}. Expected <run_id>/best_run.pth.")

    rows: List[Dict[str, Any]] = []
    best_row: Optional[Dict[str, Any]] = None
    best_value = initial_best_value(best_mode)
    ema_decay = float(cfg.get("training", {}).get("ema", 0.9999))

    for i, cand in enumerate(candidates, start=1):
        ckpt = Path(cand["checkpoint"])
        print(f"[{i}/{len(candidates)}] eval {ckpt}", flush=True)
        try:
            load_state_into_model(model, ckpt, device, ema_decay=ema_decay, use_ema=not args.no_ema)
            eval_info = evaluate_loss(model, graph, noise, tokenizer, eval_samples, cfg, device, limit=eval_limit)
        except Exception as exc:
            row = {
                "run_id": cand.get("run_id"),
                "run_dir": str(cand.get("run_dir", "")),
                "checkpoint": str(ckpt),
                "checkpoint_name": cand.get("checkpoint_name"),
                "error": str(exc),
            }
            rows.append(row)
            print(f"[warn] failed: {exc}", flush=True)
            continue

        metric_value = metric_from_eval(eval_info, best_metric_name, best_mode)
        row: Dict[str, Any] = {
            "rank_metric": metric_value,
            "metric_value": metric_value,
            "best_metric_name": best_metric_name,
            "best_mode": best_mode,
            "run_id": cand.get("run_id"),
            "run_dir": str(cand.get("run_dir", "")),
            "checkpoint": str(ckpt),
            "checkpoint_name": cand.get("checkpoint_name"),
            "eval_split": eval_split,
            "eval_limit": eval_limit_for_logs,
            "eval_loss": eval_info.get("eval_loss"),
            "eval_count": eval_info.get("eval_count"),
            "rollout_loss": eval_info.get("rollout_loss"),
            "rollout_reward": eval_info.get("rollout_reward"),
            "rollout_reward_std": eval_info.get("rollout_reward_std"),
            "rollout_entropy": eval_info.get("rollout_entropy"),
            "rollout_logprob": eval_info.get("rollout_logprob"),
            "rollout_anchor_loss": eval_info.get("rollout_anchor_loss"),
            "metrics_json": str(cand.get("metrics_json", "")),
            "metrics_csv": str(cand.get("metrics_csv", "")),
            "run_info_json": str(cand.get("run_info_json", "")),
        }
        add_before_after_fields(row, baseline_eval)
        rows.append(row)

        improved = metric_better(metric_value, best_value, best_mode)
        print(
            f"    {best_metric_name}={metric_value:.6g} eval_loss={float(eval_info.get('eval_loss', float('inf'))):.6g} "
            f"R={float(eval_info.get('rollout_reward', 0.0)):+.4f} improved={'yes' if improved else 'no'}",
            flush=True,
        )
        if improved:
            best_value = metric_value
            best_row = row

    # Sort for readable report.
    reverse = best_mode == "max"
    rows_sorted = sorted(rows, key=lambda r: float(r.get("rank_metric", initial_best_value("max" if reverse else "min"))) if r.get("rank_metric", "") != "" else (float("-inf") if reverse else float("inf")), reverse=reverse)

    if best_row is None:
        for extra_name in ["best_selection_report.csv", "best_selection_report.json"]:
            try:
                (out_root / extra_name).unlink()
            except FileNotFoundError:
                pass
        raise RuntimeError("No candidate evaluated successfully. No root report was written to keep the compact root layout.")

    copy_best_outputs(
        out_root=out_root,
        best=best_row,
        baseline_eval=baseline_eval,
        all_rows=rows_sorted,
        cfg=cfg,
        start_name=start_name,
        eval_split=eval_split,
        eval_limit_label_value=eval_limit_for_logs,
        best_metric_name=best_metric_name,
        best_mode=best_mode,
    )
    print("\nSelected best checkpoint:", flush=True)
    print(f"  {best_row['checkpoint']}", flush=True)
    print(f"  {best_metric_name}={best_row.get('metric_value')}", flush=True)
    print(f"  baseline_loss={baseline_eval.get('eval_loss')} after_loss={best_row.get('eval_loss')} delta={best_row.get('delta_loss')}", flush=True)
    print(f"Wrote root best files under: {out_root}", flush=True)


if __name__ == "__main__":
    main()
