from __future__ import annotations

"""Generate answer recovery chains for all eval models in rl_qra_config.yaml.

This is a diagnostic experiment script. It does NOT train and does NOT score.
It only records how the answer segment changes during an answer-mask reverse chain.

Default output:
    experiment/sample_chain/<timestamp>_<run_name>/
        chain.csv       # aligned by dataset/sample/step, one column pair per model
        detail.csv      # long-form per dataset/sample/model/step
        chain.md        # compact vertical view
        summary.json

By default the script reads config.eval.compare_models, or top-level
compare_models, and falls back to {<start>_base, rl_<start>}. It also reads
config.eval.datasets and config.eval.split, matching the eval script behavior.
"""

import argparse
import csv
import datetime as dt
import gc
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import yaml
from transformers import GPT2TokenizerFast

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
for p in (REPO_DIR, SCRIPT_DIR, REPO_DIR / "sft_answer_pipeline", REPO_DIR / "sft_rl_pipeline"):
    p_str = str(p)
    if p.exists() and p_str not in sys.path:
        sys.path.insert(0, p_str)

import graph_lib  # noqa: E402
import noise_lib  # noqa: E402
from model import SEDD  # noqa: E402
from model.ema import ExponentialMovingAverage  # noqa: E402
from answer_dataset import AnswerSegmentDataset, ordered_segments  # noqa: E402
from state_builder import encode_sample, mask_id_from_graph, project_fixed_, transition_probs  # noqa: E402

DEFAULT_CONFIG = SCRIPT_DIR / "rl_qra_config.yaml"


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_DIR / p


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(args: argparse.Namespace, cfg: Dict[str, Any]) -> torch.device:
    if args.cpu or bool(cfg.get("cpu", False)) or not torch.cuda.is_available():
        return torch.device("cpu")
    if args.gpu is not None:
        return torch.device(f"cuda:{int(args.gpu)}")
    cfg_gpu = (cfg.get("run") or {}).get("cuda_device", None)
    if cfg_gpu is None or str(cfg_gpu).lower() in {"", "none", "null", "auto", "cuda"}:
        return torch.device("cuda")
    return torch.device(f"cuda:{int(cfg_gpu)}")


def selected_start_name(cfg: Dict[str, Any], cli_start: Optional[str]) -> str:
    return str(cli_start or (cfg.get("run") or {}).get("selected") or cfg.get("selected") or "QRA")


def selected_start_cfg(cfg: Dict[str, Any], start_name: str) -> Dict[str, Any]:
    return ((cfg.get("starts") or {}).get(start_name) or {})


def resolve_base_data_dir(cfg: Dict[str, Any], start_name: str) -> Path:
    data_cfg = cfg.get("data") or {}
    start_cfg = selected_start_cfg(cfg, start_name)
    data_dir = data_cfg.get("data_dir") or start_cfg.get("data_dir") or "rl_qra_pipeline/data/S1K_RL"
    return repo_path(data_dir)


def resolve_dataset_path(cfg: Dict[str, Any], start_name: str, dataset_name: str, split: str) -> Path:
    base = resolve_base_data_dir(cfg, start_name)
    if dataset_name in {".", "current", base.name}:
        return base / f"{split}.jsonl"
    return base.parent / dataset_name / f"{split}.jsonl"


def default_compare_models(cfg: Dict[str, Any], start_name: str) -> Dict[str, Dict[str, str]]:
    start_cfg = selected_start_cfg(cfg, start_name)
    output_dir = start_cfg.get("output_dir", f"rl_qra_pipeline/modelparameter/rl_{start_name}")
    compare: Dict[str, Dict[str, str]] = {}
    init_ckpt = start_cfg.get("init_checkpoint") or f"rl_qra_pipeline/modelparameter/{start_name}/best.pth"
    compare[f"{start_name}_base"] = {"checkpoint": str(init_ckpt)}
    compare[f"rl_{start_name}"] = {"checkpoint": str(Path(output_dir) / "best.pth")}
    return compare


def resolve_compare_models(args: argparse.Namespace, cfg: Dict[str, Any], start_name: str) -> Dict[str, Dict[str, str]]:
    # Match eval config first: eval.compare_models is the intended location.
    eval_cfg = cfg.get("eval") or {}
    compare = eval_cfg.get("compare_models") or cfg.get("compare_models") or default_compare_models(cfg, start_name)
    if not isinstance(compare, dict) or not compare:
        compare = default_compare_models(cfg, start_name)

    out: Dict[str, Dict[str, str]] = {}
    for name, spec in compare.items():
        if args.models and str(name) not in set(args.models):
            continue
        if isinstance(spec, dict):
            ckpt = spec.get("checkpoint") or spec.get("path") or spec.get("ckpt")
        else:
            ckpt = str(spec)
        if not ckpt:
            continue
        out[str(name)] = {"checkpoint": str(ckpt)}
    if not out:
        raise RuntimeError("No compare models selected from config.eval.compare_models / compare_models.")
    return out


def resolve_eval_datasets(args: argparse.Namespace, cfg: Dict[str, Any], start_name: str) -> List[str]:
    if args.datasets:
        return list(args.datasets)
    eval_cfg = cfg.get("eval") or {}
    base = resolve_base_data_dir(cfg, start_name)
    return list(eval_cfg.get("datasets") or [base.name])


def normalize_text(x: str) -> str:
    x = str(x or "")
    x = x.replace("\\mathrm{~m}", " m").replace("\\mathrm{m}", "m")
    x = x.replace("\\left", "").replace("\\right", "")
    x = re.sub(r"\s+", " ", x).strip()
    if len(x) > 1 and x[-1] in ".;":
        x = x[:-1].strip()
    return x


def get_segment_text(sample: Dict[str, Any], names: Sequence[str]) -> str:
    segs = sample.get("segments") or {}
    if isinstance(segs, dict):
        for name in names:
            if name in segs:
                seg = segs[name]
                if isinstance(seg, dict):
                    txt = str(seg.get("text", ""))
                else:
                    txt = str(seg)
                if txt.strip():
                    return txt.strip()
    return ""


def extract_gt_answer(sample: Dict[str, Any]) -> str:
    for key in ("answer", "solution", "final_answer", "target"):
        val = sample.get(key)
        if val is not None and str(val).strip():
            return normalize_text(str(val))
    text = get_segment_text(sample, ["answer", "final_answer", "target"])
    if text:
        return normalize_text(text)
    train_parts = []
    for _, seg in ordered_segments(sample):
        if isinstance(seg, dict) and bool(seg.get("train", False)):
            train_parts.append(str(seg.get("text", "")))
    return normalize_text("".join(train_parts)) if train_parts else ""


def answer_kind(answer: str) -> str:
    a = normalize_text(answer)
    c = a.replace(" ", "")
    if re.fullmatch(r"[A-Ea-e]", c):
        return "letter"
    if re.fullmatch(r"[+-]?\d+", c):
        return "integer"
    if re.fullmatch(r"[+-]?(?:\d+\.\d+|\.\d+)", c):
        return "decimal"
    if re.fullmatch(r"[\(\[][+-]?(?:\d+(?:\.\d+)?|\.\d+),[+-]?(?:\d+(?:\.\d+)?|\.\d+)[\)\]]", c):
        return "interval"
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*=.+", c):
        return "equation"
    if re.fullmatch(r"[+-]?(?:\d+\.\d+|\.\d+|\d+)(m|mm|cm|km|kg|g|s|ms|N|J|W|V|A|Hz|m/s|m/s\^2)", c):
        return "unit_decimal"
    if any(sym in c for sym in ["^", "/", "sqrt", "pi", "\\sqrt", "\\frac", "*", "+", "-"]) and re.search(r"[A-Za-z0-9]", c):
        return "symbolic"
    return "short_text"


def load_samples(cfg: Dict[str, Any], start_name: str, dataset_name: str, split: str, tokenizer) -> List[Dict[str, Any]]:
    path = resolve_dataset_path(cfg, start_name, dataset_name, split)
    if not path.exists():
        raise FileNotFoundError(f"Data split not found: {path}")
    model_cfg = cfg.get("model") or {}
    ds = AnswerSegmentDataset(
        path,
        tokenizer,
        int(model_cfg.get("max_length", 1024)),
        min_target_tokens=int(model_cfg.get("min_target_tokens", 1)),
        drop_overlength=bool(model_cfg.get("drop_overlength", True)),
        write_report=False,
    )
    return ds.samples


def pick_diverse_samples(samples: List[Dict[str, Any]], sample_ids: Sequence[str], num_samples: int, per_type: int) -> List[Dict[str, Any]]:
    if sample_ids:
        wanted = set(sample_ids)
        chosen = [s for s in samples if str(s.get("id", "")) in wanted]
        missing = wanted - {str(s.get("id", "")) for s in chosen}
        if missing:
            print(f"[warn] missing sample ids: {sorted(missing)}", flush=True)
        return chosen[:num_samples]

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        buckets[answer_kind(extract_gt_answer(s))].append(s)

    priority = ["interval", "equation", "unit_decimal", "symbolic", "decimal", "integer", "letter", "short_text"]
    chosen: List[Dict[str, Any]] = []
    seen = set()
    for k in priority:
        for s in buckets.get(k, [])[:per_type]:
            sid = str(s.get("id", id(s)))
            if sid not in seen:
                chosen.append(s)
                seen.add(sid)
            if len(chosen) >= num_samples:
                return chosen
    for s in samples:
        sid = str(s.get("id", id(s)))
        if sid not in seen:
            chosen.append(s)
            seen.add(sid)
        if len(chosen) >= num_samples:
            break
    return chosen


def load_model_for_trace(cfg: Dict[str, Any], checkpoint: Path, device: torch.device):
    model_cfg = cfg.get("model") or {}
    training_cfg = cfg.get("training") or {}
    pretrained = model_cfg.get("pretrained", "louaaron/sedd-medium")
    model = SEDD.from_pretrained(pretrained).to(device)
    model.config.model.length = int(model_cfg.get("max_length", getattr(model.config.model, "length", 1024)))

    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model_state = state["model"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        model_state = state["model_state_dict"]
    else:
        model_state = state
    model.load_state_dict(model_state, strict=True)

    if isinstance(state, dict) and "ema" in state:
        try:
            ema = ExponentialMovingAverage(model.parameters(), decay=float(training_cfg.get("ema", 0.9999)))
            ema.load_state_dict(state["ema"])
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
        except Exception as exc:
            print(f"[warn] failed to apply EMA from {checkpoint}: {exc}", flush=True)

    graph = graph_lib.get_graph(model.config, device)
    noise = noise_lib.get_noise(model.config).to(device)
    model.eval()
    return model, graph, noise


def visible_answer_tokens(tokenizer, ids: Sequence[int], mask_id: int, vocab_size: int) -> Tuple[str, List[str]]:
    pieces: List[str] = []
    token_texts: List[str] = []
    for tok in ids:
        ti = int(tok)
        if ti == int(mask_id) or ti < 0 or ti >= vocab_size:
            pieces.append("[MASK]")
            token_texts.append("[MASK]")
        else:
            txt = tokenizer.decode([ti])
            pieces.append(txt)
            token_texts.append(txt)
    return "".join(pieces), token_texts


def trace_one_sample(model, graph, noise, tokenizer, sample: Dict[str, Any], cfg: Dict[str, Any], device: torch.device, args: argparse.Namespace, model_name: str, dataset_name: str) -> Dict[str, Any]:
    model_cfg = cfg.get("model") or {}
    max_length = int(model_cfg.get("max_length", 1024))
    encoded = encode_sample(sample, tokenizer, max_length, tokenizer.eos_token_id)
    answer_pos = [p for p in encoded.layout.answer_positions if p < encoded.layout.real_len]
    if not answer_pos:
        raise ValueError(f"sample {sample.get('id', '')} has no answer positions")

    mask_id = int(mask_id_from_graph(graph))
    vocab_size = int(getattr(tokenizer, "vocab_size", 50257))
    gt_answer = extract_gt_answer(sample)

    x = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    x[:, answer_pos] = mask_id
    project_fixed_(x, encoded.layout.fixed_locs, encoded.layout.fixed_ids.to(device))

    num_steps = int(args.steps)
    t_start = float(args.t_start)
    t_end = float(args.t_end)
    dt_step = (t_start - t_end) / max(1, num_steps)

    rows: List[Dict[str, Any]] = []
    prev_token_texts: Optional[List[str]] = None
    prev_ids = [int(x[0, p].item()) for p in answer_pos]

    with torch.no_grad():
        for step in range(num_steps + 1):
            t_val = max(t_end, t_start - step * dt_step)
            curr_ids = [int(x[0, p].item()) for p in answer_pos]
            ans_text, token_texts = visible_answer_tokens(tokenizer, curr_ids, mask_id, vocab_size)
            changed = [i for i, (a, b) in enumerate(zip(prev_ids, curr_ids)) if a != b]
            rows.append({
                "dataset": dataset_name,
                "model": model_name,
                "sample_id": str(sample.get("id", "")),
                "answer_kind": answer_kind(gt_answer),
                "gt_answer": gt_answer,
                "step": int(step),
                "t": float(t_val),
                "answer_text": ans_text,
                "answer_token_texts": token_texts,
                "answer_token_ids": curr_ids,
                "mask_count": sum(1 for z in curr_ids if int(z) == mask_id or int(z) < 0),
                "changed_answer_indices": changed,
            })

            if step == num_steps:
                break

            t_tensor = torch.tensor([t_val], dtype=torch.float32, device=device)
            probs = transition_probs(
                model,
                graph,
                noise,
                x,
                t_tensor,
                dt_step,
                kind=str(args.transition_kind),
                train=False,
                fixed_locs=encoded.layout.fixed_locs,
                fixed_ids=encoded.layout.fixed_ids.to(device),
            )

            prev_ids = curr_ids
            prev_token_texts = token_texts

            if args.mode == "greedy":
                next_x = probs.argmax(dim=-1)
            else:
                flat = probs.view(-1, probs.shape[-1])
                next_x = torch.multinomial(flat.float().clamp_min(0.0), num_samples=1).view_as(x)

            if args.freeze_filled:
                keep = torch.zeros_like(x, dtype=torch.bool)
                for p in answer_pos:
                    keep[:, p] = x[:, p] != mask_id
                next_x = torch.where(keep, x, next_x)

            x = next_x.to(device)
            project_fixed_(x, encoded.layout.fixed_locs, encoded.layout.fixed_ids.to(device))

    return {
        "dataset": dataset_name,
        "model": model_name,
        "sample_id": str(sample.get("id", "")),
        "answer_kind": answer_kind(gt_answer),
        "gt_answer": gt_answer,
        "num_steps": num_steps,
        "mode": args.mode,
        "freeze_filled": bool(args.freeze_filled),
        "trace": rows,
    }


def clean_chain_text(text: str) -> str:
    return str(text or "").replace("[MASK]", "□").replace("\n", "\\n")


def token_for_display(tok: str) -> str:
    tok = str(tok or "")
    tok = tok.replace("[MASK]", "□").replace("\n", "\\n")
    return tok if tok else "∅"


def change_string(prev_tokens: Optional[Sequence[str]], curr_tokens: Sequence[str]) -> str:
    if prev_tokens is None:
        return ""
    changes: List[str] = []
    n = max(len(prev_tokens), len(curr_tokens))
    for i in range(n):
        a = prev_tokens[i] if i < len(prev_tokens) else ""
        b = curr_tokens[i] if i < len(curr_tokens) else ""
        if a != b:
            changes.append(f"{i}:{token_for_display(a)}→{token_for_display(b)}")
    return ";".join(changes)


def compact_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    prev_tokens: Optional[List[str]] = None
    for row in result.get("trace", []):
        toks = list(row.get("answer_token_texts") or [])
        rows.append({
            "dataset": str(result.get("dataset", "")),
            "sample_id": str(result.get("sample_id", "")),
            "model": str(result.get("model", "")),
            "answer_kind": str(result.get("answer_kind", "")),
            "gt_answer": str(result.get("gt_answer", "")),
            "step": int(row.get("step", 0)),
            "t": float(row.get("t", 0.0)),
            "answer": clean_chain_text(row.get("answer_text", "")),
            "change": change_string(prev_tokens, toks),
            "mask_count": int(row.get("mask_count", 0)),
            "changed_answer_indices": ";".join(map(str, row.get("changed_answer_indices", []))),
            "answer_token_ids": " ".join(map(str, row.get("answer_token_ids", []))),
            "answer_token_texts": " | ".join(token_for_display(t) for t in toks),
        })
        prev_tokens = toks
    return rows


def write_detail_csv(path: Path, results: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "dataset", "sample_id", "model", "gt_answer", "answer_kind", "step", "t",
        "answer", "change", "mask_count", "changed_answer_indices", "answer_token_ids", "answer_token_texts",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            if "trace" not in res:
                continue
            for r in compact_rows(res):
                out = {k: r.get(k, "") for k in fieldnames}
                out["t"] = f"{float(out['t']):.4f}"
                writer.writerow(out)


def write_chain_csv(path: Path, results: List[Dict[str, Any]]) -> None:
    compact: List[Dict[str, Any]] = []
    model_order: List[str] = []
    for res in results:
        if "trace" not in res:
            continue
        m = str(res.get("model", ""))
        if m not in model_order:
            model_order.append(m)
        compact.extend(compact_rows(res))

    grouped: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    meta: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in compact:
        dataset = r["dataset"]
        sid = r["sample_id"]
        step = int(r["step"])
        key = (dataset, sid, step)
        if key not in grouped:
            grouped[key] = {"dataset": dataset, "sample_id": sid, "step": step, "t": f"{float(r['t']):.4f}"}
        meta.setdefault((dataset, sid), {"gt_answer": r["gt_answer"], "answer_kind": r["answer_kind"]})
        m = r["model"]
        grouped[key][f"{m}_answer"] = r["answer"]
        grouped[key][f"{m}_change"] = r["change"]

    fieldnames = ["dataset", "sample_id", "gt_answer", "answer_kind", "step", "t"]
    for m in model_order:
        fieldnames += [f"{m}_answer", f"{m}_change"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (dataset, sid, step), row in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
            out = {k: "" for k in fieldnames}
            out.update(row)
            out.update(meta.get((dataset, sid), {}))
            writer.writerow(out)


def write_chain_md(path: Path, results: List[Dict[str, Any]]) -> None:
    by_sample: Dict[Tuple[str, str], Dict[str, Dict[int, Dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    model_order: List[str] = []
    meta: Dict[Tuple[str, str], Dict[str, str]] = {}
    for res in results:
        if "trace" not in res:
            continue
        dataset = str(res.get("dataset", ""))
        sid = str(res.get("sample_id", ""))
        m = str(res.get("model", ""))
        if m not in model_order:
            model_order.append(m)
        meta.setdefault((dataset, sid), {"gt_answer": str(res.get("gt_answer", "")), "answer_kind": str(res.get("answer_kind", ""))})
        for r in compact_rows(res):
            by_sample[(dataset, sid)][m][int(r["step"])] = r

    lines: List[str] = []
    for key in sorted(by_sample.keys()):
        dataset, sid = key
        mta = meta.get(key, {})
        lines.append(f"## {dataset} / {sid}")
        lines.append(f"GT: `{mta.get('gt_answer', '')}`")
        lines.append("")
        header = "| step | t | " + " | ".join([f"{m} | Δ" for m in model_order]) + " |"
        sep = "|---:|---:|" + "|".join(["---|---" for _ in model_order]) + "|"
        lines.append(header)
        lines.append(sep)
        steps = sorted({st for m in model_order for st in by_sample[key].get(m, {}).keys()})
        for st in steps:
            first = None
            for m in model_order:
                first = by_sample[key].get(m, {}).get(st)
                if first:
                    break
            t = f"{float(first['t']):.3f}" if first else ""
            cells = [str(st), t]
            for m in model_order:
                r = by_sample[key].get(m, {}).get(st)
                if r:
                    ans = str(r["answer"]).replace("|", "\\|")
                    chg = str(r["change"]).replace("|", "\\|")
                    cells += [f"`{ans}`", chg]
                else:
                    cells += ["", ""]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate compact SEDD answer recovery chains for eval models.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--start", default=None)
    parser.add_argument("--split", default=None, help="Default: config.eval.split or test.")
    parser.add_argument("--datasets", action="append", default=[], help="Dataset name under parent of current data_dir. Can repeat. Default: config.eval.datasets.")
    parser.add_argument("--models", action="append", default=[], help="Only trace selected model names from config eval.compare_models. Can repeat.")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of samples per dataset.")
    parser.add_argument("--samples-per-type", type=int, default=2)
    parser.add_argument("--sample-id", action="append", default=[], help="Specific sample id. Can repeat; applied within each dataset.")
    parser.add_argument("--run-name", default="sample_chain")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--t-start", type=float, default=0.95)
    parser.add_argument("--t-end", type=float, default=0.01)
    parser.add_argument("--transition-kind", default="analytic", choices=["analytic", "denoise"])
    parser.add_argument("--mode", default="sample", choices=["sample", "greedy"])
    parser.add_argument("--freeze-filled", action="store_true", help="Monotonic-fill diagnostic mode; not the true SEDD chain.")
    parser.add_argument("--strict-checkpoints", action="store_true", help="Error if a configured checkpoint is missing. Default: warn and skip.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    start_name = selected_start_name(cfg, args.start)
    eval_cfg = cfg.get("eval") or {}
    split = str(args.split or eval_cfg.get("split") or "test")
    datasets = resolve_eval_datasets(args, cfg, start_name)
    compare_models = resolve_compare_models(args, cfg, start_name)
    device = choose_device(args, cfg)
    set_seed(args.seed)

    tokenizer = GPT2TokenizerFast.from_pretrained((cfg.get("model") or {}).get("tokenizer", "gpt2"))
    tokenizer.pad_token = tokenizer.eos_token

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_DIR / "experiment" / "sample_chain" / f"{stamp}_{args.run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[chain] device={device}", flush=True)
    print(f"[chain] start={start_name} split={split} datasets={datasets}", flush=True)
    print(f"[chain] models={list(compare_models.keys())}", flush=True)
    print(f"[chain] out={out_dir}", flush=True)

    selected_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for dataset_name in datasets:
        samples = load_samples(cfg, start_name, dataset_name, split, tokenizer)
        chosen = pick_diverse_samples(samples, args.sample_id, int(args.num_samples), int(args.samples_per_type))
        if not chosen:
            print(f"[warn] no samples selected for dataset={dataset_name}", flush=True)
            continue
        selected_by_dataset[dataset_name] = chosen
        print(f"[chain] dataset={dataset_name} loaded={len(samples)} selected={len(chosen)}", flush=True)

    if not selected_by_dataset:
        raise RuntimeError("No samples selected from any dataset.")

    # Resolve checkpoint paths and optionally skip missing, same spirit as eval script.
    model_plan: List[Tuple[str, Path]] = []
    missing: List[Tuple[str, Path]] = []
    for model_name, spec in compare_models.items():
        ckpt = repo_path(spec["checkpoint"])
        if ckpt.exists():
            model_plan.append((model_name, ckpt))
        else:
            missing.append((model_name, ckpt))
    if missing:
        msg = "; ".join([f"{n}={p}" for n, p in missing])
        if args.strict_checkpoints:
            raise FileNotFoundError(f"Missing configured checkpoint(s): {msg}")
        print(f"[warn] skip missing checkpoint(s): {msg}", flush=True)
    if not model_plan:
        raise RuntimeError("No existing checkpoints to trace.")

    summary = {
        "time": stamp,
        "config": str(args.config),
        "start": start_name,
        "split": split,
        "datasets": datasets,
        "device": str(device),
        "steps": int(args.steps),
        "t_start": float(args.t_start),
        "t_end": float(args.t_end),
        "transition_kind": str(args.transition_kind),
        "mode": str(args.mode),
        "freeze_filled": bool(args.freeze_filled),
        "models": [{"name": n, "checkpoint": str(p)} for n, p in model_plan],
        "missing_models": [{"name": n, "checkpoint": str(p)} for n, p in missing],
        "samples": {
            d: [{"id": s.get("id", ""), "answer": extract_gt_answer(s), "kind": answer_kind(extract_gt_answer(s))} for s in ss]
            for d, ss in selected_by_dataset.items()
        },
        "note": "This is a chain dump for configured eval models. It does not calculate reward or score.",
    }
    dump_json(out_dir / "summary.json", summary)

    all_results: List[Dict[str, Any]] = []
    for model_name, ckpt_path in model_plan:
        print(f"[chain] loading {model_name}: {ckpt_path}", flush=True)
        model, graph, noise = load_model_for_trace(cfg, ckpt_path, device)
        try:
            for dataset_name, chosen in selected_by_dataset.items():
                for i, sample in enumerate(chosen):
                    # Same seed for each model/sample/dataset, so stochastic differences come from model probs.
                    dataset_seed = abs(hash(dataset_name)) % 10007
                    set_seed(args.seed + dataset_seed + i * 1009)
                    print(f"[chain] {model_name} dataset={dataset_name} sample={i+1}/{len(chosen)} id={sample.get('id','')} ans={extract_gt_answer(sample)!r}", flush=True)
                    try:
                        res = trace_one_sample(model, graph, noise, tokenizer, sample, cfg, device, args, model_name, dataset_name)
                    except Exception as exc:
                        res = {
                            "dataset": dataset_name,
                            "model": model_name,
                            "sample_id": str(sample.get("id", "")),
                            "gt_answer": extract_gt_answer(sample),
                            "answer_kind": answer_kind(extract_gt_answer(sample)),
                            "error": repr(exc),
                        }
                        print(f"[warn] failed {model_name}/{dataset_name}/{sample.get('id','')}: {exc}", flush=True)
                    all_results.append(res)
        finally:
            del model, graph, noise
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_chain_csv(out_dir / "chain.csv", all_results)
    write_detail_csv(out_dir / "detail.csv", all_results)
    write_chain_md(out_dir / "chain.md", all_results)
    dump_json(out_dir / "errors.json", [r for r in all_results if "error" in r])

    print(f"[chain] wrote {out_dir / 'chain.csv'}", flush=True)
    print(f"[chain] wrote {out_dir / 'detail.csv'}", flush=True)
    print(f"[chain] wrote {out_dir / 'chain.md'}", flush=True)


if __name__ == "__main__":
    main()
