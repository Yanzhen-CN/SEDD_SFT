from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from transformers import GPT2TokenizerFast

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

for path in (
    REPO_DIR,
    SCRIPT_DIR,
    REPO_DIR / "sft_answer_pipeline",
    REPO_DIR / "sft_rl_pipeline",
):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

DEFAULT_CONFIG = SCRIPT_DIR / "rl_qra_config.yaml"


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_DIR / p


def dump_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = ["dataset", "model", "loss", "batches", "checkpoint"]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def selected_start_name(config: Dict[str, Any]) -> str:
    run_cfg = config.get("run", {}) or {}
    return str(run_cfg.get("selected") or config.get("selected") or "QRA")


def selected_start_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    starts = config.get("starts", {}) or {}
    return starts.get(selected_start_name(config), {}) or {}


def resolve_data_dir(config: Dict[str, Any]) -> Path:
    data_cfg = config.get("data", {}) or {}
    start_cfg = selected_start_cfg(config)

    data_dir = (
        data_cfg.get("data_dir")
        or start_cfg.get("data_dir")
        or "rl_qra_pipeline/data/S1K_RL"
    )
    return repo_path(data_dir)


def default_compare_models(config: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    start_name = selected_start_name(config)
    start_cfg = selected_start_cfg(config)
    output_dir = start_cfg.get("output_dir", f"rl_qra_pipeline/modelparameter/rl_{start_name}")

    compare = {}
    init_ckpt = start_cfg.get("init_checkpoint")
    if init_ckpt:
        compare[f"{start_name}_base"] = {"checkpoint": init_ckpt}

    compare[f"rl_{start_name}"] = {"checkpoint": str(Path(output_dir) / "best.pth")}
    return compare


def make_loader(config: Dict[str, Any], dataset_name: str, split: str):
    from answer_dataset import make_answer_loader

    tokenizer = GPT2TokenizerFast.from_pretrained(config.get("model", {}).get("tokenizer", "gpt2"))
    tokenizer.pad_token = tokenizer.eos_token

    data_dir = resolve_data_dir(config)

    # Normal case: data_dir points to rl_qra_pipeline/data/S1K_RL.
    if dataset_name in {".", "current", data_dir.name}:
        data_path = data_dir / f"{split}.jsonl"
    else:
        # Multi-dataset comparison case: data root is the parent of data_dir.
        data_path = data_dir.parent / dataset_name / f"{split}.jsonl"

    if not data_path.exists():
        raise FileNotFoundError(f"Eval data not found: {data_path}")

    model_cfg = config.get("model", {}) or {}
    eval_cfg = config.get("eval", {}) or {}

    return make_answer_loader(
        data_path,
        tokenizer,
        int(model_cfg.get("max_length", 1024)),
        int(model_cfg.get("min_target_tokens", 1)),
        int(eval_cfg.get("batch_size", 1)),
        False,
        int(eval_cfg.get("num_workers", 0)),
        drop_overlength=bool(model_cfg.get("drop_overlength", True)),
        write_report=bool(model_cfg.get("write_load_reports", False)),
    )


def load_model(config: Dict[str, Any], checkpoint: str, device: torch.device):
    import graph_lib
    import noise_lib
    from model import SEDD
    from model.ema import ExponentialMovingAverage

    model_cfg = config.get("model", {}) or {}
    training_cfg = config.get("training", {}) or {}

    ckpt_path = repo_path(checkpoint)
    if not ckpt_path.exists():
        print(f"[eval] skip missing checkpoint: {ckpt_path}", flush=True)
        return None

    model = SEDD.from_pretrained(model_cfg.get("pretrained", "louaaron/sedd-medium")).to(device)
    model.config.model.length = int(model_cfg.get("max_length", model.config.model.length))

    ema = ExponentialMovingAverage(model.parameters(), decay=float(training_cfg.get("ema", 0.9999)))
    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "model" in state:
        model_state = state["model"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        model_state = state["model_state_dict"]
    else:
        model_state = state

    model.load_state_dict(model_state, strict=True)

    if isinstance(state, dict) and "ema" in state:
        ema.load_state_dict(state["ema"])

    graph = graph_lib.get_graph(model.config, device)
    noise = noise_lib.get_noise(model.config).to(device)
    return model, graph, noise, ema, str(ckpt_path)


def eval_model(
    config: Dict[str, Any],
    model_name: str,
    model_tuple: Any,
    datasets: List[str],
    split: str,
    device: torch.device,
) -> List[Dict[str, Any]]:
    from answer_losses import evaluate_answer_loss

    model, graph, noise, ema, checkpoint = model_tuple
    eval_cfg = config.get("eval", {}) or {}
    eval_batches = int(eval_cfg.get("eval_batches", 0))

    rows: List[Dict[str, Any]] = []
    for dataset_name in datasets:
        loader = make_loader(config, dataset_name, split)
        loss, batches = evaluate_answer_loss(
            model,
            ema,
            noise,
            graph,
            loader,
            device,
            eval_batches=eval_batches,
        )
        row = {
            "dataset": dataset_name,
            "split": split,
            "model": model_name,
            "loss": float(loss),
            "batches": int(batches),
            "checkpoint": checkpoint,
        }
        rows.append(row)
        print(row, flush=True)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RL-QRA checkpoints on S1K_RL/QRA test or val split.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", default=None, help="Default: config.eval.split or test.")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    eval_cfg = config.get("eval", {}) or {}

    if args.cpu or bool(config.get("cpu", False)) or not torch.cuda.is_available():
        device = torch.device("cpu")
    elif args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        run_cfg = config.get("run", {}) or {}
        cfg_gpu = run_cfg.get("cuda_device", None)
        device = torch.device("cuda" if cfg_gpu in {None, "", "null", "none"} else f"cuda:{int(cfg_gpu)}")

    split = str(args.split or eval_cfg.get("split") or "test")
    data_dir = resolve_data_dir(config)
    datasets = list(eval_cfg.get("datasets") or [data_dir.name])
    compare_models = config.get("compare_models") or default_compare_models(config)

    print(f"[eval] device={device}", flush=True)
    print(f"[eval] data_dir={data_dir}", flush=True)
    print(f"[eval] datasets={datasets} split={split}", flush=True)
    print(f"[eval] compare_models={list(compare_models.keys())}", flush=True)

    rows: List[Dict[str, Any]] = []
    for model_name, model_cfg in compare_models.items():
        checkpoint = model_cfg.get("checkpoint") if isinstance(model_cfg, dict) else str(model_cfg)
        model_tuple = load_model(config, checkpoint, device)
        if model_tuple is None:
            continue

        try:
            rows.extend(eval_model(config, model_name, model_tuple, datasets, split, device))
        finally:
            del model_tuple
            cleanup()

    out_dir = repo_path(eval_cfg.get("output_dir", "rl_qra_pipeline/test_result"))
    write_csv(rows, out_dir / "test_results.csv")
    dump_json(rows, out_dir / "test_results.json")

    # Robust: config may not have output block.
    output_root = repo_path((config.get("output") or {}).get("root_dir", "rl_qra_pipeline/modelparameter"))
    for model_name in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model_name]
        model_dir = output_root / model_name
        dump_json(model_rows, model_dir / "best_test_result.json")
        write_csv(model_rows, model_dir / "best_test_result.csv")

    print(f"[eval] wrote {out_dir / 'test_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
