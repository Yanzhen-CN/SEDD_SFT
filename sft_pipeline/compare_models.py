import argparse
import gc
import json
import re
import sys
from pathlib import Path

import yaml

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

DEFAULT_CONFIG = Path(__file__).resolve().parent / "sft_config.yaml"
DEFAULT_MODEL_ROOT = Path(__file__).resolve().parent / "modelparameter"
DEFAULT_QA_DATA = Path(__file__).resolve().parent / "data" / "QA"
DEFAULT_QAR_DATA = Path(__file__).resolve().parent / "data" / "QAR"

LOSS_RE = re.compile(r"step:\s*(\d+),\s*(training|evaluation)_loss:\s*([0-9.eE+-]+)")


def load_comparison_config(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config.get("comparison", {})


def choose(cli_value, config, key, default=None):
    return cli_value if cli_value is not None else config.get(key, default)


def dump_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def cleanup_cuda():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


def parse_losses(run_dir):
    # kept for potential internal use, but not used in output
    pass


def get_source_run_dir(run_dir):
    run_path = Path(run_dir)
    best_eval_path = run_path / "best_eval.json"
    if best_eval_path.exists():
        best_eval = json.loads(best_eval_path.read_text(encoding="utf-8"))
        if best_eval.get("source_run_dir"):
            return best_eval["source_run_dir"]
    run_info_path = run_path / "run_info.json"
    if run_info_path.exists():
        run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
        if run_info.get("work_dir"):
            return run_info["work_dir"]
    return str(run_path)


def find_checkpoint(run_dir, checkpoint_name="best"):
    run_path = Path(run_dir)
    if checkpoint_name == "best":
        candidates = [
            run_path / "best.pth",
            run_path / "checkpoints" / "best.pth",
            run_path / "checkpoints-meta" / "checkpoint.pth",
        ]
    else:
        candidates = [run_path / checkpoint_name]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No checkpoint found under {run_dir}")


def load_local_model(run_dir, device, checkpoint_name="best"):
    import torch
    import graph_lib
    import noise_lib
    import utils
    from model import SEDD
    from model.ema import ExponentialMovingAverage

    run_path = Path(run_dir)
    ckpt_path = find_checkpoint(run_path, checkpoint_name=checkpoint_name)
    loaded_state = torch.load(ckpt_path, map_location=device)
    meta = loaded_state.get("meta", {}) or {}

    config_dir = run_path

    best_eval_path = run_path / "best_eval.json"
    if best_eval_path.exists():
        best_eval = json.loads(best_eval_path.read_text(encoding="utf-8"))
        if best_eval.get("source_run_dir"):
            config_dir = Path(best_eval["source_run_dir"])

    run_info_path = run_path / "run_info.json"
    if run_info_path.exists():
        run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
        if run_info.get("work_dir"):
            config_dir = Path(run_info["work_dir"])

    try:
        cfg = utils.load_hydra_config_from_run(str(config_dir))
        model = SEDD(cfg).to(device)
        graph = graph_lib.get_graph(cfg, device)
        noise = noise_lib.get_noise(cfg).to(device)
        ema_decay = float(getattr(cfg.training, "ema", 0.9999))
    except Exception:
        pretrained_name = meta.get("pretrained", "louaaron/sedd-medium")
        model = SEDD.from_pretrained(pretrained_name).to(device)
        if meta.get("max_length") is not None:
            model.config.model.length = int(meta["max_length"])
        graph = graph_lib.get_graph(model.config, device)
        noise = noise_lib.get_noise(model.config).to(device)
        ema_decay = float(meta.get("ema", 0.9999))

    ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)

    model.load_state_dict(loaded_state["model"])
    if "ema" in loaded_state:
        ema.load_state_dict(loaded_state["ema"])
        ema.store(model.parameters())
        ema.copy_to(model.parameters())

    model.eval()
    if hasattr(noise, "eval"):
        noise.eval()

    return model, graph, noise, str(ckpt_path)


def row_to_raw_sample(row):
    if row.get("text"):
        return row["text"], extract_prompt(row["text"])
    prompt = row.get("prompt", "")
    target = row.get("target", "")
    raw = prompt + target
    return raw, prompt if prompt else extract_prompt(raw)


def load_jsonl_example(data_dir, split="test", index=0):
    data_dir = Path(data_dir)
    candidate_splits = [split]
    for fallback in ["validation", "train"]:
        if fallback not in candidate_splits:
            candidate_splits.append(fallback)
    for split_name in candidate_splits:
        path = data_dir / f"{split_name}.jsonl"
        if not path.exists():
            continue
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("text") or row.get("prompt") or row.get("target"):
                    rows.append(row)
        if rows:
            idx = index % len(rows)
            raw_sample, prompt = row_to_raw_sample(rows[idx])
            return {
                "split": split_name,
                "index": idx,
                "path": str(path),
                "raw_sample": raw_sample,
                "prompt": prompt,
            }
    raise FileNotFoundError(f"No usable jsonl split found under {data_dir}")


def extract_prompt(raw_text):
    marker = "\nAssistant:"
    pos = raw_text.find(marker)
    if pos < 0:
        return raw_text[: min(len(raw_text), 256)]
    end = pos + len(marker)
    if end < len(raw_text) and raw_text[end] in {" ", "\n"}:
        end += 1
    return raw_text[:end]


def completion_after_prompt(generated_text, prompt):
    if generated_text.startswith(prompt):
        return generated_text[len(prompt):]
    marker = "Assistant:"
    pos = generated_text.find(marker)
    if pos >= 0:
        return generated_text[pos + len(marker):]
    return generated_text


def read_test_results(model_root):
    # Not used in compact output
    pass


def sample_text(model, graph, noise, tokenizer, prompt, length, steps, device, seed=None):
    import torch
    import sampling

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    prefix_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if len(prefix_ids) >= length:
        prefix_ids = prefix_ids[: length - 1]

    input_ids = torch.tensor(prefix_ids, device=device)[None]
    input_locs = list(range(len(prefix_ids)))

    def proj_fun(x):
        x[:, input_locs] = input_ids
        return x

    sampling_fn = sampling.get_pc_sampler(
        graph,
        noise,
        (1, length),
        "analytic",
        steps,
        device=device,
        proj_fun=proj_fun,
    )

    model.eval()
    with torch.no_grad():
        sample = proj_fun(sampling_fn(model))

    return tokenizer.batch_decode(sample)[0]


def run_model_inference(model_key, checkpoint_path, examples, tokenizer, length, steps, device, seed):
    import copy
    import torch
    from load_model import load_model

    new_examples = copy.deepcopy(examples)

    model = graph = noise = None
    try:
        print(f"\n=== Running {model_key} from {checkpoint_path} ===")
        if checkpoint_path.startswith("louaaron/"):
            model, graph, noise = load_model(checkpoint_path, device)
        else:
            model, graph, noise, _ = load_local_model(checkpoint_path, device)

        if hasattr(noise, "eval"):
            noise.eval()

        for ex in new_examples:
            prompt = ex["prompt"]
            restored_text = sample_text(
                model=model,
                graph=graph,
                noise=noise,
                tokenizer=tokenizer,
                prompt=prompt,
                length=length,
                steps=steps,
                device=device,
                seed=seed + ex["index"],
            )
            ex["model_outputs"][model_key] = {
                "checkpoint": checkpoint_path,
                "restored_text": restored_text,
                "generated_after_prompt": completion_after_prompt(restored_text, prompt),
            }
        return new_examples
    finally:
        del model, graph, noise
        cleanup_cuda()
        print(f"Cleared GPU memory after {model_key}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate compact JSON comparison: for each dataset, show prompt and model outputs."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--qa-run", default=None, help="Path to QA best/run directory.")
    parser.add_argument("--qar-run", default=None, help="Path to QAR best/run directory.")
    parser.add_argument("--model-root", default=None, help="Pipeline modelparameter directory.")
    parser.add_argument("--qa-data", default=None)
    parser.add_argument("--qar-data", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-sample", action="store_true", default=None)
    args = parser.parse_args()

    comparison_cfg = load_comparison_config(args.config)

    model_root = Path(choose(args.model_root, comparison_cfg, "model_root", str(DEFAULT_MODEL_ROOT)))
    pretrained = choose(args.pretrained, comparison_cfg, "pretrained", "louaaron/sedd-medium")
    qa_run = choose(args.qa_run, comparison_cfg, "qa_run")
    qar_run = choose(args.qar_run, comparison_cfg, "qar_run")
    qa_data = Path(choose(args.qa_data, comparison_cfg, "qa_data", str(DEFAULT_QA_DATA)))
    qar_data = Path(choose(args.qar_data, comparison_cfg, "qar_data", str(DEFAULT_QAR_DATA)))
    split = choose(args.split, comparison_cfg, "split", "test")
    sample_index = int(choose(args.sample_index, comparison_cfg, "sample_index", 0))
    length = int(choose(args.length, comparison_cfg, "length", 512))
    steps = int(choose(args.steps, comparison_cfg, "steps", 128))
    seed = int(choose(args.seed, comparison_cfg, "seed", 42))
    out = choose(args.out, comparison_cfg, "out", "sft_pipeline/reports/model_comparison.json")
    no_sample = bool(choose(args.no_sample, comparison_cfg, "no_sample", False))

    out_path = Path(out)
    if out_path.suffix.lower() != ".json":
        out_path = out_path.with_suffix(".json")

    # Auto-detect run directories if not given
    if qa_run is None and (model_root / "QA" / "best.pth").exists():
        qa_run = str(model_root / "QA")
    if qar_run is None and (model_root / "QAR" / "best.pth").exists():
        qar_run = str(model_root / "QAR")

    # Load one example from QA and QAR datasets
    qa_example = load_jsonl_example(qa_data, split=split, index=sample_index)
    qar_example = load_jsonl_example(qar_data, split=split, index=sample_index)

    examples = [
        {
            "dataset": "QA-test",
            "split": qa_example["split"],
            "index": qa_example["index"],
            "prompt": qa_example["prompt"],
            "model_outputs": {},
        },
        {
            "dataset": "QAR-test",
            "split": qar_example["split"],
            "index": qar_example["index"],
            "prompt": qar_example["prompt"],
            "model_outputs": {},
        },
    ]

    if not no_sample:
        import torch
        from transformers import GPT2TokenizerFast

        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        models_to_run = []
        if pretrained:
            models_to_run.append(("pretrain", pretrained))
        if qa_run:
            models_to_run.append(("QA-best", qa_run))
        if qar_run:
            models_to_run.append(("QAR-best", qar_run))

        for model_key, checkpoint in models_to_run:
            examples = run_model_inference(
                model_key=model_key,
                checkpoint_path=checkpoint,
                examples=examples,
                tokenizer=tokenizer,
                length=length,
                steps=steps,
                device=device,
                seed=seed,
            )

    # Build compact report
    compact_report = {}
    for ex in examples:
        ds = ex["dataset"]
        compact_report[ds] = {
            "sample": ex["prompt"],
            "pretrain": ex["model_outputs"].get("pretrain", {}).get("generated_after_prompt", None),
            "QA-best": ex["model_outputs"].get("QA-best", {}).get("generated_after_prompt", None),
            "QAR-best": ex["model_outputs"].get("QAR-best", {}).get("generated_after_prompt", None),
        }

    dump_json(compact_report, out_path)
    print(f"Wrote compact comparison to {out_path}")


if __name__ == "__main__":
    main()