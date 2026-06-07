import argparse
import json
import re
import sys
from pathlib import Path

import yaml

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
DEFAULT_CONFIG = Path(__file__).resolve().parent / "sft_config.yaml"


LOSS_RE = re.compile(r"step:\s*(\d+),\s*(training|evaluation)_loss:\s*([0-9.eE+-]+)")


def load_comparison_config(path):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config.get("comparison", {})


def choose(cli_value, config, key, default=None):
    return cli_value if cli_value is not None else config.get(key, default)


def parse_losses(run_dir):
    run_path = Path(run_dir)
    log_path = run_path / "logs"
    if not log_path.exists():
        log_path = run_path / "train.log"
    if not log_path.exists():
        return {"run_dir": str(run_path), "error": "logs/train.log file not found"}

    rows = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = LOSS_RE.search(line)
        if match:
            rows.append({
                "step": int(match.group(1)),
                "kind": match.group(2),
                "loss": float(match.group(3)),
            })

    eval_rows = [row for row in rows if row["kind"] == "evaluation"]
    train_rows = [row for row in rows if row["kind"] == "training"]
    best_eval = min(eval_rows, key=lambda row: row["loss"]) if eval_rows else None
    final_eval = eval_rows[-1] if eval_rows else None
    final_train = train_rows[-1] if train_rows else None

    best_file = run_path / "best_eval.json"
    saved_best = None
    if best_file.exists():
        saved_best = json.loads(best_file.read_text(encoding="utf-8"))

    return {
        "run_dir": str(run_path),
        "best_eval_from_log": best_eval,
        "best_eval_saved": saved_best,
        "final_eval": final_eval,
        "final_train": final_train,
        "num_loss_points": len(rows),
    }


def load_local_model(run_dir, device, checkpoint_name="best"):
    import torch
    import graph_lib
    import noise_lib
    import utils
    from model import SEDD
    from model.ema import ExponentialMovingAverage

    run_path = Path(run_dir)
    config_dir = run_path
    run_info_path = run_path / "run_info.json"
    if run_info_path.exists():
        run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
        config_dir = Path(run_info["work_dir"])

    cfg = utils.load_hydra_config_from_run(str(config_dir))
    model = SEDD(cfg).to(device)
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg.training.ema)
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)

    if checkpoint_name == "best":
        ckpt_path = run_path / "checkpoints" / "best.pth"
        if not ckpt_path.exists():
            ckpt_path = run_path / "best.pth"
        if not ckpt_path.exists():
            ckpt_path = run_path / "checkpoints-meta" / "checkpoint.pth"
    else:
        ckpt_path = run_path / checkpoint_name

    loaded_state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(loaded_state["model"])
    ema.load_state_dict(loaded_state["ema"])
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    return model, graph, noise, str(ckpt_path)


def load_random_model(reference_run_dir, device):
    import graph_lib
    import noise_lib
    import utils
    from model import SEDD

    cfg = utils.load_hydra_config_from_run(reference_run_dir)
    model = SEDD(cfg).to(device)
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    return model, graph, noise


def sample_text(model, graph, noise, tokenizer, prefix, length, steps, device):
    import torch
    import sampling

    prefix_ids = tokenizer(prefix).input_ids
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
    with torch.no_grad():
        sample = proj_fun(sampling_fn(model))
    return tokenizer.batch_decode(sample)[0]


def write_report(path, losses, samples):
    lines = ["# SFT Model Comparison", ""]
    lines.append("## Loss Summary")
    lines.append("")
    for name, stats in losses.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(stats, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    lines.append("## Samples")
    lines.append("")
    for name, text in samples.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("```text")
        lines.append(text.strip())
        lines.append("```")
        lines.append("")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Compare random, pretrained, QA-SFT, and QAR-SFT outputs.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--qa-run", default=None, help="Path to an exp_local/sft_QA run directory.")
    parser.add_argument("--qar-run", default=None, help="Path to an exp_local/sft_QAR run directory.")
    parser.add_argument("--random-like", default=None, help="Use this run config for a random-init baseline.")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-sample", action="store_true", default=None)
    args = parser.parse_args()
    comparison_cfg = load_comparison_config(args.config)

    pretrained = choose(args.pretrained, comparison_cfg, "pretrained", "louaaron/sedd-medium")
    qa_run = choose(args.qa_run, comparison_cfg, "qa_run")
    qar_run = choose(args.qar_run, comparison_cfg, "qar_run")
    random_like = choose(args.random_like, comparison_cfg, "random_like")
    prefix = choose(args.prefix, comparison_cfg, "prefix", "User: Solve the problem carefully.\nAssistant:\n")
    length = int(choose(args.length, comparison_cfg, "length", 512))
    steps = int(choose(args.steps, comparison_cfg, "steps", 128))
    out = choose(args.out, comparison_cfg, "out", "sft_pipeline/reports/model_comparison.md")
    no_sample = bool(choose(args.no_sample, comparison_cfg, "no_sample", False))

    losses = {}
    if qa_run:
        losses["QA"] = parse_losses(qa_run)
    if qar_run:
        losses["QAR"] = parse_losses(qar_run)

    samples = {}
    if not no_sample:
        import torch
        from transformers import GPT2TokenizerFast
        from load_model import load_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        if random_like:
            model, graph, noise = load_random_model(random_like, device)
            samples["random_init"] = sample_text(model, graph, noise, tokenizer, prefix, length, steps, device)

        model, graph, noise = load_model(pretrained, device)
        samples["pretrained"] = sample_text(model, graph, noise, tokenizer, prefix, length, steps, device)

        if qa_run:
            model, graph, noise, ckpt = load_local_model(qa_run, device)
            samples["QA_best"] = sample_text(model, graph, noise, tokenizer, prefix, length, steps, device)
            losses.setdefault("QA", {})["sample_checkpoint"] = ckpt

        if qar_run:
            model, graph, noise, ckpt = load_local_model(qar_run, device)
            samples["QAR_best"] = sample_text(model, graph, noise, tokenizer, prefix, length, steps, device)
            losses.setdefault("QAR", {})["sample_checkpoint"] = ckpt

    write_report(out, losses, samples)
    print(f"Wrote comparison report to {out}")


if __name__ == "__main__":
    main()
