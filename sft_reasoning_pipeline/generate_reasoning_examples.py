import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import GPT2TokenizerFast

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
ANSWER_PIPELINE_DIR = REPO_DIR / "sft_answer_pipeline"
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(ANSWER_PIPELINE_DIR))

DEFAULT_CONFIG = SCRIPT_DIR / "reasoning_config.yaml"


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def selected_run(config):
    selected = config.get("run", {}).get("selected", "QRA")
    if selected == "all":
        return (list(config.get("runs", {}).keys()) or ["QRA"])[0]
    if isinstance(selected, list):
        return selected[0]
    return selected


def dataset_for_run(config, run_name):
    return config.get("runs", {}).get(run_name, {}).get("dataset", run_name)


def segment_text(sample, train=None):
    from answer_dataset import sample_text
    return sample_text(sample, train=train)


def segment_value(sample, name, default=""):
    segment = sample.get("segments", {}).get(name)
    if not segment:
        return default
    return segment.get("text", default)


def load_filtered_samples(config, dataset_name, tokenizer, limit):
    from answer_dataset import AnswerSegmentDataset

    data_root = Path(config["data"].get("output_dir", SCRIPT_DIR / "data"))
    dataset = AnswerSegmentDataset(
        data_root / dataset_name / "test.jsonl",
        tokenizer,
        int(config["generation"].get("max_length", config["model"].get("max_length", 512))),
        min_target_tokens=int(config["model"].get("min_target_tokens_by_mode", {}).get(dataset_name, config["model"].get("min_target_tokens", 1))),
        drop_overlength=bool(config["model"].get("drop_overlength", True)),
        write_report=bool(config["model"].get("write_load_reports", True)),
    )
    return dataset.samples[:limit]


def load_model_tuple(config, kind, run_name, device):
    import graph_lib
    import noise_lib
    from model import SEDD
    from model.ema import ExponentialMovingAverage

    pretrained = config["model"].get("pretrained", "louaaron/sedd-medium")
    model = SEDD.from_pretrained(pretrained).to(device)
    model.config.model.length = int(config["generation"].get("max_length", config["model"].get("max_length", 512)))
    ema = ExponentialMovingAverage(model.parameters(), decay=float(config["training"].get("ema", 0.9999)))
    checkpoint = pretrained

    if kind == run_name:
        ckpt = Path(config["results"].get("output_dir", SCRIPT_DIR / "modelparameter")) / run_name / "best.pth"
        if not ckpt.exists():
            return None
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"], strict=True)
        ema.load_state_dict(state["ema"])
        checkpoint = str(ckpt)
        ema.store(model.parameters())
        ema.copy_to(model.parameters())

    graph = graph_lib.get_graph(model.config, device)
    noise = noise_lib.get_noise(model.config).to(device)
    model.eval()
    return model, graph, noise, checkpoint


def encode_sample_no_truncation(sample, tokenizer, max_length):
    from answer_dataset import ordered_segments

    ids, train_mask = [], []
    for _, segment in ordered_segments(sample):
        token_ids = tokenizer(segment.get("text", ""), add_special_tokens=False).input_ids
        is_train = bool(segment.get("train", False))
        ids.extend(token_ids)
        train_mask.extend([1 if is_train else 0] * len(token_ids))
    if len(ids) > max_length:
        raise ValueError(f"Sample {sample.get('id')} is over-length after filtering: {len(ids)}>{max_length}")
    if not any(train_mask):
        raise ValueError(f"Sample {sample.get('id')} has no target tokens")
    return ids, train_mask


def sample_infilling(model, graph, noise, tokenizer, sample, length, steps, device):
    import sampling

    ids, train_mask = encode_sample_no_truncation(sample, tokenizer, int(length))
    real_len = len(ids)
    pad_len = int(length) - real_len
    if pad_len > 0:
        ids += [tokenizer.eos_token_id] * pad_len
        train_mask += [0] * pad_len
    fixed_locs = [i for i, is_train in enumerate(train_mask) if not is_train]
    fixed_ids = torch.tensor([ids[i] for i in fixed_locs], device=device)[None]

    def proj_fun(x):
        x[:, fixed_locs] = fixed_ids
        return x

    sampling_fn = sampling.get_pc_sampler(graph, noise, (1, int(length)), "analytic", int(steps), device=device, proj_fun=proj_fun)
    target_positions = [i for i, is_train in enumerate(train_mask[:real_len]) if is_train]
    with torch.no_grad():
        generated = proj_fun(sampling_fn(model))
    target_ids = generated[0, target_positions]
    return tokenizer.decode(target_ids).strip()


def write_report(path, records, run_name):
    lines = [f"# {run_name} Reasoning-conditioned Answer Generation Examples", ""]
    for item in records:
        lines.append(f"## {item['id']}")
        lines.append("")
        lines.append("### Fixed question + teacher reasoning")
        lines.append("```text")
        lines.append(item["prompt"].strip())
        lines.append("```")
        lines.append("")
        lines.append("### Reference answer")
        lines.append("```text")
        lines.append(item["reference"].strip())
        lines.append("```")
        lines.append("")
        for model_name, text in item["generations"].items():
            lines.append(f"### {model_name}")
            lines.append("```text")
            lines.append(text.strip())
            lines.append("```")
            lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate QRA examples: fixed teacher reasoning -> answer.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config = load_config(args.config)
    run_name = selected_run(config)
    dataset_name = dataset_for_run(config, run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    rows = load_filtered_samples(config, dataset_name, tokenizer, int(config["generation"].get("num_examples", 5)))

    loaded = {}
    for kind in ["pretrained", run_name]:
        model_tuple = load_model_tuple(config, kind if kind != "pretrained" else "pretrained", run_name, device)
        if model_tuple is not None:
            loaded[kind] = model_tuple

    records = []
    for idx, row in enumerate(rows):
        item = {
            "id": row.get("id", str(idx)),
            "question": segment_value(row, "user"),
            "teacher_reasoning": segment_value(row, "reasoning"),
            "prompt": segment_text(row, train=False),
            "reference": segment_text(row, train=True),
            "generations": {},
        }
        for model_name, (model, graph, noise, _) in loaded.items():
            item["generations"][model_name] = sample_infilling(
                model,
                graph,
                noise,
                tokenizer,
                row,
                int(config["generation"].get("max_length", 512)),
                int(config["generation"].get("steps", 128)),
                device,
            )
        records.append(item)

    out_dir = Path(config["generation"].get("output_dir", SCRIPT_DIR / "reports"))
    write_report(out_dir / f"reasoning_answer_generation_{run_name}.md", records, run_name)
    (out_dir / f"reasoning_answer_generation_{run_name}.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / f'reasoning_answer_generation_{run_name}.md'}", flush=True)


if __name__ == "__main__":
    main()
