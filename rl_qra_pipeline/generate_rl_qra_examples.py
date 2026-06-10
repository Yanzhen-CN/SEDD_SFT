import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import GPT2TokenizerFast


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
for path in (REPO_DIR, REPO_DIR / "sft_answer_pipeline"):
    sys.path.insert(0, str(path))

from generation_schema import completion_from_full_text, make_generation_record, split_sections, write_generation_markdown  # noqa: E402

DEFAULT_CONFIG = SCRIPT_DIR / "rl_qra_config.yaml"


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def repo_path(path):
    p = Path(path)
    return p if p.is_absolute() else REPO_DIR / p


def load_samples(config, dataset_name, split, tokenizer, limit):
    from answer_dataset import AnswerSegmentDataset

    data_root = repo_path(Path(config["data"]["data_dir"]).parent)
    dataset = AnswerSegmentDataset(
        data_root / dataset_name / f"{split}.jsonl",
        tokenizer,
        int(config["generation"].get("max_length", config["model"].get("max_length", 1024))),
        min_target_tokens=int(config["model"].get("min_target_tokens", 1)),
        drop_overlength=bool(config["model"].get("drop_overlength", True)),
        write_report=bool(config["model"].get("write_load_reports", False)),
    )
    return dataset.samples[:limit]


def load_model_tuple(config, checkpoint, device):
    import graph_lib
    import noise_lib
    from model import SEDD
    from model.ema import ExponentialMovingAverage

    model = SEDD.from_pretrained(config["model"].get("pretrained", "louaaron/sedd-medium")).to(device)
    model.config.model.length = int(config["generation"].get("max_length", config["model"].get("max_length", 1024)))
    ema = ExponentialMovingAverage(model.parameters(), decay=float(config["training"].get("ema", 0.9999)))
    ckpt_path = repo_path(checkpoint)
    if not ckpt_path.exists():
        print(f"Skip {ckpt_path}: not found")
        return None
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state.get("model", state), strict=True)
    if isinstance(state, dict) and "ema" in state:
        ema.load_state_dict(state["ema"])
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    graph = graph_lib.get_graph(model.config, device)
    noise = noise_lib.get_noise(model.config).to(device)
    model.eval()
    return model, graph, noise


def encode_sample(sample, tokenizer, max_length):
    from answer_dataset import ordered_segments

    ids, train_mask = [], []
    for _, segment in ordered_segments(sample):
        toks = tokenizer(segment.get("text", ""), add_special_tokens=False).input_ids
        train = bool(segment.get("train", False))
        ids.extend(toks)
        train_mask.extend([1 if train else 0] * len(toks))
    if len(ids) > max_length:
        raise ValueError(f"{sample.get('id')} exceeds max_length: {len(ids)}>{max_length}")
    if not any(train_mask):
        raise ValueError(f"{sample.get('id')} has no target tokens")
    return ids, train_mask


def sample_infilling(model, graph, noise, tokenizer, sample, length, steps, device):
    import sampling

    ids, train_mask = encode_sample(sample, tokenizer, int(length))
    real_len = len(ids)
    if real_len < int(length):
        pad = int(length) - real_len
        ids += [tokenizer.eos_token_id] * pad
        train_mask += [0] * pad

    fixed_locs = [idx for idx, train in enumerate(train_mask) if not train]
    fixed_ids = torch.tensor([ids[idx] for idx in fixed_locs], device=device)[None]

    def proj_fun(x):
        x[:, fixed_locs] = fixed_ids
        return x

    sampler = sampling.get_pc_sampler(
        graph,
        noise,
        (1, int(length)),
        "analytic",
        int(steps),
        device=device,
        proj_fun=proj_fun,
    )
    with torch.no_grad():
        generated = proj_fun(sampler(model))
        full_text = tokenizer.batch_decode(generated[:, :real_len])[0].strip()
    return completion_from_full_text(full_text)


def qra_sections(row, completion):
    sections = split_sections(completion, default_reasoning=row.get("reasoning", ""))
    sections["reasoning"] = str(row.get("reasoning", "")).strip()
    return sections


def main():
    parser = argparse.ArgumentParser(description="Generate RL-QRA qualitative examples.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config = load_config(args.config)
    dataset_name = config.get("generation", {}).get("dataset", "QRA")
    device = torch.device("cuda" if torch.cuda.is_available() and not bool(config.get("cpu", False)) else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained(config["model"].get("tokenizer", "gpt2"))
    tokenizer.pad_token = tokenizer.eos_token

    rows = load_samples(config, dataset_name, "test", tokenizer, int(config["generation"].get("num_examples", 5)))
    loaded = {}
    for model_name, model_cfg in (config.get("compare_models") or {}).items():
        checkpoint = model_cfg.get("checkpoint") if isinstance(model_cfg, dict) else model_cfg
        model_tuple = load_model_tuple(config, checkpoint, device)
        if model_tuple is not None:
            loaded[model_name] = model_tuple

    records = []
    data_path = repo_path(Path(config["data"]["data_dir"]).parent) / dataset_name / "test.jsonl"
    for row in rows:
        generations = {}
        for model_name, (model, graph, noise) in loaded.items():
            completion = sample_infilling(
                model,
                graph,
                noise,
                tokenizer,
                row,
                int(config["generation"].get("max_length", 1024)),
                int(config["generation"].get("steps", 128)),
                device,
            )
            generations[model_name] = qra_sections(row, completion)
        records.append(make_generation_record(row, row.get("mode", dataset_name), "test", data_path, generations))

    out_dir = repo_path(config["generation"].get("output_dir", "rl_qra_pipeline/reports"))
    write_generation_markdown(out_dir / f"rl_qra_generation_{dataset_name}.md", f"RL QRA Generation {dataset_name}", records)
    (out_dir / f"rl_qra_generation_{dataset_name}.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {out_dir / f'rl_qra_generation_{dataset_name}.md'}")


if __name__ == "__main__":
    main()
