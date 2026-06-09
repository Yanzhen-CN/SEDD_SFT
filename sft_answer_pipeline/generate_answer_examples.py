import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import GPT2TokenizerFast

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_DIR))
DEFAULT_CONFIG = SCRIPT_DIR / "answer_config.yaml"


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_filtered_samples(config, dataset_name, split, tokenizer, limit):
    from answer_dataset import AnswerSegmentDataset

    data_root = Path(config["data"].get("output_dir", SCRIPT_DIR / "data"))
    dataset = AnswerSegmentDataset(
        data_root / dataset_name / f"{split}.jsonl",
        tokenizer,
        int(config["generation"].get("max_length", config["model"].get("max_length", 512))),
        min_target_tokens=int(config["model"].get("min_target_tokens", 32)),
        drop_overlength=bool(config["model"].get("drop_overlength", True)),
        write_report=bool(config["model"].get("write_load_reports", True)),
    )
    return dataset.samples[:limit]


def segment_text(sample, train=None):
    from answer_dataset import sample_text

    return sample_text(sample, train=train)


def assistant_completion(sample):
    """Return text after the fixed `Assistant:` header, including fixed anchors.

    This is important when `Reasoning:` and `Answer:` are train=False anchors:
    `sample_text(train=True)` contains only the generated contents and therefore
    cannot show the section boundary.
    """
    from answer_dataset import ordered_segments

    parts = []
    seen_assistant = False
    for name, segment in ordered_segments(sample):
        if seen_assistant:
            parts.append(segment.get("text", ""))
        if name == "assistant_label":
            seen_assistant = True
    return "".join(parts)


def completion_from_full_text(text):
    raw = str(text or "")
    marker = "Assistant:"
    idx = raw.find(marker)
    if idx >= 0:
        return raw[idx + len(marker):].strip()
    return raw.strip()


def load_model_tuple(config, kind, device):
    import graph_lib
    import noise_lib
    from model import SEDD
    from model.ema import ExponentialMovingAverage

    pretrained = config["model"].get("pretrained", "louaaron/sedd-medium")
    model = SEDD.from_pretrained(pretrained).to(device)
    model.config.model.length = int(config["generation"].get("max_length", config["model"].get("max_length", 512)))
    ema = ExponentialMovingAverage(model.parameters(), decay=float(config["training"].get("ema", 0.9999)))
    checkpoint = pretrained

    if kind in ["QA", "QAR"]:
        ckpt = Path(config["results"].get("output_dir", SCRIPT_DIR / "modelparameter")) / kind / "best.pth"
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

    ids = []
    train_mask = []
    for _, segment in ordered_segments(sample):
        token_ids = tokenizer(segment.get("text", ""), add_special_tokens=False).input_ids
        is_train = bool(segment.get("train", False))
        ids.extend(token_ids)
        train_mask.extend([1 if is_train else 0] * len(token_ids))
    if len(ids) > max_length:
        raise ValueError(
            f"Filtered generation sample {sample.get('id')} is still over-length: "
            f"{len(ids)} > {max_length}. Regenerate or reload with stricter filtering."
        )
    if not any(train_mask):
        raise ValueError(f"Generation sample {sample.get('id')} has no train target tokens.")
    return ids, train_mask


def sample_segment_infilling(model, graph, noise, tokenizer, sample, length, steps, device):
    import sampling

    ids, train_mask = encode_sample_no_truncation(sample, tokenizer, int(length))
    real_len = len(ids)
    pad_len = int(length) - real_len
    if pad_len > 0:
        ids = ids + [tokenizer.eos_token_id] * pad_len
        train_mask = train_mask + [0] * pad_len

    fixed_locs = [idx for idx, is_train in enumerate(train_mask) if not is_train]
    fixed_ids = torch.tensor([ids[idx] for idx in fixed_locs], device=device)[None]

    def proj_fun(x):
        x[:, fixed_locs] = fixed_ids
        return x

    sampling_fn = sampling.get_pc_sampler(
        graph,
        noise,
        (1, int(length)),
        "analytic",
        int(steps),
        device=device,
        proj_fun=proj_fun,
    )

    target_positions = [idx for idx, is_train in enumerate(train_mask[:real_len]) if is_train]
    with torch.no_grad():
        generated = proj_fun(sampling_fn(model))
        full_text = tokenizer.batch_decode(generated[:, :real_len])[0].strip()
        generated_target_ids = generated[0, target_positions]

    return {
        "prompt": segment_text(sample, train=False),
        "generated_full": full_text,
        "generated_completion": completion_from_full_text(full_text),
        "generated_target": tokenizer.decode(generated_target_ids).strip(),
    }


def write_report(path, records):
    lines = ["# Answer SFT Generation Examples", ""]
    lines.append("Structural anchors such as `Reasoning:` and `Answer:` may be fixed conditioning tokens; therefore the completion view is the main qualitative output.")
    lines.append("")
    for item in records:
        lines.append(f"## {item['id']}")
        lines.append("")
        lines.append(f"Mode: `{item.get('mode', '')}`")
        lines.append("")
        lines.append("### Fixed conditioning tokens")
        lines.append("```text")
        lines.append(item["prompt"])
        lines.append("```")
        lines.append("")
        lines.append("### Reference assistant completion")
        lines.append("```text")
        lines.append(item["reference_completion"].strip())
        lines.append("```")
        lines.append("")
        lines.append("### Reference generated-token target only")
        lines.append("```text")
        lines.append(item["reference_target"].strip())
        lines.append("```")
        lines.append("")
        for model_name, result in item["generations"].items():
            lines.append(f"### {model_name} completion")
            lines.append("```text")
            lines.append(result["completion"].strip())
            lines.append("```")
            lines.append("")
            lines.append(f"### {model_name} generated-token target only")
            lines.append("```text")
            lines.append(result["target"].strip())
            lines.append("```")
            lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate qualitative examples from pretrained/QA/QAR models.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset", default="QAR", choices=["QA", "QAR"])
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    rows = load_filtered_samples(
        config,
        args.dataset,
        "test",
        tokenizer,
        int(config["generation"].get("num_examples", 5)),
    )

    model_kinds = ["pretrained", "QA", "QAR"]
    loaded = {}
    for kind in model_kinds:
        model_tuple = load_model_tuple(config, kind if kind != "pretrained" else "pretrained", device)
        if model_tuple is not None:
            loaded[kind] = model_tuple

    records = []
    for idx, row in enumerate(rows):
        item = {
            "id": row.get("id", str(idx)),
            "mode": row.get("mode", args.dataset),
            "prompt": segment_text(row, train=False),
            "reference_target": segment_text(row, train=True),
            "reference_completion": assistant_completion(row),
            "answer": row.get("answer", ""),
            "reasoning": row.get("reasoning", ""),
            "generations": {},
        }
        for model_name, (model, graph, noise, _) in loaded.items():
            sample = sample_segment_infilling(
                model,
                graph,
                noise,
                tokenizer,
                row,
                int(config["generation"].get("max_length", 512)),
                int(config["generation"].get("steps", 128)),
                device,
            )
            item["generations"][model_name] = {
                "completion": sample["generated_completion"],
                "target": sample["generated_target"],
                "full": sample["generated_full"],
            }
        records.append(item)

    out_dir = Path(config["generation"].get("output_dir", SCRIPT_DIR / "reports"))
    write_report(out_dir / f"answer_generation_{args.dataset}.md", records)
    (out_dir / f"answer_generation_{args.dataset}.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out_dir / f'answer_generation_{args.dataset}.md'}")


if __name__ == "__main__":
    main()
