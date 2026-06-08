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


def read_jsonl(path, limit):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) >= limit:
                    break
    return rows


def segment_text(segments, train=None):
    selected = segments
    if train is not None:
        selected = [segment for segment in segments if bool(segment["train"]) is train]
    return "".join(segment["text"] for segment in selected)


def prompt_until_assistant(segments):
    parts = []
    for segment in segments:
        parts.append(segment["text"])
        if segment.get("name") == "assistant_label":
            break
    return "".join(parts)


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


def sample_answer(model, graph, noise, tokenizer, prompt, length, answer_budget, steps, device):
    import sampling

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    max_prompt = max(1, length - answer_budget)
    if len(prompt_ids) > max_prompt:
        prompt_ids = prompt_ids[-max_prompt:]
    input_ids = torch.tensor(prompt_ids, device=device)[None]
    input_locs = list(range(len(prompt_ids)))

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
    full_text = tokenizer.batch_decode(sample)[0]
    prompt_text = tokenizer.decode(prompt_ids)
    answer_text = tokenizer.batch_decode(sample[:, len(prompt_ids):])[0]
    return {
        "prompt": prompt_text,
        "generated_full": full_text,
        "generated_answer": answer_text.strip(),
    }


def write_report(path, records):
    lines = ["# Answer SFT Generation Examples", ""]
    for item in records:
        lines.append(f"## {item['id']}")
        lines.append("")
        lines.append("### Prompt")
        lines.append("```text")
        lines.append(item["prompt"])
        lines.append("```")
        lines.append("")
        lines.append("### Reference")
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
    parser = argparse.ArgumentParser(description="Generate qualitative answer examples from pretrained/QA/QAR models.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset", default="QA", choices=["QA", "QAR"])
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    rows = read_jsonl(
        Path(config["data"].get("output_dir", SCRIPT_DIR / "data")) / args.dataset / "test.jsonl",
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
            "id": str(idx),
            "prompt": prompt_until_assistant(row),
            "reference": segment_text(row, train=True),
            "generations": {},
        }
        for model_name, (model, graph, noise, _) in loaded.items():
            sample = sample_answer(
                model,
                graph,
                noise,
                tokenizer,
                item["prompt"],
                int(config["generation"].get("max_length", 512)),
                int(config["generation"].get("answer_token_budget", 256)),
                int(config["generation"].get("steps", 128)),
                device,
            )
            item["generations"][model_name] = sample["generated_answer"]
        records.append(item)

    out_dir = Path(config["generation"].get("output_dir", SCRIPT_DIR / "reports"))
    write_report(out_dir / f"answer_generation_{args.dataset}.md", records)
    (out_dir / f"answer_generation_{args.dataset}.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {out_dir / f'answer_generation_{args.dataset}.md'}")


if __name__ == "__main__":
    main()
