import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "rl_config.yaml"
sys.path.insert(0, str(REPO_DIR))


def load_config(path=DEFAULT_CONFIG):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


SEGMENT_ORDER = ["user_label", "user", "assistant_label", "assistant", "reasoning_label", "reasoning"]


def ordered_segments(sample):
    segments = sample["segments"]
    return [(name, segments[name]) for name in SEGMENT_ORDER if name in segments]


def segment_text(sample, train=None):
    selected = ordered_segments(sample)
    if train is not None:
        selected = [(name, seg) for name, seg in selected if bool(seg["train"]) is train]
    return "".join(seg["text"] for _, seg in selected)


def prompt_until_assistant(sample):
    parts = []
    for name, seg in ordered_segments(sample):
        parts.append(seg["text"])
        if name == "assistant_label":
            break
    return "".join(parts)


def load_policy(config, device, checkpoint_path=None):
    import graph_lib
    import noise_lib
    from model import SEDD
    from model.ema import ExponentialMovingAverage

    model = SEDD.from_pretrained(config["model"].get("pretrained", "louaaron/sedd-medium")).to(device)
    model.config.model.length = int(config["model"].get("max_length", model.config.model.length))
    ema = ExponentialMovingAverage(model.parameters(), decay=float(config["training"].get("ema", 0.9999)))

    ckpt_value = checkpoint_path if checkpoint_path is not None else config["model"].get("init_checkpoint", "")
    ckpt = Path(ckpt_value) if ckpt_value else None
    if ckpt and ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"], strict=True)
        if "ema" in state:
            ema.load_state_dict(state["ema"])
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
        print(f"Loaded checkpoint: {ckpt}")
    else:
        if ckpt:
            print(f"Warning: checkpoint not found, using pretrained only: {ckpt}")
        else:
            print("Using pretrained only; no local checkpoint was requested.")

    model.eval()
    graph = graph_lib.get_graph(model.config, device)
    noise = noise_lib.get_noise(model.config).to(device)
    return model, graph, noise, ema, str(ckpt) if ckpt else config["model"].get("pretrained", "louaaron/sedd-medium")


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
    return tokenizer.batch_decode(sample[:, len(prompt_ids):])[0].strip()
