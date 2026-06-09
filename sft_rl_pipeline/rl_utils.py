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
sys.path.insert(0, str(REPO_DIR / "sft_answer_pipeline"))


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


def ordered_segments(sample):
    """Use the shared SFT segment order.

    This is crucial for v2 QAR samples, where the assistant completion is:
        Reasoning: ...\n\nAnswer: ...
    and the per-sample `segment_order` includes answer_label/answer.
    """
    from answer_dataset import ordered_segments as _ordered_segments

    return _ordered_segments(sample)


def segment_text(sample, train=None):
    from answer_dataset import sample_text

    return sample_text(sample, train=train)


def prompt_until_assistant(sample):
    parts = []
    for name, seg in ordered_segments(sample):
        parts.append(seg.get("text", ""))
        if name == "assistant_label":
            break
    return "".join(parts)


def load_filtered_samples(config, split, tokenizer, limit=None):
    """Load the same filtered data that training/eval will see.

    We intentionally do not read raw JSONL directly for best-of-K, because raw
    QAR can contain over-length examples.  The dataset loader drops them and
    writes a report, matching training behavior.
    """
    from answer_dataset import AnswerSegmentDataset

    data_dir = Path(config["data"]["data_dir"])
    dataset = AnswerSegmentDataset(
        data_dir / f"{split}.jsonl",
        tokenizer,
        int(config["model"].get("max_length", 512)),
        min_target_tokens=int(config["model"].get("min_target_tokens", 32)),
        drop_overlength=bool(config["model"].get("drop_overlength", True)),
        write_report=bool(config["model"].get("write_load_reports", True)),
    )
    samples = dataset.samples
    return samples[:limit] if limit else samples


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
    """Fallback prompt-only generation. Segment infilling is preferred."""
    import sampling

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    max_prompt = max(1, int(length) - int(answer_budget))
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
        (1, int(length)),
        "analytic",
        int(steps),
        device=device,
        proj_fun=proj_fun,
    )
    with torch.no_grad():
        sample = proj_fun(sampling_fn(model))
    return tokenizer.batch_decode(sample[:, len(prompt_ids):])[0].strip()


def encode_sample_no_truncation(sample, tokenizer, max_length):
    ids = []
    train_mask = []
    segment_token_lens = {}
    for name, seg in ordered_segments(sample):
        token_ids = tokenizer(seg.get("text", ""), add_special_tokens=False).input_ids
        is_train = bool(seg.get("train", False))
        ids.extend(token_ids)
        train_mask.extend([1 if is_train else 0] * len(token_ids))
        segment_token_lens[name] = len(token_ids)

    if len(ids) > int(max_length):
        raise ValueError(
            f"Filtered RL sample {sample.get('id')} is over-length: {len(ids)} > {max_length}. "
            "Regenerate/copy filtered QAR data or increase model.max_length."
        )
    if not any(train_mask):
        raise ValueError(f"RL sample {sample.get('id')} has no train target tokens.")
    return ids, train_mask, segment_token_lens


def sample_segment_infilling(model, graph, noise, tokenizer, sample, length, min_target_tokens, steps, device):
    """Generate only train=True tokens while clamping condition tokens.

    No truncation is performed.  This keeps RL generation aligned with the
    training dataset and avoids silently dropping the Answer section.
    """
    import sampling

    ids, train_mask, segment_token_lens = encode_sample_no_truncation(sample, tokenizer, int(length))
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
        generated_target = tokenizer.decode(generated[0, target_positions])
        full_text = tokenizer.batch_decode(generated[:, :real_len])[0]

    return {
        "prompt": segment_text(sample, train=False),
        "reference_target": segment_text(sample, train=True),
        "generated": full_text.strip(),
        "generated_target": generated_target.strip(),
        "segment_token_lens": segment_token_lens,
    }
