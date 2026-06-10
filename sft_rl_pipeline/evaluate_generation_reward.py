"""Evaluate generation-level reward for a checkpoint.

This complements loss evaluation.  Loss measures exact target reconstruction under
DWDSE, while generation reward measures whether sampled anchored-QAR completions
have useful reasoning/answer contents under the current rule reward.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import GPT2TokenizerFast

from generation_schema import make_generation_record, write_generation_markdown
from reward import score_answer
from rl_utils import (
    DEFAULT_CONFIG,
    assistant_completion,
    load_config,
    load_filtered_samples,
    load_policy,
    sample_segment_infilling,
    set_seed,
)


def numeric_components(reward):
    skip = {"score", "base_score", "fatal_multiplier", "active_weight_sum", "chars", "reasoning_chars", "answer_chars", "reasoning_tokens", "answer_tokens"}
    return {k: float(v) for k, v in reward.items() if isinstance(v, (int, float)) and k not in skip}


def mean_dict(dicts):
    sums, counts = {}, {}
    for d in dicts:
        for k, v in d.items():
            sums[k] = sums.get(k, 0.0) + float(v)
            counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / max(1, counts[k]) for k in sorted(sums)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated QAR reward for one checkpoint.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["training"].get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model, graph, noise, _, checkpoint = load_policy(config, device, checkpoint_path=args.checkpoint)
    bok = config.get("best_of_k", {})
    split = args.split or bok.get("split", "test")
    num_examples = args.num_examples if args.num_examples is not None else int(bok.get("num_examples", 20))
    steps = args.steps if args.steps is not None else int(bok.get("steps", 128))
    samples = load_filtered_samples(config, split, tokenizer, limit=num_examples)

    rows = []
    components = []
    for idx, sample in enumerate(samples):
        generated = sample_segment_infilling(
            model,
            graph,
            noise,
            tokenizer,
            sample,
            int(config["model"].get("max_length", 512)),
            int(config["model"].get("min_target_tokens", 32)),
            int(steps),
            device,
        )
        reference_completion = assistant_completion(sample)
        reward = score_answer(generated["generated_completion"], reference_completion, config.get("reward", {}))
        comp = numeric_components(reward)
        components.append(comp)
        row = make_generation_record(
            sample,
            sample.get("mode", "QAR"),
            split,
            Path(config["data"]["data_dir"]) / f"{split}.jsonl",
            {"generated": generated.get("generated_sections") or {"reasoning": "", "answer": generated["generated_completion"]}},
        )
        row["checkpoint"] = checkpoint
        row["generation_metrics"] = {
            "score": reward["score"],
            "base_score": reward.get("base_score", reward["score"]),
            "fatal_multiplier": reward.get("fatal_multiplier", 1.0),
            "components": comp,
        }
        rows.append(row)
        print(f"{row['id']}: reward={row['generation_metrics']['score']:.4f} fatal={row['generation_metrics']['fatal_multiplier']:.2f}")

    summary = {
        "checkpoint": checkpoint,
        "config": args.config,
        "split": split,
        "num_examples": len(rows),
        "steps": steps,
        "mean_reward": sum(r["generation_metrics"]["score"] for r in rows) / max(1, len(rows)),
        "mean_base_score": sum(r["generation_metrics"]["base_score"] for r in rows) / max(1, len(rows)),
        "mean_fatal_multiplier": sum(r["generation_metrics"]["fatal_multiplier"] for r in rows) / max(1, len(rows)),
        "mean_components": mean_dict(components),
    }

    out_dir = Path(config["results"].get("report_dir", "sft_rl_pipeline/reports"))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or Path(str(checkpoint)).stem.replace(".", "_")
    json_path = out_dir / f"generation_reward_{tag}.json"
    md_path = out_dir / f"generation_reward_{tag}.md"
    summary_path = out_dir / f"generation_reward_{tag}_summary.json"
    csv_path = out_dir / f"generation_reward_{tag}_summary.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_generation_markdown(md_path, "Generation Reward Examples", rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["checkpoint", "config", "split", "num_examples", "steps", "mean_reward", "mean_base_score", "mean_fatal_multiplier"] + [f"component_{k}" for k in summary["mean_components"]]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        flat = {k: v for k, v in summary.items() if k != "mean_components"}
        flat.update({f"component_{k}": v for k, v in summary["mean_components"].items()})
        writer.writerow(flat)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
