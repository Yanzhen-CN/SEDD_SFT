import argparse
import json
from pathlib import Path

import torch
from transformers import GPT2TokenizerFast

from reward import score_answer
from generation_schema import make_generation_record, split_sections, write_generation_markdown


def numeric_reward_components(reward):
    skip = {"score", "base_score", "fatal_multiplier", "active_weight_sum", "chars", "reasoning_chars", "answer_chars", "reasoning_tokens", "answer_tokens"}
    return {k: float(v) for k, v in reward.items() if k not in skip and isinstance(v, (int, float))}


def mean_dict(rows):
    sums = {}
    counts = {}
    for row in rows:
        for k, v in row.items():
            if isinstance(v, (int, float)):
                sums[k] = sums.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / max(1, counts[k]) for k in sorted(sums)}

from rl_utils import (
    DEFAULT_CONFIG,
    assistant_completion,
    load_config,
    load_filtered_samples,
    load_policy,
    prompt_until_assistant,
    sample_answer,
    sample_segment_infilling,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser(description="Reward-guided best-of-K sampling for anchored QAR SFT policy.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["training"].get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model, graph, noise, _, checkpoint = load_policy(config, device, checkpoint_path=args.checkpoint)
    bok = config["best_of_k"]
    split = bok.get("split", "test")
    samples = load_filtered_samples(config, split, tokenizer, limit=int(bok.get("num_examples", 20)))

    rows = []
    for idx, sample in enumerate(samples):
        prompt = prompt_until_assistant(sample)
        reference_completion = assistant_completion(sample)
        candidates = []
        for k in range(int(bok.get("k", 4))):
            if bok.get("generation_mode", "segment_infilling") == "segment_infilling":
                generated = sample_segment_infilling(
                    model,
                    graph,
                    noise,
                    tokenizer,
                    sample,
                    int(config["model"].get("max_length", 512)),
                    int(config["model"].get("min_target_tokens", 32)),
                    int(bok.get("steps", 128)),
                    device,
                )
                # Score the assistant completion, not generated_target.  In anchored
                # QAR, generated_target omits fixed labels such as Reasoning:/Answer:.
                score_text = generated["generated_completion"]
                display_answer = generated["generated_completion"]
                sections = generated.get("generated_sections") or split_sections(display_answer)
                target_only = generated["generated_target"]
            else:
                score_text = sample_answer(
                    model,
                    graph,
                    noise,
                    tokenizer,
                    prompt,
                    int(config["model"].get("max_length", 512)),
                    int(bok.get("answer_token_budget", 256)),
                    int(bok.get("steps", 128)),
                    device,
                )
                display_answer = score_text
                sections = split_sections(display_answer)
                target_only = score_text
            reward = score_answer(score_text, reference_completion, config.get("reward", {}))
            candidates.append(
                {
                    "k": k,
                    "answer": score_text,
                    "target_only": target_only,
                    "display_answer": display_answer,
                    "sections": sections,
                    "reward": reward,
                }
            )

        best = max(candidates, key=lambda row: row["reward"]["score"])
        first = candidates[0]
        record = make_generation_record(
            sample,
            sample.get("mode", "QAR"),
            split,
            Path(config["data"]["data_dir"]) / f"{split}.jsonl",
            {
                "first": first["sections"],
                "best": best["sections"],
            },
        )
        record["checkpoint"] = checkpoint
        record["generation_metrics"] = {
            "first_reward": first["reward"]["score"],
            "best_reward": best["reward"]["score"],
            "reward_gain": best["reward"]["score"] - first["reward"]["score"],
            "first_components": numeric_reward_components(first["reward"]),
            "best_components": numeric_reward_components(best["reward"]),
        }
        record["candidates"] = candidates
        rows.append(record)
        print(
            f"{record['id']}: first={record['generation_metrics']['first_reward']:.3f}, "
            f"best={record['generation_metrics']['best_reward']:.3f}, gain={record['generation_metrics']['reward_gain']:.3f}"
        )

    out_dir = Path(config["results"].get("report_dir", "sft_rl_pipeline/reports"))
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.tag or Path(config["results"].get("output_dir", "sft_rl_pipeline/modelparameter/RL-QAR")).name
    json_path = out_dir / f"best_of_k_{run_name}.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "num_examples": len(rows),
        "mean_first_reward": sum(row["generation_metrics"]["first_reward"] for row in rows) / max(1, len(rows)),
        "mean_best_reward": sum(row["generation_metrics"]["best_reward"] for row in rows) / max(1, len(rows)),
        "mean_reward_gain": sum(row["generation_metrics"]["reward_gain"] for row in rows) / max(1, len(rows)),
        "mean_first_components": mean_dict([row["generation_metrics"].get("first_components", {}) for row in rows]),
        "mean_best_components": mean_dict([row["generation_metrics"].get("best_components", {}) for row in rows]),
    }
    summary_path = out_dir / f"best_of_k_{run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# SFT-RL Best-of-K Anchored QAR",
        "",
        "JSON records use the common generation schema. Reward summaries are below.",
        "",
    ]
    if rows:
        avg_gain = sum(row["generation_metrics"]["reward_gain"] for row in rows) / len(rows)
        md.append(f"Average reward gain over first sample: `{avg_gain:.4f}`")
        md.append(f"Mean first reward: `{summary['mean_first_reward']:.4f}`")
        md.append(f"Mean best reward: `{summary['mean_best_reward']:.4f}`")
        md.append("")
        md.append("### Mean component scores")
        md.append("```json")
        md.append(json.dumps({"first": summary["mean_first_components"], "best": summary["mean_best_components"]}, ensure_ascii=False, indent=2))
        md.append("```")
        md.append("")
    md_path = out_dir / f"best_of_k_{run_name}.md"
    md_path.write_text("\n".join(md) + "\n\n", encoding="utf-8")
    write_generation_markdown(out_dir / f"best_of_k_{run_name}_examples.md", "SFT-RL Best-of-K Examples", rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
