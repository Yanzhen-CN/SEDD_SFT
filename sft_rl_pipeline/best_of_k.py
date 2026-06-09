import argparse
import json
from pathlib import Path

import torch
from transformers import GPT2TokenizerFast

from reward import score_answer
from rl_utils import DEFAULT_CONFIG, load_config, load_policy, prompt_until_assistant, read_jsonl, sample_answer, sample_segment_infilling, segment_text, set_seed


def main():
    parser = argparse.ArgumentParser(description="Reward-guided best-of-K sampling for QAR SFT policy.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config["training"].get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model, graph, noise, _, checkpoint = load_policy(config, device, checkpoint_path=args.checkpoint)

    bok = config["best_of_k"]
    data_path = Path(config["data"]["data_dir"]) / f"{bok.get('split', 'test')}.jsonl"
    samples = read_jsonl(data_path, limit=int(bok.get("num_examples", 20)))
    rows = []

    for idx, sample in enumerate(samples):
        prompt = prompt_until_assistant(sample)
        reference = segment_text(sample, train=True)
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
                answer = generated["generated_target"]
            else:
                answer = sample_answer(
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
            reward = score_answer(answer, reference, config.get("reward", {}))
            candidates.append({"k": k, "answer": answer, "reward": reward})
        best = max(candidates, key=lambda row: row["reward"]["score"])
        first = candidates[0]
        record = {
            "id": sample.get("id", str(idx)),
            "checkpoint": checkpoint,
            "prompt": prompt,
            "reference": reference,
            "first_reward": first["reward"]["score"],
            "best_reward": best["reward"]["score"],
            "reward_gain": best["reward"]["score"] - first["reward"]["score"],
            "first_answer": first["answer"],
            "best_answer": best["answer"],
            "candidates": candidates,
        }
        rows.append(record)
        print(f"{record['id']}: first={record['first_reward']:.3f}, best={record['best_reward']:.3f}, gain={record['reward_gain']:.3f}")

    out_dir = Path(config["results"].get("report_dir", "SFT_RL/reports"))
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.tag or Path(config["results"].get("output_dir", "SFT_RL/modelparameter/RL-QAR")).name
    json_path = out_dir / f"best_of_k_{run_name}.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# SFT-RL Best-of-K QAR", ""]
    if rows:
        avg_gain = sum(row["reward_gain"] for row in rows) / len(rows)
        md.append(f"Average reward gain over first sample: `{avg_gain:.4f}`")
        md.append("")
    for row in rows:
        md.extend([
            f"## {row['id']}",
            "",
            f"first reward: `{row['first_reward']:.4f}`",
            f"best reward: `{row['best_reward']:.4f}`",
            "",
            "### Prompt",
            "```text",
            row["prompt"],
            "```",
            "### First",
            "```text",
            row["first_answer"],
            "```",
            "### Best",
            "```text",
            row["best_answer"],
            "```",
            "",
        ])
    md_path = out_dir / f"best_of_k_{run_name}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
