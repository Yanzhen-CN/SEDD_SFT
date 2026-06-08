import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import GPT2TokenizerFast

from rl_utils import load_config, load_policy


DEFAULT_ITEMS = [
    ("pretrained", "sft_rl_pipeline/rl_config_pretrained.yaml", "pretrained.pth"),
    ("best_QAR", "sft_rl_pipeline/rl_config.yaml", "SFT_RL/modelparameter/best-QAR/best_QAR.pth"),
    ("pretrain_rl", "sft_rl_pipeline/rl_config_pretrained.yaml", "SFT_RL/modelparameter/pretrain_rl/best_RL_QAR.pth"),
    ("sft_rl", "sft_rl_pipeline/rl_config.yaml", "SFT_RL/modelparameter/sft_rl/best_RL_QAR.pth"),
]


def dump_results(rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "rl_test_results.json"
    csv_path = out_dir / "rl_test_results.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "checkpoint", "test_loss", "batches", "config"])
        writer.writeheader()
        writer.writerows(rows)
    print(json_path)
    print(csv_path)


def evaluate_item(name, config_path, checkpoint_path, eval_batches):
    from answer_dataset import make_answer_loader
    from answer_losses import evaluate_answer_loss

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    data_dir = Path(config["data"]["data_dir"])
    loader = make_answer_loader(
        data_dir / "test.jsonl",
        tokenizer,
        int(config["model"].get("max_length", 512)),
        int(config["model"].get("min_target_tokens", 32)),
        1,
        False,
        int(config["training"].get("num_workers", 0)),
    )
    model, graph, noise, ema, loaded_checkpoint = load_policy(config, device, checkpoint_path=checkpoint_path)
    test_loss, batches = evaluate_answer_loss(model, ema, noise, graph, loader, device, eval_batches)
    row = {
        "name": name,
        "checkpoint": loaded_checkpoint,
        "test_loss": test_loss,
        "batches": batches,
        "config": str(config_path),
    }
    print(f"{name}: test_loss={test_loss:.6f} batches={batches} checkpoint={loaded_checkpoint}")
    return row


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL experiment checkpoints on the QAR test set.")
    parser.add_argument("--eval-batches", type=int, default=0)
    parser.add_argument("--output-dir", default="SFT_RL/test_results")
    args = parser.parse_args()

    rows = []
    for name, config_path, checkpoint_path in DEFAULT_ITEMS:
        if not Path(checkpoint_path).exists():
            print(f"skip {name}: checkpoint not found: {checkpoint_path}")
            continue
        rows.append(evaluate_item(name, config_path, checkpoint_path, args.eval_batches))
    dump_results(rows, Path(args.output_dir))


if __name__ == "__main__":
    main()
