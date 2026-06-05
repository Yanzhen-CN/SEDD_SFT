import argparse
import subprocess
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
REPO_DIR = PIPELINE_DIR.parent


def train(args):
    data_dir = f"sft_pipeline/data/{args.dataset}"
    run_dir = f"exp_local/sft_{args.dataset}/${{now:%Y.%m.%d}}/${{now:%H%M%S}}"
    command = [
        "python",
        "train.py",
        "ngpus=1",
        "model=small",
        f"model.length={args.length}",
        f"training.batch_size={args.batch_size}",
        f"eval.batch_size={args.batch_size}",
        f"training.n_iters={args.steps}",
        "training.accum=1",
        "training.snapshot_sampling=False",
        "eval.perplexity=False",
        "data.cache_dir=sft_pipeline/cache",
        f"data.train={data_dir}",
        f"data.valid={data_dir}",
        "noise.type=loglinear",
        "graph.type=absorb",
        f"hydra.run.dir={run_dir}",
    ]

    print("official SEDD command:")
    print(" ".join(command))
    if args.execute:
        subprocess.run(command, cwd=REPO_DIR, check=True)
    else:
        print("dry run only; add --execute to start training")


def main():
    parser = argparse.ArgumentParser(description="Training launcher for matched QA/QAR SEDD SFT runs.")
    parser.add_argument("--dataset", choices=["QA", "QAR"], default="QA")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
