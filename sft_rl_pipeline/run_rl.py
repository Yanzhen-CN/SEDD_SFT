import argparse
import os
import subprocess
import sys
from pathlib import Path

from rl_utils import DEFAULT_CONFIG

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent


def run_cmd(cmd, env=None):
    print("$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.check_call([str(x) for x in cmd], cwd=str(REPO_DIR), env=env)


def main():
    parser = argparse.ArgumentParser(description="Run SFT-RL reward-weighted continuation / best-of-K steps.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--gpu", default=None, help="Optional CUDA_VISIBLE_DEVICES override, e.g. --gpu 2")
    parser.add_argument("--best-of-k", action="store_true", help="Run reward-guided best-of-K sampling.")
    parser.add_argument("--reward-eval", action="store_true", help="Run generation-level reward evaluation for the selected checkpoint.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint for reward-eval or best-of-k.")
    parser.add_argument("--tag", default=None, help="Optional tag for reward-eval or best-of-k reports.")
    parser.add_argument("--train", action="store_true", help="Run reward-weighted DWDSE training.")
    parser.add_argument("--run-name", default="", help="Optional suffix for the training run directory.")
    args = parser.parse_args()

    env = os.environ.copy()
    if args.gpu is not None and str(args.gpu) != "":
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"[rl] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}", flush=True)

    run_all = not args.best_of_k and not args.reward_eval and not args.train
    if run_all or args.best_of_k:
        cmd = [sys.executable, SCRIPT_DIR / "best_of_k.py", "--config", args.config]
        if args.checkpoint:
            cmd.extend(["--checkpoint", args.checkpoint])
        if args.tag:
            cmd.extend(["--tag", args.tag])
        run_cmd(cmd, env=env)
    if args.reward_eval:
        cmd = [sys.executable, SCRIPT_DIR / "evaluate_generation_reward.py", "--config", args.config]
        if args.checkpoint:
            cmd.extend(["--checkpoint", args.checkpoint])
        if args.tag:
            cmd.extend(["--tag", args.tag])
        run_cmd(cmd, env=env)
    if run_all or args.train:
        cmd = [sys.executable, SCRIPT_DIR / "train_reward_weighted.py", "--config", args.config]
        if args.run_name:
            cmd.extend(["--run-name", args.run_name])
        run_cmd(cmd, env=env)


if __name__ == "__main__":
    main()
