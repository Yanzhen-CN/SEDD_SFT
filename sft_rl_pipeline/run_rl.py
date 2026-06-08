import argparse
import subprocess
import sys
from pathlib import Path

from rl_utils import DEFAULT_CONFIG


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent


def main():
    parser = argparse.ArgumentParser(description="Run SFT-RL exploration steps.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--best-of-k", action="store_true", help="Run reward-guided best-of-K sampling.")
    parser.add_argument("--train", action="store_true", help="Run reward-weighted DWDSE training.")
    args = parser.parse_args()

    run_all = not args.best_of_k and not args.train
    if run_all or args.best_of_k:
        subprocess.check_call([sys.executable, str(SCRIPT_DIR / "best_of_k.py"), "--config", args.config], cwd=str(REPO_DIR))
    if run_all or args.train:
        subprocess.check_call([sys.executable, str(SCRIPT_DIR / "train_reward_weighted.py"), "--config", args.config], cwd=str(REPO_DIR))


if __name__ == "__main__":
    main()
