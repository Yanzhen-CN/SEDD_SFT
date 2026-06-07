import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "answer_config.yaml"


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def selected_runs(config):
    selected = config.get("run", {}).get("selected", "QA")
    if selected == "all":
        return ["QA", "QAR"]
    if isinstance(selected, list):
        return selected
    return [selected]


def maybe_prepare_data(config_path, config):
    if not config.get("data", {}).get("build", True):
        return
    cmd = [sys.executable, str(SCRIPT_DIR / "prepare_answer_data.py"), "--config", str(config_path)]
    subprocess.check_call(cmd, cwd=str(REPO_DIR))


def run_train(config_path, config, run_name):
    env = os.environ.copy()
    cuda = config.get("runs", {}).get(run_name, {}).get("cuda_visible_devices")
    if cuda is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda)
    cmd = [sys.executable, str(SCRIPT_DIR / "train_answer.py"), "--config", str(config_path), "--run", run_name]
    print(f"[{run_name}] {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(REPO_DIR), env=env)


def main():
    parser = argparse.ArgumentParser(description="One-command answer-conditioned SFT runner.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    maybe_prepare_data(config_path, config)

    if not config.get("run", {}).get("execute", True):
        print("run.execute=false, data preparation finished only.")
        return

    failures = []
    for run_name in selected_runs(config):
        code = run_train(config_path, config, run_name)
        if code != 0:
            failures.append((run_name, code))
    if failures:
        raise SystemExit(f"Run(s) failed: {failures}")


if __name__ == "__main__":
    main()
