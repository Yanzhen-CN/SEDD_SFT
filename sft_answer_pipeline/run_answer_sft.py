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
        return list(config.get("runs", {}).keys()) or ["QA", "QAR"]
    if isinstance(selected, list):
        return selected
    return [selected]


def run_cmd(cmd, env=None):
    print("$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.check_call([str(x) for x in cmd], cwd=str(REPO_DIR), env=env)


def maybe_prepare_split(config_path, config, force=False):
    data_cfg = config.get("data", {})
    light_dir = REPO_DIR / data_cfg.get("light_output_dir", "data/S1K_light")
    if force or not (light_dir / "manifest.json").exists():
        run_cmd([sys.executable, REPO_DIR / "prepare_s1k_split.py", "--config", config_path])
    else:
        print(f"[answer] reuse existing {light_dir / 'manifest.json'}", flush=True)


def prepare_answer_data(config_path):
    run_cmd([sys.executable, SCRIPT_DIR / "prepare_answer_data.py", "--config", config_path])


def env_for_run(config, run_name, gpu_override=None):
    env = os.environ.copy()
    cuda = gpu_override
    if cuda is None:
        cuda = config.get("runs", {}).get(run_name, {}).get("cuda_visible_devices")
    if cuda is not None and str(cuda) != "":
        env["CUDA_VISIBLE_DEVICES"] = str(cuda)
        print(f"[{run_name}] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}", flush=True)
    return env


def train_run(config_path, config, run_name, gpu_override=None):
    env = env_for_run(config, run_name, gpu_override)
    run_cmd([sys.executable, SCRIPT_DIR / "train_answer.py", "--config", config_path, "--run", run_name], env=env)


def maybe_eval(config_path, config):
    if config.get("run", {}).get("evaluate_after_train", False):
        run_cmd([sys.executable, SCRIPT_DIR / "test_answer_eval.py", "--config", config_path])


def maybe_generate(config_path, config, runs):
    if not config.get("run", {}).get("generate_after_train", False):
        return
    dataset = config.get("generation", {}).get("dataset")
    if dataset:
        run_cmd([sys.executable, SCRIPT_DIR / "generate_answer_examples.py", "--config", config_path, "--dataset", dataset])
        return
    seen = set()
    for run_name in runs:
        dataset_name = config.get("runs", {}).get(run_name, {}).get("dataset", run_name)
        if dataset_name in seen:
            continue
        seen.add(dataset_name)
        run_cmd([sys.executable, SCRIPT_DIR / "generate_answer_examples.py", "--config", config_path, "--dataset", dataset_name])


def maybe_visual(config):
    if config.get("run", {}).get("visual_after_train", False) and (SCRIPT_DIR / "visual_answer.py").exists():
        run_cmd([sys.executable, SCRIPT_DIR / "visual_answer.py"])


def main():
    parser = argparse.ArgumentParser(description="One-command anchored QA/QAR/QRA SEDD SFT runner.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--gpu", default=None, help="Optional CUDA_VISIBLE_DEVICES override, e.g. --gpu 0")
    parser.add_argument("--force-split", action="store_true", help="Regenerate data/S1K and data/S1K_light before pipeline prepare.")
    parser.add_argument("--skip-split", action="store_true", help="Skip root prepare_s1k_split.py even if S1K_light is missing.")
    parser.add_argument("--prepare-only", action="store_true", help="Only build data; do not train/eval/generate.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    if config.get("data", {}).get("build", True):
        if not args.skip_split:
            maybe_prepare_split(config_path, config, force=args.force_split)
        prepare_answer_data(config_path)

    if args.prepare_only or not config.get("run", {}).get("execute", True):
        print("[answer] data preparation finished; training skipped.", flush=True)
        return

    runs = selected_runs(config)
    failures = []
    for run_name in runs:
        try:
            train_run(config_path, config, run_name, gpu_override=args.gpu)
        except subprocess.CalledProcessError as exc:
            failures.append((run_name, exc.returncode))
            break

    if failures:
        raise SystemExit(f"Run(s) failed: {failures}")

    maybe_eval(config_path, config)
    maybe_generate(config_path, config, runs)
    maybe_visual(config)


if __name__ == "__main__":
    main()
