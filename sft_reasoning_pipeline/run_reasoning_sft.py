#!/usr/bin/env python3
"""
Convenience launcher for the reasoning/QRA SFT workflow.

Default behavior follows sft_reasoning_pipeline/reasoning_config.yaml. The only common
runtime override is --gpu, which sets CUDA_VISIBLE_DEVICES for all subprocesses.

Workflow:
  1) prepare root S1K split/light data if data/S1K_light is missing
  2) prepare reasoning-pipeline QRA segment data
  3) train with the shared sft_answer_pipeline/train_answer.py using reasoning_config.yaml
  4) evaluate
  5) generate examples
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_DIR / "sft_reasoning_pipeline" / "reasoning_config.yaml"
S1K_LIGHT_MANIFEST = REPO_DIR / "data" / "S1K_light" / "manifest.json"


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_cmd(cmd: List[str], env: Dict[str, str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_DIR), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reasoning/QRA SFT workflow using config defaults.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpu", type=str, default=None, help="Set CUDA_VISIBLE_DEVICES, e.g. --gpu 1")
    parser.add_argument(
        "--prepare-split",
        choices=["auto", "always", "never"],
        default="auto",
        help="Whether to run root prepare_s1k_split.py before pipeline prepare.",
    )
    parser.add_argument("--skip-prepare", action="store_true", help="Skip reasoning-pipeline data preparation.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    cfg = load_yaml(config_path)
    selected = str(cfg.get("run", {}).get("selected", "QRA"))
    if selected == "RA":
        print("Warning: config run.selected is still RA. Current data adapter writes QRA. "
              "Please change reasoning_config.yaml to selected: QRA and runs.QRA.dataset: QRA.", flush=True)

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}", flush=True)

    py = sys.executable

    split_script = REPO_DIR / "prepare_s1k_split.py"
    need_split = args.prepare_split == "always" or (
        args.prepare_split == "auto" and not S1K_LIGHT_MANIFEST.exists()
    )
    if need_split:
        if not split_script.exists():
            raise FileNotFoundError(f"Missing {split_script}. Run the data patch first.")
        run_cmd([py, "-u", str(split_script), "--config", str(config_path)], env)
    else:
        print("Skipping root S1K split preparation.", flush=True)

    if not args.skip_prepare:
        run_cmd([py, "-u", "sft_reasoning_pipeline/prepare_reasoning_data.py", "--config", str(config_path)], env)

    if not args.skip_train:
        # The reasoning pipeline reuses the answer training loop; the config selects QRA.
        run_cmd([py, "-u", "sft_answer_pipeline/train_answer.py", "--config", str(config_path)], env)

    if not args.skip_eval:
        run_cmd([py, "-u", "sft_reasoning_pipeline/test_reasoning_eval.py", "--config", str(config_path)], env)

    if not args.skip_generate:
        run_cmd([py, "-u", "sft_reasoning_pipeline/generate_reasoning_examples.py", "--config", str(config_path)], env)

    print("\nReasoning/QRA SFT workflow finished.", flush=True)


if __name__ == "__main__":
    main()
