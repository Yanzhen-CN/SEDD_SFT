import argparse
import subprocess
from pathlib import Path

import yaml

import data_process


PIPELINE_DIR = Path(__file__).resolve().parent
REPO_DIR = PIPELINE_DIR.parent


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def maybe_build_data(config):
    data_cfg = config.get("data", {})
    if not data_cfg.get("build", True):
        print("skip data build")
        return

    args = argparse.Namespace(
        valid_ratio=data_cfg.get("valid_ratio", 0.05),
        seed=data_cfg.get("seed", 42),
        arrow=data_cfg.get("arrow_path"),
    )
    data_process.build_datasets(args)


def build_train_command(config):
    run_cfg = config.get("run", {})
    sedd_cfg = config.get("sedd", {})

    dataset = run_cfg.get("dataset", "QA")
    data_dir = f"sft_pipeline/data/{dataset}"
    run_dir = f"exp_local/sft_{dataset}/${{now:%Y.%m.%d}}/${{now:%H%M%S}}"

    return [
        "python",
        "train.py",
        f"ngpus={sedd_cfg.get('ngpus', 1)}",
        f"model={sedd_cfg.get('model', 'small')}",
        f"model.length={run_cfg.get('length', 256)}",
        f"training.batch_size={run_cfg.get('batch_size', 2)}",
        f"eval.batch_size={run_cfg.get('batch_size', 2)}",
        f"training.n_iters={run_cfg.get('steps', 200)}",
        f"training.accum={sedd_cfg.get('accum', 1)}",
        f"training.snapshot_sampling={str(sedd_cfg.get('snapshot_sampling', False)).lower()}",
        f"eval.perplexity={str(sedd_cfg.get('perplexity', False)).lower()}",
        f"data.cache_dir={sedd_cfg.get('cache_dir', 'sft_pipeline/cache')}",
        f"data.train={data_dir}",
        f"data.valid={data_dir}",
        f"noise.type={sedd_cfg.get('noise', 'loglinear')}",
        f"graph.type={sedd_cfg.get('graph', 'absorb')}",
        f"hydra.run.dir={run_dir}",
    ]


def main():
    parser = argparse.ArgumentParser(description="Build QA/QAR data and launch SEDD SFT from one config.")
    parser.add_argument("--config", default=str(PIPELINE_DIR / "sft_config.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    maybe_build_data(config)

    command = build_train_command(config)
    print("official SEDD command:")
    print(" ".join(command))

    execute = config.get("run", {}).get("execute", True) and not args.dry_run
    if execute:
        subprocess.run(command, cwd=REPO_DIR, check=True)
    else:
        print("dry run only")


if __name__ == "__main__":
    main()
