import argparse
import os
import subprocess
from pathlib import Path

import yaml

try:
    from . import data_process
except ImportError:
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


def resolve_selected_runs(config):
    runs = config.get("runs", {})
    selected = config.get("run", {}).get("selected", "all")

    if selected == "all":
        names = list(runs.keys())
    elif isinstance(selected, str):
        names = [selected]
    else:
        names = list(selected)

    missing = [name for name in names if name not in runs]
    if missing:
        raise ValueError(f"Unknown run name(s): {missing}. Available runs: {list(runs.keys())}")
    return names


def merged_run_config(config, name):
    run_cfg = dict(config.get("defaults", {}))
    run_cfg.update(config.get("runs", {})[name])
    run_cfg["name"] = name
    return run_cfg


def build_train_command(config, run_cfg):
    sedd_cfg = config.get("sedd", {})
    results_cfg = config.get("results", {})

    dataset = run_cfg.get("dataset", "QA")
    data_dir = f"sft_pipeline/data/{dataset}"
    run_name = run_cfg.get("name", dataset)
    run_dir = f"exp_local/sft_{run_name}/${{now:%Y.%m.%d}}/${{now:%H%M%S}}"
    model_name = run_cfg.get("model", sedd_cfg.get("model", "small"))
    pretrained_model = run_cfg.get("pretrained_model", sedd_cfg.get("pretrained_model"))
    best_root = results_cfg.get("best_dir")
    best_dir = f"{best_root}/{run_name}" if best_root else None

    command = [
        "python",
        "train.py",
        f"ngpus={sedd_cfg.get('ngpus', 1)}",
        f"model={model_name}",
        f"model.length={run_cfg.get('length', 256)}",
        f"training.batch_size={run_cfg.get('batch_size', 2)}",
        f"eval.batch_size={run_cfg.get('batch_size', 2)}",
        f"training.n_iters={run_cfg.get('steps', 200)}",
        f"training.log_freq={run_cfg.get('log_freq', 10)}",
        f"training.eval_freq={run_cfg.get('eval_freq', 50)}",
        f"training.snapshot_freq={run_cfg.get('snapshot_freq', run_cfg.get('steps', 200))}",
        f"training.snapshot_freq_for_preemption={run_cfg.get('snapshot_freq', run_cfg.get('steps', 200))}",
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
    if pretrained_model:
        command.append(f"+pretrained_model={pretrained_model}")
    if "save_best" in results_cfg:
        command.append(f"++results.save_best={str(results_cfg.get('save_best')).lower()}")
    if best_dir:
        command.append(f"++results.best_dir={best_dir}")
    command.append(f"++results.run_name={run_name}")
    return command


def run_one(command, run_cfg, dry_run):
    name = run_cfg.get("name", run_cfg.get("dataset", "run"))
    env = os.environ.copy()
    cuda_visible_devices = run_cfg.get("cuda_visible_devices")
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    print(f"\n[{name}] official SEDD command:")
    if cuda_visible_devices is not None:
        print(f"[{name}] CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
    print(" ".join(command))

    if dry_run:
        print(f"[{name}] dry run only")
        return None

    return subprocess.Popen(command, cwd=REPO_DIR, env=env)


def wait_processes(processes):
    failed = []
    for name, process in processes:
        code = process.wait()
        if code != 0:
            failed.append((name, code))
    if failed:
        raise SystemExit(f"Run(s) failed: {failed}")


def main():
    parser = argparse.ArgumentParser(description="Build QA/QAR data and launch SEDD SFT from one config.")
    parser.add_argument("--config", default=str(PIPELINE_DIR / "sft_config.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    maybe_build_data(config)

    selected_names = resolve_selected_runs(config)
    dry_run = args.dry_run or not config.get("run", {}).get("execute", True)
    parallel = config.get("run", {}).get("parallel", False)

    processes = []
    for name in selected_names:
        run_cfg = merged_run_config(config, name)
        command = build_train_command(config, run_cfg)
        process = run_one(command, run_cfg, dry_run)
        if process is not None:
            processes.append((name, process))
            if not parallel:
                wait_processes(processes)
                processes.clear()

    if processes:
        wait_processes(processes)


if __name__ == "__main__":
    main()
