import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader


REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
DEFAULT_CONFIG = Path(__file__).resolve().parent / "sft_config.yaml"


def load_json(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_test_config(path):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config.get("test_eval", {})


def choose(cli_value, config, key, default):
    return cli_value if cli_value is not None else config.get(key, default)


def dump_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def can_load_best(model_dir):
    model_dir = Path(model_dir)
    return (model_dir / "best_eval.json").exists() and (model_dir / "best.pth").exists()


def get_reference_cfg(output_root):
    import utils

    output_root = Path(output_root)
    for dirname in ["QA", "QAR"]:
        best_eval_path = output_root / dirname / "best_eval.json"
        if not best_eval_path.exists():
            continue

        best_eval = load_json(best_eval_path)
        source_run_dir = best_eval.get("source_run_dir")
        if source_run_dir:
            return utils.load_hydra_config_from_run(source_run_dir)

    raise FileNotFoundError(
        f"No reference config found under {output_root / 'QA'} or {output_root / 'QAR'}"
    )


def load_best_model(model_dir, device):
    import graph_lib
    import noise_lib
    import utils
    from model import SEDD
    from model.ema import ExponentialMovingAverage

    model_dir = Path(model_dir)
    best_eval_path = model_dir / "best_eval.json"
    ckpt_path = model_dir / "best.pth"

    if not best_eval_path.exists():
        raise FileNotFoundError(f"Missing {best_eval_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing {ckpt_path}")

    best_eval = load_json(best_eval_path)
    source_run_dir = best_eval.get("source_run_dir")
    if not source_run_dir:
        raise KeyError(f"{best_eval_path} does not contain source_run_dir")

    cfg = utils.load_hydra_config_from_run(source_run_dir)

    model = SEDD(cfg).to(device)
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg.training.ema)

    loaded_state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(loaded_state["model"], strict=True)

    if "ema" in loaded_state:
        ema.load_state_dict(loaded_state["ema"])

    model.eval()

    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    if hasattr(noise, "eval"):
        noise.eval()

    return model, graph, noise, cfg, str(ckpt_path), ema


def load_pretrained_model(model_name, reference_cfg, device):
    import graph_lib
    import noise_lib
    from model import SEDD
    from model.ema import ExponentialMovingAverage

    model = SEDD.from_pretrained(model_name).to(device)
    model.eval()

    graph = graph_lib.get_graph(reference_cfg, device)
    noise = noise_lib.get_noise(reference_cfg).to(device)
    if hasattr(noise, "eval"):
        noise.eval()

    ema = ExponentialMovingAverage(model.parameters(), decay=reference_cfg.training.ema)

    return model, graph, noise, reference_cfg, model_name, ema


def evaluate(
    model,
    graph,
    noise,
    cfg,
    dataset_path,
    split,
    eval_batches,
    batch_size,
    device,
    ema,
    num_proc,
):
    import data
    import losses

    dataset = data.get_dataset(
        dataset_path,
        split,
        cache_dir=cfg.data.cache_dir,
        block_size=cfg.model.length,
        num_proc=num_proc,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    eval_step_fn = losses.get_step_fn(
        noise=noise,
        graph=graph,
        train=False,
        optimize_fn=None,
        accum=cfg.training.accum,
    )

    state = {
        "model": model,
        "ema": ema,
    }

    model.eval()
    if hasattr(noise, "eval"):
        noise.eval()

    total = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(loader):
            if eval_batches > 0 and batch_idx >= eval_batches:
                break

            batch = batch_dict["input_ids"].to(device, non_blocking=True)
            loss = eval_step_fn(state, batch)

            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                raise ValueError(
                    f"Non-finite loss: dataset={dataset_path}, split={split}, "
                    f"batch={batch_idx}, loss={loss_value}"
                )

            total += loss_value
            count += 1

    if count == 0:
        raise ValueError(f"No batches found for {dataset_path}, split={split}")

    return total / count, count


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["dataset", "model", "loss", "eval_batches", "checkpoint"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(rows, output_root):
    output_root = Path(output_root)

    result_dir = output_root / "test_result"
    result_dir.mkdir(parents=True, exist_ok=True)

    write_csv(rows, result_dir / "test_results.csv")
    dump_json(rows, result_dir / "test_results.json")

    for model_name in ["QA-best", "QAR-best"]:
        model_rows = [row for row in rows if row["model"] == model_name]
        if not model_rows:
            continue

        target_dir = output_root / model_name.replace("-best", "")
        target_dir.mkdir(parents=True, exist_ok=True)

        dump_json(model_rows, target_dir / "best_test_result.json")
        write_csv(model_rows, target_dir / "best_test_result.csv")

    pretrained_rows = [row for row in rows if row["model"] == "pretrained"]
    if pretrained_rows:
        pretrained_dir = output_root / "pretrained" / "medium"
        pretrained_dir.mkdir(parents=True, exist_ok=True)

        dump_json(pretrained_rows, pretrained_dir / "test_result.json")
        write_csv(pretrained_rows, pretrained_dir / "test_result.csv")

    print(f"Wrote {result_dir / 'test_results.csv'}")
    print(f"Wrote {result_dir / 'test_results.json'}")


def build_row(dataset_name, model_name, loss, used_batches, checkpoint):
    return {
        "dataset": dataset_name,
        "model": model_name,
        "loss": loss,
        "eval_batches": used_batches,
        "checkpoint": checkpoint,
    }


def evaluate_loaded_model(
    model_name,
    model,
    graph,
    noise,
    cfg,
    checkpoint,
    ema,
    datasets,
    eval_batches,
    batch_size,
    device,
    num_proc,
):
    rows = []

    for dataset_name, dataset_path in datasets:
        print(f"Evaluating model={model_name}, dataset={dataset_name}")

        loss, used_batches = evaluate(
            model=model,
            graph=graph,
            noise=noise,
            cfg=cfg,
            dataset_path=dataset_path,
            split="test",
            eval_batches=eval_batches,
            batch_size=batch_size,
            device=device,
            ema=ema,
            num_proc=num_proc,
        )

        row = build_row(
            dataset_name=dataset_name,
            model_name=model_name,
            loss=loss,
            used_batches=used_batches,
            checkpoint=checkpoint,
        )

        rows.append(row)
        print(row)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained, QA-best, and QAR-best on QA/QAR test sets."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--qa-data", default=None)
    parser.add_argument("--qar-data", default=None)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--eval-batches", type=int, default=None, help="0 means evaluate the full test split.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-proc", type=int, default=None, help="Number of processes used by the HF dataset map step.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_test_config(args.config)
    output_root = Path(choose(args.output_root, config, "output_root", "sft_pipeline/modelparameter"))
    qa_data = choose(args.qa_data, config, "qa_data", "sft_pipeline/data/QA")
    qar_data = choose(args.qar_data, config, "qar_data", "sft_pipeline/data/QAR")
    pretrained = choose(args.pretrained, config, "pretrained", "louaaron/sedd-medium")
    eval_batches = int(choose(args.eval_batches, config, "eval_batches", 0))
    batch_size = int(choose(args.batch_size, config, "batch_size", 1))
    num_proc = int(choose(args.num_proc, config, "num_proc", 8))
    seed = int(choose(args.seed, config, "seed", 42))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = [
        ("QA-test", qa_data),
        ("QAR-test", qar_data),
    ]

    print(f"Device: {device}")
    print(f"Output root: {output_root}")
    print(f"Eval batches: {eval_batches}")
    print(f"Batch size: {batch_size}")
    print(f"Num proc: {num_proc}")

    rows = []

    reference_cfg = get_reference_cfg(output_root)

    model = graph = noise = cfg = checkpoint = ema = None
    try:
        print("=" * 80)
        print("Loading pretrained")
        model, graph, noise, cfg, checkpoint, ema = load_pretrained_model(
            pretrained,
            reference_cfg,
            device,
        )
        rows.extend(
            evaluate_loaded_model(
                model_name="pretrained",
                model=model,
                graph=graph,
                noise=noise,
                cfg=cfg,
                checkpoint=checkpoint,
                ema=ema,
                datasets=datasets,
                eval_batches=eval_batches,
                batch_size=batch_size,
                device=device,
                num_proc=num_proc,
            )
        )
    finally:
        del model, graph, noise, cfg, checkpoint, ema
        cleanup_cuda()

    for model_name, dirname in [("QA-best", "QA"), ("QAR-best", "QAR")]:
        model = graph = noise = cfg = checkpoint = ema = None

        try:
            print("=" * 80)
            print(f"Loading {model_name}")

            model_dir = output_root / dirname
            if not can_load_best(model_dir):
                print(f"Skip {model_name}: missing {model_dir / 'best.pth'} or best_eval.json")
                continue

            model, graph, noise, cfg, checkpoint, ema = load_best_model(
                model_dir,
                device,
            )

            rows.extend(
                evaluate_loaded_model(
                    model_name=model_name,
                    model=model,
                    graph=graph,
                    noise=noise,
                    cfg=cfg,
                    checkpoint=checkpoint,
                    ema=ema,
                    datasets=datasets,
                    eval_batches=eval_batches,
                    batch_size=batch_size,
                    device=device,
                    num_proc=num_proc,
                )
            )

        finally:
            del model, graph, noise, cfg, checkpoint, ema
            cleanup_cuda()

    if not rows:
        raise RuntimeError("No test results were produced.")

    write_outputs(rows, output_root)


if __name__ == "__main__":
    main()
