import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_best_model(model_dir, device):
    import graph_lib
    import noise_lib
    import utils
    from model import SEDD
    from model.ema import ExponentialMovingAverage

    model_dir = Path(model_dir)
    best_eval = load_json(model_dir / "best_eval.json")
    cfg = utils.load_hydra_config_from_run(best_eval["source_run_dir"])

    model = SEDD(cfg).to(device)
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg.training.ema)
    ckpt_path = model_dir / "best.pth"
    loaded_state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(loaded_state["model"])
    ema.load_state_dict(loaded_state["ema"])
    ema.store(model.parameters())
    ema.copy_to(model.parameters())

    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    return model, graph, noise, cfg, str(ckpt_path)


def can_load_best(model_dir):
    model_dir = Path(model_dir)
    return (model_dir / "best_eval.json").exists() and (model_dir / "best.pth").exists()


def load_pretrained_model(model_name, reference_cfg, device):
    import graph_lib
    import noise_lib
    from model import SEDD
    from model.ema import ExponentialMovingAverage

    model = SEDD.from_pretrained(model_name).to(device)
    graph = graph_lib.get_graph(reference_cfg, device)
    noise = noise_lib.get_noise(reference_cfg).to(device)
    ema = ExponentialMovingAverage(model.parameters(), decay=reference_cfg.training.ema)
    return model, graph, noise, reference_cfg, model_name, ema


def evaluate(model, graph, noise, cfg, dataset_path, split, eval_batches, batch_size, device, ema=None):
    import data
    import losses
    from model.ema import ExponentialMovingAverage

    dataset = data.get_dataset(
        dataset_path,
        split,
        cache_dir=cfg.data.cache_dir,
        block_size=cfg.model.length,
        num_proc=8,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    iterator = iter(loader)
    eval_step_fn = losses.get_step_fn(noise, graph, False, None, cfg.training.accum)
    if ema is None:
        ema = ExponentialMovingAverage(model.parameters(), decay=cfg.training.ema)

    state = {"model": model, "ema": ema}
    total = 0.0
    count = 0
    for _ in range(eval_batches):
        try:
            batch = next(iterator)["input_ids"].to(device)
        except StopIteration:
            break
        loss = eval_step_fn(state, batch)
        total += float(loss.item())
        count += 1
    if count == 0:
        raise ValueError(f"No batches found for {dataset_path} split={split}")
    return total / count, count


def write_outputs(rows, output_root):
    output_root = Path(output_root)
    result_dir = output_root / "test_result"
    result_dir.mkdir(parents=True, exist_ok=True)

    csv_path = result_dir / "test_results.csv"
    fieldnames = ["dataset", "model", "loss", "eval_batches", "checkpoint"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(result_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    for model_name in ["QA-best", "QAR-best"]:
        model_rows = [row for row in rows if row["model"] == model_name]
        if not model_rows:
            continue
        target_dir = output_root / model_name.replace("-best", "")
        target_dir.mkdir(parents=True, exist_ok=True)
        with open(target_dir / "best_test_result.json", "w", encoding="utf-8") as f:
            json.dump(model_rows, f, indent=2)

    pretrained_rows = [row for row in rows if row["model"] == "pretrained"]
    if pretrained_rows:
        pretrained_dir = output_root / "pretrained" / "medium"
        pretrained_dir.mkdir(parents=True, exist_ok=True)
        with open(pretrained_dir / "test_result.json", "w", encoding="utf-8") as f:
            json.dump(pretrained_rows, f, indent=2)

    print(f"Wrote {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained, QA-best, and QAR-best on QA/QAR test sets.")
    parser.add_argument("--output-root", default="sft_pipeline/modelparameter")
    parser.add_argument("--qa-data", default="sft_pipeline/data/QA")
    parser.add_argument("--qar-data", default="sft_pipeline/data/QAR")
    parser.add_argument("--pretrained", default="louaaron/sedd-medium")
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = Path(args.output_root)

    local_models = []
    reference_cfg = None

    for model_name, dirname in [("QA-best", "QA"), ("QAR-best", "QAR")]:
        model_dir = output_root / dirname
        if not can_load_best(model_dir):
            print(f"Skip {model_name}: {model_dir / 'best.pth'} or best_eval.json not found")
            continue
        model, graph, noise, cfg, ckpt = load_best_model(model_dir, device)
        local_models.append((model_name, model, graph, noise, cfg, ckpt, None))
        if reference_cfg is None:
            reference_cfg = cfg

    if reference_cfg is None:
        raise FileNotFoundError(
            f"No best checkpoint found under {output_root / 'QA'} or {output_root / 'QAR'}"
        )

    pretrained_model, pretrained_graph, pretrained_noise, pretrained_cfg, pretrained_ckpt, pretrained_ema = load_pretrained_model(
        args.pretrained,
        reference_cfg,
        device,
    )

    datasets = [("QA-test", args.qa_data), ("QAR-test", args.qar_data)]
    models = [("pretrained", pretrained_model, pretrained_graph, pretrained_noise, pretrained_cfg, pretrained_ckpt, pretrained_ema)]
    models.extend(local_models)

    rows = []
    for dataset_name, dataset_path in datasets:
        for model_name, model, graph, noise, cfg, ckpt, ema in models:
            loss, used_batches = evaluate(
                model,
                graph,
                noise,
                cfg,
                dataset_path,
                "test",
                args.eval_batches,
                args.batch_size,
                device,
                ema=ema,
            )
            row = {
                "dataset": dataset_name,
                "model": model_name,
                "loss": loss,
                "eval_batches": used_batches,
                "checkpoint": ckpt,
            }
            rows.append(row)
            print(row)

    write_outputs(rows, output_root)


if __name__ == "__main__":
    main()
