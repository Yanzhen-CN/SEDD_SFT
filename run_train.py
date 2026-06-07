import datetime
import os
import os.path
import gc
import json
import logging
from itertools import chain

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

import data
import losses
import sampling
import graph_lib
import noise_lib
import utils
from model import SEDD
from model.ema import ExponentialMovingAverage
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from omegaconf import OmegaConf


torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # initialize the process group
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30)
    )


def cleanup():
    dist.destroy_process_group()


def run_multiprocess(rank, world_size, cfg, port):
    try:
        setup(rank, world_size, port)
        _run(rank, world_size, cfg)
    finally:
        cleanup()


def _run(rank, world_size, cfg):
    torch.cuda.set_device(rank)
    work_dir = cfg.work_dir
    work_dir_parts = os.path.normpath(work_dir).split(os.sep)
    run_instance = "_".join(work_dir_parts[-2:]) if len(work_dir_parts) >= 2 else os.path.basename(os.path.normpath(work_dir))
    run_name = OmegaConf.select(cfg, "results.run_name") or os.path.basename(os.path.normpath(work_dir))
    model_name = str(OmegaConf.select(cfg, "model.name") or OmegaConf.select(cfg, "model"))

    # Create directories for experimental logs
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    best_checkpoint_dir = os.path.join(work_dir, "checkpoints", "best.pth")
    best_eval_path = os.path.join(work_dir, "best_eval.json")
    pipeline_output_dir = OmegaConf.select(cfg, "results.output_dir")
    pipeline_global_dir = os.path.join(str(pipeline_output_dir), str(run_name)) if pipeline_output_dir else None
    pipeline_run_dir = os.path.join(pipeline_global_dir, run_instance) if pipeline_global_dir else None
    pipeline_run_best_checkpoint_dir = os.path.join(pipeline_run_dir, "best.pth") if pipeline_run_dir else None
    pipeline_run_best_eval_path = os.path.join(pipeline_run_dir, "best_eval.json") if pipeline_run_dir else None
    pipeline_run_improvement_log_path = os.path.join(pipeline_run_dir, "improvement_log.jsonl") if pipeline_run_dir else None
    pipeline_run_metrics_path = os.path.join(pipeline_run_dir, "metrics.jsonl") if pipeline_run_dir else None
    pipeline_global_best_checkpoint_dir = os.path.join(pipeline_global_dir, "best.pth") if pipeline_global_dir else None
    pipeline_global_best_eval_path = os.path.join(pipeline_global_dir, "best_eval.json") if pipeline_global_dir else None
    pipeline_global_improvement_log_path = os.path.join(pipeline_global_dir, "improvement_log.jsonl") if pipeline_global_dir else None
    pipeline_log_path = os.path.join(pipeline_run_dir, "train.log") if pipeline_run_dir else None
    pipeline_run_info_path = os.path.join(pipeline_run_dir, "run_info.json") if pipeline_run_dir else None
    pretrained_reference_dir = os.path.join(str(pipeline_output_dir), "pretrained", model_name) if pipeline_output_dir else None
    if rank == 0:
        utils.makedirs(sample_dir)
        utils.makedirs(checkpoint_dir)
        utils.makedirs(os.path.dirname(checkpoint_meta_dir))
        if pipeline_run_dir:
            utils.makedirs(pipeline_run_dir)
        if pipeline_global_dir:
            utils.makedirs(pipeline_global_dir)

    # logging
    if rank == 0:
        logger = utils.get_logger(os.path.join(work_dir, "logs"))
        if pipeline_log_path:
            pipeline_handler = logging.FileHandler(pipeline_log_path, mode="w")
            pipeline_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            logger.addHandler(pipeline_handler)
    def mprint(msg):
        if rank == 0:
            logger.info(msg)

    def write_metric(record):
        if rank != 0 or not pipeline_run_metrics_path:
            return
        with open(pipeline_run_metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    mprint(work_dir)
    mprint(cfg)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        mprint("Found {} CUDA devices.".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mprint(
                "{} \t Memory: {:.2f}GB".format(
                    props.name, props.total_memory / (1024 ** 3)
                )
            )
    else:
        mprint("WARNING: Using device {}".format(device))
    mprint(f"Found {os.cpu_count()} total number of CPUs.")

    # build token graph
    graph = graph_lib.get_graph(cfg, device)
    
    # build score model
    score_model = SEDD(cfg).to(device)
    pretrained_model = OmegaConf.select(cfg, "pretrained_model")
    if pretrained_model:
        mprint(f"Loading pretrained SEDD weights from {pretrained_model}")
        pretrained_score_model = SEDD.from_pretrained(str(pretrained_model)).to(device)
        score_model.load_state_dict(pretrained_score_model.state_dict(), strict=True)
        del pretrained_score_model
        mprint("Pretrained SEDD weights loaded.")
        if rank == 0 and pretrained_reference_dir and OmegaConf.select(cfg, "results.save_pretrained_reference", default=True):
            utils.makedirs(pretrained_reference_dir)
            with open(os.path.join(pretrained_reference_dir, "model_info.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "pretrained_model": str(pretrained_model),
                        "architecture": model_name,
                        "note": "Weights are loaded from this Hugging Face model at training time; they are not duplicated here.",
                    },
                    f,
                    indent=2,
                )

    score_model = DDP(score_model, device_ids=[rank], static_graph=True, find_unused_parameters=True)

    num_parameters = sum(p.numel() for p in score_model.parameters())
    mprint(f"Number of parameters in the model: {num_parameters}")

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=cfg.training.ema)
    mprint(score_model)
    mprint(f"EMA: {ema}")

    # build noise
    noise = noise_lib.get_noise(cfg).to(device)
    noise = DDP(noise, device_ids=[rank], static_graph=True)
    sampling_eps = 1e-5


    # build optimization state
    optimizer = losses.get_optimizer(cfg, chain(score_model.parameters(), noise.parameters()))
    mprint(f"Optimizer: {optimizer}")
    scaler = torch.cuda.amp.GradScaler()
    mprint(f"Scaler: {scaler}")
    state = dict(optimizer=optimizer, scaler=scaler, model=score_model, noise=noise, ema=ema, step=0) 
    if rank == 0 and pipeline_run_info_path:
        with open(pipeline_run_info_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
                    "run_name": run_name,
                    "run_instance": run_instance,
                    "work_dir": work_dir,
                    "pipeline_run_dir": pipeline_run_dir,
                    "pipeline_global_dir": pipeline_global_dir,
                    "model": model_name,
                    "pretrained_model": str(pretrained_model) if pretrained_model else None,
                    "train_data": str(cfg.data.train),
                    "valid_data": str(cfg.data.valid),
                    "length": int(cfg.model.length),
                    "steps": int(cfg.training.n_iters),
                    "batch_size": int(cfg.training.batch_size),
                },
                f,
                indent=2,
            )


    # load in state
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)
    initial_step = int(state['step'])

    
    # load in tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # Build data iterators
    train_ds, eval_ds = data.get_dataloaders(cfg)

    # mprint(f"Length of datasets: {len(train_ds)}, {len(eval_ds)}")

    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(cfg)
    train_step_fn = losses.get_step_fn(noise, graph, True, optimize_fn, cfg.training.accum)
    eval_step_fn = losses.get_step_fn(noise, graph, False, optimize_fn, cfg.training.accum)

    eval_batches = int(OmegaConf.select(cfg, "results.eval_batches", default=1))
    min_valid_loss = float(OmegaConf.select(cfg, "results.min_valid_loss", default=0.0))

    def evaluate_loss(num_batches):
        total_eval_loss = 0
        for _ in range(max(1, num_batches)):
            if cfg.data.valid != "text8":
                eval_batch = next(eval_iter)['input_ids'].to(device)
            else:
                eval_batch = next(train_iter).to(device)
            this_eval_loss = eval_step_fn(state, eval_batch)
            dist.all_reduce(this_eval_loss)
            this_eval_loss /= world_size
            total_eval_loss = total_eval_loss + this_eval_loss
        return total_eval_loss / max(1, num_batches)


    if cfg.training.snapshot_sampling:
        sampling_shape = (cfg.training.batch_size // (cfg.ngpus * cfg.training.accum), cfg.model.length)
        sampling_fn = sampling.get_sampling_fn(cfg, graph, noise, sampling_shape, sampling_eps, device)

    num_train_steps = cfg.training.n_iters
    mprint(f"Starting training loop at step {initial_step}.")
    run_best_eval_loss = None
    global_best_eval_loss = None
    pretrained_eval_loss = None
    if rank == 0 and OmegaConf.select(cfg, "results.save_best", default=False):
        if os.path.exists(best_eval_path):
            with open(best_eval_path, "r", encoding="utf-8") as f:
                run_best_eval_loss = float(json.load(f)["evaluation_loss"])
            mprint("Loaded existing run best evaluation_loss: %.5e" % run_best_eval_loss)
        if pipeline_global_best_eval_path and os.path.exists(pipeline_global_best_eval_path):
            with open(pipeline_global_best_eval_path, "r", encoding="utf-8") as f:
                global_best_eval_loss = float(json.load(f)["evaluation_loss"])
            mprint("Loaded existing global best evaluation_loss: %.5e" % global_best_eval_loss)

    if OmegaConf.select(cfg, "results.save_best", default=False):
        pretrained_eval_loss_tensor = evaluate_loss(eval_batches)
        pretrained_eval_loss = float(pretrained_eval_loss_tensor.item())
        mprint("pretrain_evaluation_loss: %.5e over %d batch(es)" % (pretrained_eval_loss, eval_batches))
        if rank == 0:
            pretrained_record = {
                "run_name": run_name,
                "run_instance": run_instance,
                "step": int(initial_step),
                "pretrain_evaluation_loss": pretrained_eval_loss,
                "eval_batches": eval_batches,
                "work_dir": work_dir,
                "pretrained_model": str(pretrained_model) if pretrained_model else None,
                "model": model_name,
            }
            with open(os.path.join(work_dir, "pretrain_eval.json"), "w", encoding="utf-8") as f:
                json.dump(pretrained_record, f, indent=2)
            if pipeline_run_dir:
                with open(os.path.join(pipeline_run_dir, "pretrain_eval.json"), "w", encoding="utf-8") as f:
                    json.dump(pretrained_record, f, indent=2)
            if pipeline_global_dir:
                with open(os.path.join(pipeline_global_dir, "latest_pretrain_eval.json"), "w", encoding="utf-8") as f:
                    json.dump(pretrained_record, f, indent=2)
            if pretrained_reference_dir:
                with open(os.path.join(pretrained_reference_dir, "latest_eval.json"), "w", encoding="utf-8") as f:
                    json.dump(pretrained_record, f, indent=2)
            if run_best_eval_loss is None:
                run_best_eval_loss = pretrained_eval_loss
            if global_best_eval_loss is None:
                global_best_eval_loss = pretrained_eval_loss
            write_metric({
                "run_name": run_name,
                "run_instance": run_instance,
                "kind": "pretrain_evaluation",
                "step": int(initial_step),
                "loss": pretrained_eval_loss,
                "pretrained_model": str(pretrained_model) if pretrained_model else None,
            })


    while state['step'] < num_train_steps + 1:
        step = state['step']


        if cfg.data.train != "text8":
            batch = next(train_iter)['input_ids'].to(device)
        else:
            batch = next(train_iter).to(device)
        loss = train_step_fn(state, batch)

        # flag to see if there was movement ie a full batch got computed
        if step != state['step']:
            if step % cfg.training.log_freq == 0:
                dist.all_reduce(loss)
                loss /= world_size

                train_value = float(loss.item())
                mprint("step: %d, training_loss: %.5e" % (step, train_value))
                write_metric({
                    "run_name": run_name,
                    "run_instance": run_instance,
                    "kind": "training",
                    "step": int(step),
                    "loss": train_value,
                })
            
            if step % cfg.training.snapshot_freq_for_preemption == 0 and rank == 0:
                utils.save_checkpoint(checkpoint_meta_dir, state)

            if step % cfg.training.eval_freq == 0:
                eval_loss = evaluate_loss(eval_batches)

                eval_value = float(eval_loss.item())
                is_valid_eval_loss = np.isfinite(eval_value) and eval_value > min_valid_loss
                mprint("step: %d, evaluation_loss: %.5e over %d batch(es)" % (step, eval_value, eval_batches))
                write_metric({
                    "run_name": run_name,
                    "run_instance": run_instance,
                    "kind": "evaluation",
                    "step": int(step),
                    "loss": eval_value,
                    "eval_batches": eval_batches,
                    "valid_for_best": bool(is_valid_eval_loss),
                    "loss_drop_from_pretrain": pretrained_eval_loss - eval_value if pretrained_eval_loss is not None else None,
                    "run_best_before": run_best_eval_loss,
                    "global_best_before": global_best_eval_loss,
                })
                if rank == 0 and OmegaConf.select(cfg, "results.save_best", default=False) and is_valid_eval_loss:
                    if run_best_eval_loss is None or eval_value < run_best_eval_loss:
                        previous_best = run_best_eval_loss
                        run_best_eval_loss = eval_value
                        utils.save_checkpoint(best_checkpoint_dir, state)
                        if pipeline_run_best_checkpoint_dir:
                            utils.save_checkpoint(pipeline_run_best_checkpoint_dir, state)
                        run_record = {
                            "run_name": run_name,
                            "run_instance": run_instance,
                            "scope": "run",
                            "step": int(step),
                            "pretrain_evaluation_loss": pretrained_eval_loss,
                            "eval_batches": eval_batches,
                            "previous_best_loss": previous_best,
                            "evaluation_loss": eval_value,
                            "loss_drop_from_pretrain": pretrained_eval_loss - eval_value if pretrained_eval_loss is not None else None,
                            "loss_drop_from_previous_best": previous_best - eval_value if previous_best is not None else None,
                            "source_run_dir": work_dir,
                            "source_checkpoint": best_checkpoint_dir,
                            "path": pipeline_run_best_checkpoint_dir,
                        }
                        with open(best_eval_path, "w", encoding="utf-8") as f:
                            json.dump(run_record, f, indent=2)
                        if pipeline_run_best_eval_path:
                            with open(pipeline_run_best_eval_path, "w", encoding="utf-8") as f:
                                json.dump(run_record, f, indent=2)
                        if pipeline_run_improvement_log_path:
                            with open(pipeline_run_improvement_log_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(run_record) + "\n")
                        mprint("new run best evaluation_loss: %.5e at step %d" % (eval_value, step))

                    if global_best_eval_loss is None or eval_value < global_best_eval_loss:
                        previous_global_best = global_best_eval_loss
                        global_best_eval_loss = eval_value
                        if pipeline_global_best_checkpoint_dir:
                            utils.save_checkpoint(pipeline_global_best_checkpoint_dir, state)
                        global_record = {
                            "run_name": run_name,
                            "run_instance": run_instance,
                            "scope": "global",
                            "step": int(step),
                            "pretrain_evaluation_loss": pretrained_eval_loss,
                            "eval_batches": eval_batches,
                            "previous_best_loss": previous_global_best,
                            "evaluation_loss": eval_value,
                            "loss_drop_from_pretrain": pretrained_eval_loss - eval_value if pretrained_eval_loss is not None else None,
                            "loss_drop_from_previous_best": previous_global_best - eval_value if previous_global_best is not None else None,
                            "source_run_dir": work_dir,
                            "source_checkpoint": best_checkpoint_dir,
                            "path": pipeline_global_best_checkpoint_dir,
                        }
                        if pipeline_global_best_eval_path:
                            with open(pipeline_global_best_eval_path, "w", encoding="utf-8") as f:
                                json.dump(global_record, f, indent=2)
                        if pipeline_global_improvement_log_path:
                            with open(pipeline_global_improvement_log_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(global_record) + "\n")
                        mprint("new global best evaluation_loss: %.5e at step %d" % (eval_value, step))

            if step > 0 and step % cfg.training.snapshot_freq == 0 or step == num_train_steps:
                # Save the checkpoint.
                save_step = step // cfg.training.snapshot_freq
                if rank == 0:
                    utils.save_checkpoint(os.path.join(
                        checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

                # Generate and save samples
                if cfg.training.snapshot_sampling:
                    mprint(f"Generating text at step: {step}")

                    this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                    utils.makedirs(this_sample_dir)

                    ema.store(score_model.parameters())
                    ema.copy_to(score_model.parameters())
                    sample = sampling_fn(score_model)
                    ema.restore(score_model.parameters())

                    sentences = tokenizer.batch_decode(sample)
                    
                    file_name = os.path.join(this_sample_dir, f"sample_{rank}.txt")
                    with open(file_name, 'w') as file:
                        for sentence in sentences:
                            file.write(sentence + "\n")
                            file.write("============================================================================================\n")

                    if cfg.eval.perplexity:
                        with torch.no_grad():
                            eval_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
                            batches = sample.shape[0] // cfg.eval.perplexity_batch_size
                            total_perplexity = 0
                            for i in range(batches):
                                s = sample[i * cfg.eval.perplexity_batch_size:(i + 1) * cfg.eval.perplexity_batch_size]
                                loss, logits = eval_model(s, labels=s)[:2]
                                logits = logits.transpose(-1, -2)
                                perplexity = F.cross_entropy(logits[..., :-1], s[..., 1:], reduction="none").mean(dim=-1).exp().mean()
                                total_perplexity += perplexity
                            total_perplexity /= batches
                            dist.all_reduce(total_perplexity)
                            total_perplexity /= world_size
                            mprint(f"Generative Perplexity at step: {step}. Perplexity: {total_perplexity:.3f}.")

                            del eval_model, logits, loss

                    dist.barrier()
