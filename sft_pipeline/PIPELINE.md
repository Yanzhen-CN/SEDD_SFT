# SEDD SFT Pipeline

## Goal

- Build QA and QAR supervised text datasets from S1K.
- Fine-tune from official pretrained SEDD weights, not from random initialization.
- The current default experiment is `QAR + sedd-medium + louaaron/sedd-medium`.
- Use validation score entropy loss to measure whether the model adapts to the QAR data distribution.

## How To Run

Run from the repository root on the server:

```bash
python run_sft.py
```

Main parameters are in:

```text
sft_pipeline/sft_config.yaml
```

Current key config:

```yaml
run:
  selected: QAR

sedd:
  model: medium
  pretrained_model: louaaron/sedd-medium

results:
  save_best: true
  output_dir: sft_pipeline/modelparameter
```

The official README examples are pretraining examples. They use very large effective batch sizes and long training schedules. For this single-GPU SFT pipeline, the config uses gradient accumulation:

```yaml
defaults:
  batch_size: 4

sedd:
  accum: 4

optim:
  lr: 3.0e-5
  warmup: 100
```

With `ngpus=1`, this keeps the per-step micro-batch at `batch_size / accum = 1`, while the optimizer update uses an effective batch size of 4.

## Data Flow

```text
sft_pipeline/data_process.py
  -> read simplescaling/s1K-1.1
  -> write sft_pipeline/data/QA
  -> write sft_pipeline/data/QAR
```

Current split is 800/100/100 for the 1000-row S1K dataset:

```yaml
data:
  valid_ratio: 0.10
  test_ratio: 0.10
```

QA format:

```text
User: question
Assistant: solution
```

QAR format:

```text
User: question
Assistant:
reasoning trajectory

Final Answer: solution
```

QA and QAR use the same examples and the same train/validation split.

## Training Flow

```text
run_sft.py
  -> sft_pipeline/run_sft.py
  -> train.py
  -> run_train.py
```

Training should log:

```text
Loading pretrained SEDD weights from louaaron/sedd-medium
Pretrained SEDD weights loaded.
```

## Result Directories

Official SEDD output:

```text
exp_local/sft_QAR/YYYY.MM.DD/HHMMSS/
```

Pipeline output for one run:

```text
sft_pipeline/modelparameter/QAR/YYYY.MM.DD_HHMMSS/
```

Global best across QAR runs:

```text
sft_pipeline/modelparameter/QAR/
```

Pretrained model reference:

```text
sft_pipeline/modelparameter/pretrained/medium/
```

## Important Files

Per-run directory:

```text
run_info.json
pretrain_eval.json
metrics.jsonl
metrics.csv
improvement_log.jsonl
best_eval.json
best.pth
train.log
```

Meaning:

- `run_info.json`: model, data, length, step count, and official output directory.
- `pretrain_eval.json`: validation loss before any SFT update.
- `metrics.jsonl`: full curve data, including pretrain, training, and evaluation records.
- `metrics.csv`: same curve data in CSV form for plotting.
- `improvement_log.jsonl`: only records evaluation points that improve the best loss.
- `best_eval.json`: best validation loss for this run.
- `best.pth`: best checkpoint for this run.
- `train.log`: copied log for this run.

Global QAR directory:

```text
sft_pipeline/modelparameter/QAR/best.pth
sft_pipeline/modelparameter/QAR/best_eval.json
sft_pipeline/modelparameter/QAR/improvement_log.jsonl
sft_pipeline/modelparameter/QAR/latest_pretrain_eval.json
sft_pipeline/modelparameter/pretrained/medium/model_info.json
sft_pipeline/modelparameter/pretrained/medium/latest_eval.json
```

Meaning:

- `best.pth`: current global best QAR checkpoint.
- `best_eval.json`: source run, step, and loss for the global best checkpoint.
- `improvement_log.jsonl`: history of global best updates.
- `latest_pretrain_eval.json`: latest pretrained baseline validation loss.
- `pretrained/medium/model_info.json`: pretrained model source; HF weights are not duplicated here.
- `pretrained/medium/latest_eval.json`: latest pretrained baseline validation loss for the medium model.

## Loss Logic

Before SFT, the script computes:

```text
pretrain_evaluation_loss
```

This is the run baseline. The current config averages validation loss over several batches:

```yaml
results:
  eval_batches: 8
  min_valid_loss: 1.0e-8
```

During training:

- If `evaluation_loss` is lower than the current run best, save to `QAR/YYYY.MM.DD_HHMMSS/best.pth`.
- If `evaluation_loss` is lower than the current global best, save to `QAR/best.pth`.
- If loss does not improve, do not overwrite best checkpoints.
- If loss is non-finite or too close to zero, do not use it for best checkpoint selection.

Use:

- `metrics.jsonl` for full training/evaluation curves.
- `improvement_log.jsonl` to explain why and when best checkpoints changed.

## Next Steps

- Run `QAR-medium-512-1000steps`.
- If memory is insufficient, change `length: 512` to `length: 256`.
- Plot curves from `metrics.jsonl`.
- Check whether `best_eval.json` is lower than `pretrain_eval.json`.
- Later, run QA as a matched control and compare pretrained, QA, and QAR samples with `compare_models.py`.
