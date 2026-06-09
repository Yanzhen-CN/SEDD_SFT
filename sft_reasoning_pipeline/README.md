# sft_reasoning_pipeline

This pipeline is an ablation for **reasoning-conditioned answer SFT**.
It keeps the same assistant-style layout as QAR, but changes the train mask:

```text
User: <question>                      # fixed condition, no loss
Assistant:                            # fixed condition, no loss
Reasoning:
<teacher reasoning>                   # fixed condition, no loss

Answer:
<answer>                              # train target, score-entropy loss only here
```

So this pipeline trains:

```text
p(answer | question, teacher reasoning)
```

It reuses `sft_answer_pipeline/answer_dataset.py`, `answer_losses.py`, and `train_answer.py`.

## Single run

```bash
# 1. Build RA data
python sft_reasoning_pipeline/prepare_reasoning_data.py --config sft_reasoning_pipeline/reasoning_config.yaml

# 2. Train. train_answer.py can train arbitrary run names from the config when --run is omitted.
python sft_answer_pipeline/train_answer.py --config sft_reasoning_pipeline/reasoning_config.yaml

# 3. Test loss
python sft_reasoning_pipeline/test_reasoning_eval.py --config sft_reasoning_pipeline/reasoning_config.yaml

# 4. Qualitative generation
python sft_reasoning_pipeline/generate_reasoning_examples.py --config sft_reasoning_pipeline/reasoning_config.yaml
```

## Parallel 3-GPU run

Launch three independent RA runs on CUDA 0, 1, and 2 with different seeds:

```bash
bash sft_reasoning_pipeline/launch_reasoning_three.sh
```

The script will build RA data first if `sft_reasoning_pipeline/data/RA/train.jsonl` is missing.
It writes logs to:

```text
sft_reasoning_pipeline/logs/
```

Because `train_answer.py` synchronizes the best run under the global run directory, the best checkpoint across these RA runs will be available at:

```text
sft_reasoning_pipeline/modelparameter/RA/best.pth
```

Use the shared raw S1K split first:

```bash
python prepare_s1k_split.py --config sft_reasoning_pipeline/reasoning_config.yaml
```

Token filtering now happens during `prepare_reasoning_data.py`: samples longer
than `model.max_length` are dropped, not truncated. Filter reports are written
beside each split file, for example:

```text
sft_reasoning_pipeline/data/RA/train_prepare_filter_report.json
sft_reasoning_pipeline/data/RA/validation_prepare_filter_report.json
sft_reasoning_pipeline/data/RA/test_prepare_filter_report.json
```
