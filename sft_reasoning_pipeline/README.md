# sft_reasoning_pipeline

This pipeline is an ablation for **reasoning-conditioned answer SFT**:

```text
User: <question>
Reasoning: <teacher reasoning>        # fixed condition, no loss
Assistant:
Final Answer:
<answer>                              # train target, score-entropy loss only here
```

It reuses `sft_answer_pipeline/answer_dataset.py`, `answer_losses.py`, and `train_answer.py`.

## Run

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

Token filtering happens in `AnswerSegmentDataset`: samples longer than `model.max_length` are dropped, not truncated. Load reports are written beside each split file, for example:

```text
sft_reasoning_pipeline/data/RA/train_load_report.json
sft_reasoning_pipeline/data/RA/validation_load_report.json
sft_reasoning_pipeline/data/RA/test_load_report.json
```
