# SFT-RL pipeline notes for regenerated QAR

This pipeline assumes the QAR data format from `sft_answer_pipeline` v2:

```text
User: <question>
Assistant:
Reasoning:
<reasoning>

Answer:
<answer>
```

`User` and `Assistant:` are fixed condition tokens. `Reasoning: ... Answer: ...`
is the assistant completion target.

## Important

After every QAR data-format change or QAR retraining, refresh both:

1. RL QAR data snapshot
2. RL SFT start checkpoint

```bash
python sft_rl_pipeline/copy_rl_data.py \
  --source-data sft_answer_pipeline/data/QAR \
  --target-data sft_rl_pipeline/data/QAR

mkdir -p sft_rl_pipeline/modelparameter/startpoint
cp sft_answer_pipeline/modelparameter/QAR/best.pth \
  sft_rl_pipeline/modelparameter/startpoint/QAR-best.pth
```

If you compare pretrained-start RL, also create:

```bash
python save_pretrained_checkpoint.py
# or otherwise ensure this exists:
# sft_rl_pipeline/modelparameter/startpoint/pretrained.pth
```

## Run

Single QAR-start reward-weighted continuation:

```bash
python sft_rl_pipeline/run_rl.py --config sft_rl_pipeline/rl_config.yaml --train
```

Best-of-K generation only:

```bash
python sft_rl_pipeline/run_rl.py --config sft_rl_pipeline/rl_config.yaml --best-of-k
```

Evaluate checkpoints:

```bash
python sft_rl_pipeline/test_rl_eval.py
python sft_rl_pipeline/visual_rl.py
```

## Caveat

`train_reward_weighted.py` is not true online policy-gradient RL. It is a
reward-weighted DWDSE/SFT continuation: a rule reward is computed on the
reference assistant completion and used as a stable sample weight. `best_of_k.py`
uses the same rule reward to select among generated candidates.

## Anchored QAR content reward v2

In the current QAR data, `Reasoning:` and `Answer:` are fixed anchors (`train=False`).  The reward should therefore focus on the content inside those sections, not on whether the model generated the anchor words.

Recommended workflow:

```bash
python sft_rl_pipeline/analyze_reward_vocabulary.py \
  --data-dir sft_answer_pipeline/data/QAR \
  --out sft_rl_pipeline/reports/s1k_reward_vocab_stats.json

python sft_rl_pipeline/run_rl.py --config sft_rl_pipeline/rl_config.yaml --reward-eval --tag QAR_before_RL
python sft_rl_pipeline/run_rl.py --config sft_rl_pipeline/rl_config.yaml --train --run-name content_reward_v2
python sft_rl_pipeline/run_rl.py --config sft_rl_pipeline/rl_config.yaml --checkpoint sft_rl_pipeline/modelparameter/sft_rl/best_RL_QAR.pth --reward-eval --tag after_RL
python sft_rl_pipeline/run_rl.py --config sft_rl_pipeline/rl_config.yaml --best-of-k --tag content_reward_v2
```

The training metric still includes DWDSE loss.  Generation-level reward is written to `sft_rl_pipeline/reports/generation_reward_*_summary.json`, and best-of-K component summaries are written to `sft_rl_pipeline/reports/best_of_k_*_summary.json`.
