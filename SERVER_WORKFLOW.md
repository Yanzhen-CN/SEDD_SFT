# Server workflow

This is the current manual workflow for the anchored SEDD SFT experiments.
The old one-command SFT runners are deprecated so data preparation, training,
evaluation, generation, and visualization stay explicit.

## 0. Pull and enter the environment

```bash
cd /data/cyz/sedd
git pull
conda activate sedd
mkdir -p sft_answer_pipeline/logs sft_reasoning_pipeline/logs sft_rl_pipeline/logs
```

## 1. Build the shared S1K split

Create one raw S1K split first.  All later pipelines read this same split and
then apply their own token-length filters.  This keeps train/validation/test
assignment consistent and avoids leakage when QAR/RA drop long samples.

```bash
python prepare_s1k_split.py --config sft_answer_pipeline/answer_config.yaml
```

Outputs:

```text
data/s1k_split/train.jsonl
data/s1k_split/validation.jsonl
data/s1k_split/test.jsonl
data/s1k_split/manifest.json
```

## 2. Build anchored QAR / QA data

```bash
python sft_answer_pipeline/prepare_answer_data.py --config sft_answer_pipeline/answer_config.yaml
```

Check that the data is the new anchored format:

```bash
head -n 1 sft_answer_pipeline/data/QAR/train.jsonl
```

Expected order inside `segments`:

```text
user_label -> user -> assistant_label -> reasoning_label -> reasoning -> answer_label -> answer
```

Expected mask:

```text
User/question/Assistant/Reasoning:/Answer:  train=false
reasoning content + answer content          train=true
```

Data preparation now applies tokenizer-length filtering before training.  Check
these reports to see how many long S1K examples were dropped:

```text
sft_answer_pipeline/data/QAR/train_prepare_filter_report.json
sft_answer_pipeline/data/QAR/validation_prepare_filter_report.json
sft_answer_pipeline/data/QAR/test_prepare_filter_report.json
```

The dataloader may still write `*_load_report.json`, but those should show the
already-filtered prepared data.

## 3. Train main QAR SFT

Make sure `sft_answer_pipeline/answer_config.yaml` has:

```yaml
run:
  selected: QAR
```

Run:

```bash
nohup python sft_answer_pipeline/train_answer.py \
  --config sft_answer_pipeline/answer_config.yaml \
  > sft_answer_pipeline/logs/train_QAR_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Watch:

```bash
tail -f sft_answer_pipeline/logs/train_QAR_*.log
```

Main outputs:

```text
sft_answer_pipeline/modelparameter/QAR/<time>/best.pth
sft_answer_pipeline/modelparameter/QAR/best.pth
sft_answer_pipeline/modelparameter/QAR/best_eval.json
sft_answer_pipeline/modelparameter/QAR/best_metrics.csv
```

## 4. Evaluate and generate QAR examples

```bash
python sft_answer_pipeline/test_answer_eval.py --config sft_answer_pipeline/answer_config.yaml
python sft_answer_pipeline/generate_answer_examples.py --config sft_answer_pipeline/answer_config.yaml --dataset QAR
python sft_answer_pipeline/visual_answer.py
```

Reports and figures are written under:

```text
sft_answer_pipeline/reports/
sft_answer_pipeline/modelparameter/test_result/
```

For qualitative analysis, read `generated_completion`, because fixed anchors
are not part of the generated target loss.

## 5. Reasoning pipeline: RA diagnostic

`sft_reasoning_pipeline` is an independent diagnostic pipeline.  It keeps the
same text scaffold as QAR, but teacher reasoning is fixed and only the final
answer is trained:

```text
p(answer | question, teacher reasoning)
```

Build RA data:

```bash
python sft_reasoning_pipeline/prepare_reasoning_data.py --config sft_reasoning_pipeline/reasoning_config.yaml
```

RA reads the same `data/s1k_split` and then writes its own token-filter reports:

```text
sft_reasoning_pipeline/data/RA/train_prepare_filter_report.json
sft_reasoning_pipeline/data/RA/validation_prepare_filter_report.json
sft_reasoning_pipeline/data/RA/test_prepare_filter_report.json
```

Manual single run:

```bash
nohup python sft_answer_pipeline/train_answer.py \
  --config sft_reasoning_pipeline/reasoning_config.yaml \
  > sft_reasoning_pipeline/logs/train_RA_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Optional three-seed run on GPUs 0, 1, and 2:

```bash
bash sft_reasoning_pipeline/offline_run3.sh
```

After training:

```bash
python sft_reasoning_pipeline/test_reasoning_eval.py --config sft_reasoning_pipeline/reasoning_config.yaml
python sft_reasoning_pipeline/generate_reasoning_examples.py --config sft_reasoning_pipeline/reasoning_config.yaml
```

Main outputs:

```text
sft_reasoning_pipeline/modelparameter/RA/<time>/best.pth
sft_reasoning_pipeline/modelparameter/RA/best.pth
sft_reasoning_pipeline/modelparameter/RA/best_eval.json
sft_reasoning_pipeline/modelparameter/RA/best_metrics.csv
sft_reasoning_pipeline/reports/
```

## 6. Prepare RL-style continuation

Refresh the RL data snapshot every time QAR data format changes:

```bash
python sft_rl_pipeline/copy_rl_data.py \
  --source-data sft_answer_pipeline/data/QAR \
  --target-data sft_rl_pipeline/data/QAR
```

Copy the latest QAR SFT start point:

```bash
mkdir -p sft_rl_pipeline/modelparameter/startpoint
cp sft_answer_pipeline/modelparameter/QAR/best.pth \
  sft_rl_pipeline/modelparameter/startpoint/QAR-best.pth
```

Optional: save a local pretrained checkpoint for record keeping.  The
pretrained-start RL config can also load directly from Hugging Face when
`model.init_checkpoint` is empty.

```bash
python save_pretrained_checkpoint.py
```

## 7. Run six RL-style experiments

This is the only kept batch launcher.  It starts three pretrained-start runs
and three QAR-start runs on CUDA 0-5:

```bash
bash sft_rl_pipeline/launch_rl_six.sh
```

Logs:

```text
sft_rl_pipeline/logs/train_<time>_*.log
```

Outputs:

```text
sft_rl_pipeline/modelparameter/pretrained_rl/
sft_rl_pipeline/modelparameter/sft_rl/
sft_rl_pipeline/reports/
```

## 8. RL evaluation and visualization

```bash
python sft_rl_pipeline/test_rl_eval.py
python sft_rl_pipeline/run_rl.py --config sft_rl_pipeline/rl_config.yaml --reward-eval --tag QAR_RL
python sft_rl_pipeline/run_rl.py --config sft_rl_pipeline/rl_config.yaml --best-of-k --tag QAR_RL
python sft_rl_pipeline/visual_rl.py
```

## 9. Files worth copying back locally

```text
sft_answer_pipeline/modelparameter/
sft_answer_pipeline/reports/
sft_answer_pipeline/logs/
sft_reasoning_pipeline/modelparameter/
sft_reasoning_pipeline/reports/
sft_reasoning_pipeline/logs/
sft_rl_pipeline/modelparameter/
sft_rl_pipeline/reports/
sft_rl_pipeline/logs/
```
