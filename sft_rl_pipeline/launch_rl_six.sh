#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

LOG_DIR="sft_rl_pipeline/logs"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
PID_FILE="$LOG_DIR/rl_six_${STAMP}.pids"

PRETRAIN_CKPT="sft_rl_pipeline/modelparameter/startpoint/pretrained.pth"
QAR_CKPT="sft_rl_pipeline/modelparameter/startpoint/QAR-best.pth"
QAR_DATA="sft_rl_pipeline/data/QAR/train.jsonl"

if [[ ! -f "$PRETRAIN_CKPT" ]]; then
  echo "Missing $PRETRAIN_CKPT. Run: python save_pretrained_checkpoint.py"
  exit 1
fi

if [[ ! -f "$QAR_CKPT" ]]; then
  echo "Missing $QAR_CKPT. Copy your answer-SFT best QAR checkpoint there first."
  exit 1
fi

if [[ ! -f "$QAR_DATA" ]]; then
  echo "Missing $QAR_DATA. Run: python sft_rl_pipeline/copy_rl_data.py"
  exit 1
fi

launch_one() {
  local gpu="$1"
  local tag="$2"
  local config="$3"
  local log_file="$LOG_DIR/train_${STAMP}_${tag}_cuda${gpu}.log"

  echo "Launching ${tag} on CUDA ${gpu}; log=${log_file}"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" python sft_rl_pipeline/run_rl.py \
    --config "$config" \
    --train \
    --run-name "${tag}_cuda${gpu}" \
    > "$log_file" 2>&1 &
  echo "${tag} cuda${gpu} pid=$!" | tee -a "$PID_FILE"
}

launch_one 0 pretrain_rl_1 sft_rl_pipeline/rl_config_pretrained.yaml
launch_one 1 pretrain_rl_2 sft_rl_pipeline/rl_config_pretrained.yaml
launch_one 2 pretrain_rl_3 sft_rl_pipeline/rl_config_pretrained.yaml

launch_one 3 sft_rl_1 sft_rl_pipeline/rl_config.yaml
launch_one 4 sft_rl_2 sft_rl_pipeline/rl_config.yaml
launch_one 5 sft_rl_3 sft_rl_pipeline/rl_config.yaml

echo "All jobs launched."
echo "PIDs: $PID_FILE"
echo "Watch logs with: tail -f $LOG_DIR/train_${STAMP}_*.log"
