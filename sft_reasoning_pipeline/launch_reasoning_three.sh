#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

CONFIG="${1:-sft_reasoning_pipeline/reasoning_config.yaml}"
LOG_DIR="sft_reasoning_pipeline/logs"
TMP_DIR="${LOG_DIR}/tmp_configs"
mkdir -p "$LOG_DIR" "$TMP_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
PID_FILE="${LOG_DIR}/reasoning_three_${STAMP}.pids"
DATA_DIR="sft_reasoning_pipeline/data/RA"
TRAIN_JSONL="${DATA_DIR}/train.jsonl"

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config: $CONFIG"
  exit 1
fi

if [[ ! -f "$TRAIN_JSONL" ]]; then
  echo "Missing $TRAIN_JSONL. Building RA data first..."
  python sft_reasoning_pipeline/prepare_reasoning_data.py --config "$CONFIG"
fi

if [[ ! -f "$TRAIN_JSONL" ]]; then
  echo "Still missing $TRAIN_JSONL after data preparation. Abort."
  exit 1
fi

make_config() {
  local src="$1"
  local dst="$2"
  local seed="$3"
  python - "$src" "$dst" "$seed" <<'PY'
import sys
from pathlib import Path
import yaml

src, dst, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

cfg.setdefault("run", {})["selected"] = "RA"
cfg.setdefault("training", {})["seed"] = seed
cfg.setdefault("runs", {}).setdefault("RA", {})["dataset"] = "RA"

Path(dst).parent.mkdir(parents=True, exist_ok=True)
with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
PY
}

launch_one() {
  local gpu="$1"
  local seed="$2"
  local tag="RA_seed${seed}"
  local cfg="${TMP_DIR}/reasoning_${tag}_${STAMP}.yaml"
  local log_file="${LOG_DIR}/train_${STAMP}_${tag}_cuda${gpu}.log"

  make_config "$CONFIG" "$cfg" "$seed"

  echo "Launching ${tag} on CUDA ${gpu}; config=${cfg}; log=${log_file}"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" python sft_answer_pipeline/train_answer.py \
    --config "$cfg" \
    > "$log_file" 2>&1 &
  echo "${tag} cuda${gpu} pid=$! config=${cfg} log=${log_file}" | tee -a "$PID_FILE"
}

launch_one 0 42
sleep 2
launch_one 1 43
sleep 2
launch_one 2 44

echo "All reasoning-SFT jobs launched."
echo "PIDs: $PID_FILE"
echo "Watch logs with: tail -f ${LOG_DIR}/train_${STAMP}_*.log"
echo "Best checkpoint will be synchronized under: sft_reasoning_pipeline/modelparameter/RA/best.pth"
