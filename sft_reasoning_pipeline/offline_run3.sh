#!/usr/bin/env bash
set -euo pipefail

# Run reasoning-conditioned answer SFT (RA) on GPUs 0,1,2.
# Usage:
#   bash offline_run3.sh
#   bash offline_run3.sh sft_reasoning_pipeline/reasoning_config.yaml
# Optional env overrides:
#   GPUS="0 1 2" SEEDS="42 43 44" bash offline_run3.sh

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

CONFIG="${1:-sft_reasoning_pipeline/reasoning_config.yaml}"
LOG_DIR="sft_reasoning_pipeline/logs"
TMP_DIR="${LOG_DIR}/tmp_configs"
DATA_DIR="sft_reasoning_pipeline/data/RA"
TRAIN_JSONL="${DATA_DIR}/train.jsonl"
VAL_JSONL="${DATA_DIR}/validation.jsonl"
TEST_JSONL="${DATA_DIR}/test.jsonl"

mkdir -p "$LOG_DIR" "$TMP_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
PID_FILE="${LOG_DIR}/offline_run3_${STAMP}.pids"

# Offline / cache-friendly settings. Disable these manually if the model/dataset is not cached yet.
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] Missing config: $CONFIG"
  exit 1
fi

# Build RA data if it has not been generated. In offline mode this requires the dataset/cache or arrow_path.
if [[ ! -f "$TRAIN_JSONL" || ! -f "$VAL_JSONL" || ! -f "$TEST_JSONL" ]]; then
  echo "[INFO] Missing RA jsonl files. Running prepare_reasoning_data.py first..."
  python sft_reasoning_pipeline/prepare_reasoning_data.py --config "$CONFIG"
fi

if [[ ! -f "$TRAIN_JSONL" || ! -f "$VAL_JSONL" || ! -f "$TEST_JSONL" ]]; then
  echo "[ERROR] RA data still missing after preparation. Abort."
  echo "Expected: $TRAIN_JSONL, $VAL_JSONL, $TEST_JSONL"
  exit 1
fi

# Quick format check: RA should fix question+teacher reasoning and train only answer.
python - <<'PY'
import json
from pathlib import Path
p = Path("sft_reasoning_pipeline/data/RA/train.jsonl")
row = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
print("[CHECK] first RA sample segment mask:")
for name in row.get("segment_order", []):
    seg = row["segments"][name]
    print(f"  {name:16s} train={str(seg.get('train')):5s} text={repr(seg.get('text','')[:60])}")
expected = {
    "user_label": False,
    "user": False,
    "assistant_label": False,
    "reasoning_label": False,
    "reasoning": False,
    "answer_label": False,
    "answer": True,
}
for k, v in expected.items():
    assert row["segments"][k]["train"] is v, f"bad train flag for {k}"
print("[CHECK] RA mask OK: only answer is train=True")
PY

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

# Keep run_name as RA so train_answer.py writes all three seeds under
# sft_reasoning_pipeline/modelparameter/RA/<timestamp>/ and synchronizes the best
# checkpoint to sft_reasoning_pipeline/modelparameter/RA/best.pth.
cfg.setdefault("run", {})["selected"] = "RA"
cfg.setdefault("runs", {}).setdefault("RA", {})["dataset"] = "RA"
cfg.setdefault("training", {})["seed"] = seed

Path(dst).parent.mkdir(parents=True, exist_ok=True)
with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
PY
}

launch_one() {
  local gpu="$1"
  local seed="$2"
  local tag="RA_seed${seed}"
  local cfg="${TMP_DIR}/offline_run3_${tag}_${STAMP}.yaml"
  local log_file="${LOG_DIR}/train_${STAMP}_${tag}_cuda${gpu}.log"

  make_config "$CONFIG" "$cfg" "$seed"

  echo "[LAUNCH] ${tag} on CUDA_VISIBLE_DEVICES=${gpu}"
  echo "         config: $cfg"
  echo "         log:    $log_file"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" python sft_answer_pipeline/train_answer.py \
    --config "$cfg" \
    > "$log_file" 2>&1 &
  echo "${tag} cuda${gpu} pid=$! config=${cfg} log=${log_file}" | tee -a "$PID_FILE"
}

read -r -a GPU_ARR <<< "${GPUS:-0 1 2}"
read -r -a SEED_ARR <<< "${SEEDS:-42 43 44}"

if [[ "${#GPU_ARR[@]}" -ne 3 || "${#SEED_ARR[@]}" -ne 3 ]]; then
  echo "[ERROR] Please provide exactly three GPUs and three seeds."
  echo "Example: GPUS=\"0 1 2\" SEEDS=\"42 43 44\" bash offline_run3.sh"
  exit 1
fi

launch_one "${GPU_ARR[0]}" "${SEED_ARR[0]}"
sleep 2
launch_one "${GPU_ARR[1]}" "${SEED_ARR[1]}"
sleep 2
launch_one "${GPU_ARR[2]}" "${SEED_ARR[2]}"

echo "[DONE] Launched 3 reasoning-SFT RA jobs."
echo "[DONE] PID file: $PID_FILE"
echo "[DONE] Watch logs: tail -f ${LOG_DIR}/train_${STAMP}_*.log"
echo "[DONE] Global best checkpoint target: sft_reasoning_pipeline/modelparameter/RA/best.pth"
