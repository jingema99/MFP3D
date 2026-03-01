#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run.sh
# or:
#   chmod +x run.sh && ./run.sh

ENV_NAME="myenv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"

# Dataset directory (must contain train.h5 and test.h5)
DATA_DIR="./data/gt"
# Output root directory (structure: outputs/<dataset_name>/<target>/)
BASE_OUT="./outputs"
# Dataset name used for output grouping， here gt means ground truth 3d point clouds
DATASET_NAME="gt"
# Regression target: weight | volume | energy | protein | fat | carb
TARGET="volume"
# Training epochs
EPOCHS="200"
# Batch size
BATCH_SIZE="16"
# Number of DataLoader workers
NUM_WORKERS="0"

mkdir -p "${BASE_OUT}"

echo "== Train =="
conda run --no-capture-output -n "${ENV_NAME}" python "${PROJECT_DIR}/train.py" \
  --data_dir "${DATA_DIR}" \
  --target "${TARGET}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --save_dir "${BASE_OUT}" \
  --dataset_name "${DATASET_NAME}" \
  --console_mode compact

# echo "== Test =="
# conda run --no-capture-output -n "${ENV_NAME}" python "${PROJECT_DIR}/test.py" \
#   --data_dir "${DATA_DIR}" \
#   --dataset_name "${DATASET_NAME}" \
#   --target "${TARGET}" \
#   --output_root "${BASE_OUT}" \
#   --checkpoint "${BASE_OUT}/${DATASET_NAME}/${TARGET}/best.pt" \
#   --save_csv "${BASE_OUT}/${DATASET_NAME}/${TARGET}/pred_${TARGET}.csv"

# echo "Done. Check outputs under: ${BASE_OUT}/${DATASET_NAME}/${TARGET}"
