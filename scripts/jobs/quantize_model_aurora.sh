#!/usr/bin/env bash
#PBS -N quantize_model
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -l filesystems=home:flare
#PBS -A ModCon
#PBS -q capacity

set -euo pipefail

PROJ_DIR="/home/abalaji/projects/agpt/safety-eval"
MODEL_ID="l3lab/L1-Qwen-1.5B-Exact"
MODEL_SLUG="L1-Qwen-1.5B-Exact"
CKPT_DIR="/lus/flare/projects/ModCon/abalaji/adaptive/models/${MODEL_SLUG}"

mkdir -p "${CKPT_DIR}"

echo "=== GPTQ Quantization: ${MODEL_ID} ==="
echo "Node:   $(hostname)"
echo "Output: ${CKPT_DIR}"
date

# --- Frameworks (torch/transformers/vllm, Intel XPU-aware) ---
module load frameworks

# --- Venv (layered on frameworks via --system-site-packages) ---
source "${PROJ_DIR}/.safety_env/bin/activate"

# --- HuggingFace cache ---
export HF_HOME="/lus/flare/projects/ModCon/abalaji/cache/model_weights"
export HF_DATASETS_CACHE="/lus/flare/projects/ModCon/abalaji/cache/model_weights"
export HF_HUB_CACHE="/lus/flare/projects/ModCon/abalaji/cache/model_weights/hub"

# --- General ---
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONWARNINGS=ignore

cd "${PROJ_DIR}/adaptive-compute/quantize"

for BITS in 8 4 3; do
  echo ""
  echo "--- Quantizing to int${BITS} ---"
  python quantize_model.py \
    --model-id "${MODEL_ID}" \
    --bits "${BITS}" \
    --calib-dataset safechain \
    --calib-samples 256 \
    --seed 42 \
    --output-dir "${CKPT_DIR}/gptq-int${BITS}"
done

echo ""
echo "=== Quantization complete: ${MODEL_SLUG} ==="
echo "Checkpoints: ${CKPT_DIR}/gptq-int{8,4,3}"
date
