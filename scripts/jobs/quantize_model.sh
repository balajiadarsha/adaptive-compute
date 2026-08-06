#!/usr/bin/env bash
#PBS -N quantize_model
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -l filesystems=home:eagle
#PBS -A argonne_tpc
#PBS -q preemptable

set -euo pipefail

PROJ_DIR="/home/abalaji/projects/reasoning/safety-eval"
MODEL_ID="l3lab/L1-Qwen-1.5B-Exact"
MODEL_SLUG="L1-Qwen-1.5B-Exact"
CKPT_DIR="/lus/flare/projects/ModCon/abalaji/adaptive/models/${MODEL_SLUG}"

mkdir -p "${CKPT_DIR}"

echo "=== GPTQ Quantization: ${MODEL_ID} ==="
echo "Node:   $(hostname)"
echo "Output: ${CKPT_DIR}"
date

# --- CUDA ---
ml use /soft/modulefiles
ml --ignore_cache load cudatoolkit-standalone/12.6.1

# --- Conda ---
#ml --ignore_cache load conda
#conda activate /lus/eagle/projects/argonne_tpc/abalaji/conda_env/safety_eval

# --- Proxy ---
export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export no_proxy="127.0.0.1,admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"

# --- HuggingFace cache ---
export HF_HOME="/lus/eagle/projects/argonne_tpc/abalaji/model_weights"
export HF_DATASETS_CACHE="/lus/eagle/projects/argonne_tpc/abalaji/model_weights"
export TRANSFORMERS_CACHE="/lus/eagle/projects/argonne_tpc/abalaji/model_weights/hub"
export HF_HUB_CACHE="/lus/eagle/projects/argonne_tpc/abalaji/model_weights/hub"

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
