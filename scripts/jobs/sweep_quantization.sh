#!/usr/bin/env bash
#PBS -N sweep_quantization
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -l filesystems=home:eagle
#PBS -A argonne_tpc
#PBS -q preemptable

set -euo pipefail

PROJ_DIR="/home/abalaji/projects/reasoning/safety-eval"
MODEL_ID="l3lab/L1-Qwen-1.5B-Exact"
MODEL_SLUG="L1-Qwen-1.5B-Exact"
CKPT_DIR="/lus/eagle/projects/argonne_tpc/abalaji/adaptive/checkpoints/${MODEL_SLUG}"
OUTPUT_DIR="/lus/eagle/projects/argonne_tpc/abalaji/adaptive/output/${MODEL_SLUG}-quant-sweep"

mkdir -p "${OUTPUT_DIR}"

echo "=== Quantization Sweep: ${MODEL_ID} ==="
echo "Node:   $(hostname)"
echo "Output: ${OUTPUT_DIR}"
date

# --- CUDA ---
ml use /soft/modulefiles
ml --ignore_cache load cudatoolkit-standalone/12.6.1

# --- Conda ---
ml --ignore_cache load conda
conda activate /lus/eagle/projects/argonne_tpc/abalaji/conda_env/safety_eval

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
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
export HOST_IP=${VLLM_HOST_IP}

cd "${PROJ_DIR}/adaptive-compute/eval"

python sweep_quantization.py \
  --dataset safechain \
  --bf16-model-id "${MODEL_ID}" \
  --int8-model-id "${CKPT_DIR}/gptq-int8" \
  --int4-model-id "${CKPT_DIR}/gptq-int4" \
  --int3-model-id "${CKPT_DIR}/gptq-int3" \
  --w8a8-model-id "${CKPT_DIR}/w8a8" \
  --w4a8-model-id "${CKPT_DIR}/w4a8" \
  --n 500 \
  --max-tokens 8192 \
  --seed 42 \
  --out-dir "${OUTPUT_DIR}"

echo ""
echo "=== Sweep complete: ${MODEL_SLUG} ==="
echo "Results: ${OUTPUT_DIR}/quantization_sweep_results.xlsx"
date
