#!/usr/bin/env bash
#PBS -N eval_global_step138650
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -l filesystems=home:eagle
#PBS -A argonne_tpc
#PBS -q preemptable

set -euo pipefail

PROJ_DIR="/home/abalaji/projects/reasoning/safety-eval"
MODEL_ID="/lus/eagle/projects/argonne_tpc/abalaji/agpt/models/quantized/global_step138650"
MODEL_SLUG="Aurora-GPT-2B"
OUTPUT_DIR="/lus/eagle/projects/argonne_tpc/abalaji/adaptive/output/${MODEL_SLUG}"

mkdir -p "${OUTPUT_DIR}"

echo "=== Safety Eval: ${MODEL_ID} ==="
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

echo ""
echo "--- SafeChain ---"
python eval_safety.py --dataset safechain \
  --model-id "${MODEL_ID}" \
  --max-tokens 2048 \
  --seed 42 \
  --parquet "${OUTPUT_DIR}/generations_safechain.parquet" \
  --output "${OUTPUT_DIR}/results_safechain.json"

echo ""
echo "--- WildGuardMix (WildGuardTest) ---"
python eval_safety.py --dataset wildguard \
  --model-id "${MODEL_ID}" \
  --max-tokens 2048 \
  --seed 42 \
  --parquet "${OUTPUT_DIR}/generations_wildguard.parquet" \
  --output "${OUTPUT_DIR}/results_wildguard.json"

echo ""
echo "=== Eval complete: ${MODEL_SLUG} ==="
echo "Results: ${OUTPUT_DIR}/results_safechain.json, ${OUTPUT_DIR}/results_wildguard.json"
date
