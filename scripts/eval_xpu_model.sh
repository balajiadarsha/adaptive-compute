#!/usr/bin/env bash
# Evaluate a local checkpoint on SafeChain/WildGuard using vLLM on Intel XPU.
#
# Usage: bash scripts/eval_xpu_model.sh [model_path] [output_dir]
#
# Run this from a node with XPU access. Uses ALCF's `frameworks` module
# (Python 3.12, properly linked torch/vLLM/oneCCL against the system Cray
# libfabric) rather than the local conda env set up by
# scripts/install_xpu_env.sh -- that conda env ships its own libfabric.so.1
# built without the cxi provider, which shadows the system one via RPATH
# and breaks oneCCL init on the Slingshot interconnect.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

module load frameworks

MODEL_PATH="${1:-/lus/flare/projects/ModCon/abalaji/agpt/model/global_step138650}"
MODEL_SLUG="$(basename "${MODEL_PATH}")"
OUTPUT_DIR="${2:-/lus/flare/projects/ModCon/abalaji/adaptive/output/${MODEL_SLUG}}"

mkdir -p "${OUTPUT_DIR}"

echo "=== SafeChain/WildGuard Eval (XPU): ${MODEL_PATH} ==="
echo "Node:   $(hostname)"
echo "Output: ${OUTPUT_DIR}"
date

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONWARNINGS=ignore

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export VLLM_HOST_IP=$(getent hosts $(hostname).hsn.cm.aurora.alcf.anl.gov | awk '{ print $1 }' | tr ' ' '\n' | sort | head -n 1)
export HOST_IP=${VLLM_HOST_IP}
export no_proxy="localhost,127.0.0.1"

# oneCCL/ATL fails to find an OFI network provider for single-process runs
# unless the transport is pinned explicitly (fixes "can't create providers
# for name <default>" during LLM() init on Intel XPU). ofi talks to
# libfabric/cxi directly, avoiding the mpi transport's MPI_Init -> PMIx path
# (there's no PMIx server for this bare single-process run).
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=cxi
export CCL_OP_SYNC=1
export NUMEXPR_MAX_THREADS=512
export NUMEXPR_NUM_THREADS=128

cd "${SCRIPT_DIR}/eval"

python eval_safechain.py \
  --model-id "${MODEL_PATH}" \
  --dataset safechain \
  --n 500 \
  --max-tokens 8192 \
  --seed 42 \
  --parquet "${OUTPUT_DIR}/generations.parquet" \
  --output "${OUTPUT_DIR}/results.json"

echo ""
echo "=== Eval complete: ${MODEL_SLUG} ==="
echo "Results: ${OUTPUT_DIR}/results.json"
date
