#!/usr/bin/env bash
# Create a conda env and install vLLM + adaptive-compute deps for Intel XPU.
#
# Usage: bash scripts/install_xpu_env.sh [env_name]
#
# Notes:
# - vLLM has no CUDA-free PyPI wheel; it must be built from source with
#   VLLM_TARGET_DEVICE=xpu, which uses the Intel Extension for PyTorch (IPEX)
#   toolchain instead of CUDA and never looks for CUDA_HOME.
# - Run this from inside adaptive-compute/ so the wildguard editable install
#   path (file:///../wildguard) resolves correctly. A sibling ../wildguard
#   checkout must exist first.

set -euo pipefail

ENV_NAME="/lus/flare/projects/ModCon/abalaji/conda/safety"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -d "${SCRIPT_DIR}/../wildguard" ]; then
  echo "Error: ${SCRIPT_DIR}/../wildguard not found."
  echo "Clone the wildguard repo as a sibling of adaptive-compute/ before running this script."
  exit 1
fi

if [ "${CONDA_PREFIX:-}" != "${ENV_NAME}" ]; then
  # Intel oneAPI's mpivars.deactivate.sh hook references SETVARS_CALL without a
  # default, which is fatal under `set -u` when conda re-triggers activate/
  # deactivate hooks. Export it so the hook's `[ "${SETVARS_CALL}" != "1" ]`
  # check succeeds instead of erroring on an unbound variable.
  export SETVARS_CALL="${SETVARS_CALL:-0}"
  conda activate "${ENV_NAME}"
fi

# --- vLLM (XPU build, no CUDA) ---
VLLM_BUILD_DIR="$(mktemp -d)"
git clone --depth 1 https://github.com/vllm-project/vllm.git "${VLLM_BUILD_DIR}"
cd "${VLLM_BUILD_DIR}"

# Installs the IPEX-linked torch build and XPU-specific deps; no CUDA toolkit involved.
pip install -v -r requirements/xpu.txt

# Explicitly scopes the build to the XPU backend, bypassing any CUDA_HOME lookup.
VLLM_TARGET_DEVICE=xpu python setup.py install

cd "${SCRIPT_DIR}"

# --- Remaining adaptive-compute deps ---
# --no-deps on the rest avoids pip re-resolving torch/vllm and pulling a CUDA build.
pip install --no-deps -r requirements-xpu.md

echo ""
echo "Done. Verify with:"
echo "  python -c \"import torch; print(torch.xpu.is_available())\""
echo "  python -c \"import vllm; print(vllm.__version__)\""
