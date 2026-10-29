#!/usr/bin/env bash
# FlatQuant safety generation for one model + scheme on safechain (1000
# held-out prompts), wildguard and strongreject, writing the parquet schema
# score_safety.py reads. Resumable: a dataset whose parquet exists is skipped,
# and eval_safety_flatquant.py resumes a half-finished one from
# <parquet>.ckpt.json.
#
# Usage: run_flatquant_safety.sh <MODEL_ID> <SCHEME> <BUDGET|none> <OUT_DIR> <BATCH_SIZE>
#   parquets go to <OUT_DIR>/<model basename>-flatquant-<SCHEME>[-budget<BUDGET>]/<dataset>.parquet
#   TEMPERATURE / TOP_P env vars set sampling (default 0.6 / 0.95). Cap: 8192 tokens.
set -uo pipefail

# The login environment points HF_DATASETS_CACHE at /home/abalaji/..., which
# does not exist on Perlmutter; datasets.load_dataset fails creating it.
export HF_DATASETS_CACHE=/pscratch/sd/a/abalaji/hf_cache/datasets

cd /global/u2/a/abalaji/projects/agpt/adaptive-compute/eval

MODEL_ID="${1:?MODEL_ID required}"
SCHEME="${2:?SCHEME required}"
BUDGET="${3:?BUDGET required}"
OUT_DIR="${4:?OUT_DIR required}"
BATCH_SIZE="${5:?BATCH_SIZE required}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
declare -A N=( [safechain]=1000 [wildguard]=99999 [strongreject]=99999 )

model_dir="${OUT_DIR}/${MODEL_ID##*/}-flatquant-${SCHEME}"
budget_args=()
if [ "$BUDGET" != "none" ]; then
  model_dir="${model_dir}-budget${BUDGET}"
  budget_args=(--think-budget "$BUDGET")
fi
mkdir -p "$model_dir"

failed=0
for ds in strongreject safechain wildguard; do
  parquet="${model_dir}/${ds}.parquet"
  if [ -f "$parquet" ]; then
    echo "Skipping ${model_dir##*/} ${ds} (parquet exists)"
    continue
  fi
  echo ""
  echo "======================================================================"
  echo "  ${MODEL_ID} | FlatQuant ${SCHEME} | budget=${BUDGET} | ${ds} | T=${TEMPERATURE} top_p=${TOP_P} | $(date)"
  echo "======================================================================"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 ~/.conda/envs/flatquant/bin/python eval_safety_flatquant.py \
    --dataset "$ds" --scheme "$SCHEME" --model-id "$MODEL_ID" \
    --n "${N[$ds]}" --max-tokens 8192 "${budget_args[@]}" \
    --temperature "$TEMPERATURE" --top-p "$TOP_P" \
    --seed 42 --batch-size "$BATCH_SIZE" --parquet "$parquet"
  status=$?
  if [ $status -ne 0 ]; then
    echo "ERROR: ${model_dir##*/} ${ds} failed with exit code ${status}"
    failed=1
  fi
done

echo ""
echo "=== ${model_dir##*/} safety finished at $(date) (failed=${failed}) ==="
exit $failed
