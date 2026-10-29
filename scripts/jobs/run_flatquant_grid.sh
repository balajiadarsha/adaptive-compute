#!/usr/bin/env bash
# Full-dataset FlatQuant skill eval of one model + scheme at one reasoning
# token budget on aime/gpqa/math/gsm8k, with the same per-dataset caps as
# run_skill_grid.sh's "default" mode. Resumable at two levels: a dataset
# whose json exists is skipped, and eval_skill_flatquant.py resumes a
# half-finished dataset from <parquet>.ckpt.json.
#
# Usage: run_flatquant_grid.sh <MODEL_ID> <SCHEME> <BUDGET> <OUT_DIR> <BATCH_SIZE>
#   results go to <OUT_DIR>/<model basename>-flatquant-<SCHEME>-budget<BUDGET>/<dataset>.json
#   TEMPERATURE / TOP_P env vars set sampling (default greedy).
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
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"

declare -A MAX_TOKENS=( [aime]=30000 [gpqa]=8192 [math]=8192 [gsm8k]=4096 )
model_dir="${OUT_DIR}/${MODEL_ID##*/}-flatquant-${SCHEME}-budget${BUDGET}"
mkdir -p "$model_dir"

for ds in aime gpqa math gsm8k; do
  out_json="${model_dir}/${ds}.json"
  if [ -f "$out_json" ]; then
    echo "Skipping ${model_dir##*/} ${ds} (results exist)"
    continue
  fi
  echo ""
  echo "======================================================================"
  echo "  ${MODEL_ID} | FlatQuant ${SCHEME} | budget=${BUDGET} | ${ds} | max_tokens=${MAX_TOKENS[$ds]} | T=${TEMPERATURE} top_p=${TOP_P} | $(date)"
  echo "======================================================================"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 ~/.conda/envs/flatquant/bin/python eval_skill_flatquant.py \
    --dataset "$ds" --scheme "$SCHEME" --model-id "$MODEL_ID" \
    --n 99999 --max-tokens "${MAX_TOKENS[$ds]}" --think-budget "$BUDGET" \
    --temperature "$TEMPERATURE" --top-p "$TOP_P" \
    --seed 42 --batch-size "$BATCH_SIZE" \
    --parquet "${model_dir}/${ds}.parquet" --output "$out_json"
  status=$?
  if [ $status -ne 0 ]; then
    echo "ERROR: ${model_dir##*/} ${ds} failed with exit code ${status}"
  fi
done

echo ""
echo "=== ${model_dir##*/} complete at $(date) ==="
