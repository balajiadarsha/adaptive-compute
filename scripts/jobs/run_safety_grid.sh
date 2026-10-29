#!/usr/bin/env bash
# Safety generation for one model over a grid of reasoning token budgets
# (or unconstrained, "none") on safechain (1000 held-out prompts), wildguard
# (all of WildGuardTest) and strongreject (313). Generation only: score the
# parquets afterwards with eval/score_safety.py. Resumable: any
# (budget, dataset) whose parquet already exists is skipped.
#
# Usage: run_safety_grid.sh <MODEL_ID_OR_PATH> <OUT_PREFIX> <OUT_DIR> <CAP_MODE> <BUDGET...>
#   OUT_PREFIX  parquets go to <OUT_DIR>/<OUT_PREFIX>[-budget<N>]/<dataset>.parquet
#   CAP_MODE    "default" = 8192 max tokens; "32k" = 30000 (the "No limit" rows)
#   BUDGET      one or more of 512 1024 2048 4096 ..., or "none"
#   TEMPERATURE / TOP_P env vars set sampling (default 0.6 / 0.95).
set -uo pipefail

# The login environment points HF_DATASETS_CACHE at /home/abalaji/..., which
# does not exist on Perlmutter; datasets.load_dataset fails creating it.
export HF_DATASETS_CACHE=/pscratch/sd/a/abalaji/hf_cache/datasets

cd /global/u2/a/abalaji/projects/agpt/adaptive-compute/eval

MODEL_ID="${1:?MODEL_ID_OR_PATH required}"
OUT_PREFIX="${2:?OUT_PREFIX required}"
OUT_DIR="${3:?OUT_DIR required}"
CAP_MODE="${4:?CAP_MODE required}"
shift 4
BUDGETS="$*"

TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
SEED=42
declare -A N=( [safechain]=1000 [wildguard]=99999 [strongreject]=99999 )
if [ "$CAP_MODE" = "32k" ]; then MAX_TOKENS=30000; else MAX_TOKENS=8192; fi

failed=0
for budget in $BUDGETS; do
  if [ "$budget" = "none" ]; then
    model_dir="${OUT_DIR}/${OUT_PREFIX}"
    budget_args=()
  else
    model_dir="${OUT_DIR}/${OUT_PREFIX}-budget${budget}"
    budget_args=(--think-budget "$budget")
  fi
  mkdir -p "$model_dir"
  for ds in safechain wildguard strongreject; do
    parquet="${model_dir}/${ds}.parquet"
    if [ -f "$parquet" ]; then
      echo "Skipping ${OUT_PREFIX} budget=${budget} dataset=${ds} (${parquet} exists)"
      continue
    fi
    echo ""
    echo "======================================================================"
    echo "  ${OUT_PREFIX} | budget=${budget} | ${ds} | max_tokens=${MAX_TOKENS} | T=${TEMPERATURE} top_p=${TOP_P} | $(date)"
    echo "======================================================================"
    CUDA_VISIBLE_DEVICES=0 ~/.conda/envs/reason/bin/python eval_safety.py --generate \
      --dataset "$ds" \
      --model-id "$MODEL_ID" \
      --n "${N[$ds]}" \
      --max-tokens "$MAX_TOKENS" \
      "${budget_args[@]}" \
      --seed "$SEED" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --tensor-parallel-size 1 \
      --parquet "$parquet"
    status=$?
    if [ $status -ne 0 ]; then
      echo "ERROR: ${OUT_PREFIX} budget=${budget} dataset=${ds} failed with exit code ${status}"
      failed=1
    fi
  done
done

echo ""
echo "=== ${OUT_PREFIX} safety grid finished at $(date) (failed=${failed}) ==="
exit $failed
