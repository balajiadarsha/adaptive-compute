#!/usr/bin/env bash
# Full-dataset skill eval of one model over a grid of reasoning token
# budgets (or unconstrained, "none") on aime/gpqa/math/gsm8k. Resumable:
# any (budget, dataset) whose output json already exists is skipped.
#
# Usage: run_skill_grid.sh <MODEL_ID_OR_PATH> <OUT_PREFIX> <CUDA_DEVICE> <OUT_DIR> <CAP_MODE> <BUDGET...>
#   OUT_PREFIX  results go to <OUT_DIR>/<OUT_PREFIX>[-budget<N>]/<dataset>.json
#   CAP_MODE    "default" = per-dataset caps used by the L1 runs
#               (aime 30000, gpqa/math 8192, gsm8k 4096);
#               "32k" = 30000 for every dataset (leaves room for the prompt
#               inside the 32768 context)
#   BUDGET      one or more of 512 1024 2048 4096 ..., or "none"
set -uo pipefail

cd /global/u2/a/abalaji/projects/agpt/adaptive-compute/eval

MODEL_ID="${1:?MODEL_ID_OR_PATH required}"
OUT_PREFIX="${2:?OUT_PREFIX required}"
CUDA_DEVICE="${3:?CUDA_DEVICE required}"
OUT_DIR="${4:?OUT_DIR required}"
CAP_MODE="${5:?CAP_MODE required}"
shift 5
BUDGETS="$*"

DATASETS="aime gpqa math gsm8k"
N=99999
SEED=42
# Sampling: greedy by default; override with e.g. TEMPERATURE=0.6 TOP_P=0.95
# (DeepSeek-R1-distill recommended settings -- greedy decoding makes these
# models loop until max_tokens).
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"

if [ "$CAP_MODE" = "32k" ]; then
  declare -A MAX_TOKENS=( [aime]=30000 [gpqa]=30000 [math]=30000 [gsm8k]=30000 )
else
  declare -A MAX_TOKENS=( [aime]=30000 [gpqa]=8192 [math]=8192 [gsm8k]=4096 )
fi

for budget in $BUDGETS; do
  if [ "$budget" = "none" ]; then
    model_dir="${OUT_DIR}/${OUT_PREFIX}"
    budget_args=()
  else
    model_dir="${OUT_DIR}/${OUT_PREFIX}-budget${budget}"
    budget_args=(--think-budget "$budget")
  fi
  mkdir -p "$model_dir"
  for ds in $DATASETS; do
    out_json="${model_dir}/${ds}.json"
    if [ -f "$out_json" ]; then
      echo "Skipping ${OUT_PREFIX} budget=${budget} dataset=${ds} (results exist at ${out_json})"
      continue
    fi
    echo ""
    echo "======================================================================"
    echo "  ${OUT_PREFIX} | budget=${budget} | ${ds} | max_tokens=${MAX_TOKENS[$ds]} | T=${TEMPERATURE} top_p=${TOP_P}"
    echo "======================================================================"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" ~/.conda/envs/reason/bin/python eval_skill.py \
      --dataset "$ds" \
      --model-id "$MODEL_ID" \
      --n "$N" \
      --max-tokens "${MAX_TOKENS[$ds]}" \
      "${budget_args[@]}" \
      --seed "$SEED" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --tensor-parallel-size 1 \
      --parquet "${model_dir}/${ds}.parquet" \
      --output "$out_json"
    status=$?
    if [ $status -ne 0 ]; then
      echo "ERROR: ${OUT_PREFIX} budget=${budget} dataset=${ds} failed with exit code ${status}"
    fi
  done
done

echo ""
echo "=== ${OUT_PREFIX} grid complete ==="
