#!/usr/bin/env bash
# Submits the three Perlmutter jobs that finish the skill results:
#   1. DeepSeek-R1 unconstrained rerun with a 30000-token cap on every
#      dataset (BF16, GPTQ-int8/int4, AWQ) -> results/full_eval_32k/
#   2. Reasoning-budget sweeps (512/1024/2048/4096) for DeepSeek-R1 and
#      DeepScaleR (BF16), with the same per-dataset caps as the L1 runs
#      -> results/full_eval/
#   3. FlatQuant W8A8KV8: resume MATH from its checkpoint, then GSM8K
#      -> results/full_eval/flatquant-w8a8kv8/
# Each job is one GPU on -q shared (see submit_full_eval_deepseek.sbatch
# for why shared is used on this account). All three are resumable:
# resubmitting this script skips every finished (model, budget, dataset).
set -euo pipefail

REPO=/global/u2/a/abalaji/projects/agpt/adaptive-compute
GRID=${REPO}/scripts/jobs/run_skill_grid.sh
LOGS=/pscratch/sd/a/abalaji/precision_sweep_results_data
CKPT=../quantize/checkpoints
COMMON=(-A amsc010_g -C gpu -q shared -N 1 -G 1)

sbatch "${COMMON[@]}" -J ds_32k_rerun -t 24:00:00 -o "${LOGS}/sbatch_ds_32k_rerun_%j.log" --wrap "
  bash ${GRID} deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B DeepSeek-R1-Distill-Qwen-1.5B 0 results/full_eval_32k 32k none
  bash ${GRID} ${CKPT}/DeepSeek-R1-Distill-Qwen-1.5B-gptq-int8 DeepSeek-R1-Distill-Qwen-1.5B-gptq-int8 0 results/full_eval_32k 32k none
  bash ${GRID} ${CKPT}/DeepSeek-R1-Distill-Qwen-1.5B-gptq-int4 DeepSeek-R1-Distill-Qwen-1.5B-gptq-int4 0 results/full_eval_32k 32k none
  bash ${GRID} ${CKPT}/DeepSeek-R1-Distill-Qwen-1.5B-awq-w4a16_asym DeepSeek-R1-Distill-Qwen-1.5B-awq-w4a16_asym 0 results/full_eval_32k 32k none
"

sbatch "${COMMON[@]}" -J budget_ds_dsr -t 24:00:00 -o "${LOGS}/sbatch_budget_ds_dsr_%j.log" --wrap "
  bash ${GRID} deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B DeepSeek-R1-Distill-Qwen-1.5B 0 results/full_eval default 512 1024 2048 4096
  bash ${GRID} agentica-org/DeepScaleR-1.5B-Preview DeepScaleR-1.5B-Preview 0 results/full_eval default 512 1024 2048 4096
"

# Calls eval_skill_flatquant.py directly with run_skill_sweep.py's FlatQuant
# arguments, not run_skill_sweep.py itself: the sweep rewrites
# results/full_eval/summary.csv and responses.parquet, which hold the
# DeepSeek vLLM results. Same --parquet path as the original run, so the
# script picks up math.parquet.ckpt.json and resumes at the last batch.
FQ_DIR=results/full_eval/flatquant-w8a8kv8
sbatch "${COMMON[@]}" -J flatquant_finish -t 47:00:00 -o "${LOGS}/sbatch_flatquant_finish_%j.log" --wrap "
  cd ${REPO}/eval
  for ds in math gsm8k; do
    cap=8192; [ \$ds = gsm8k ] && cap=4096
    [ -f ${FQ_DIR}/\$ds.json ] && { echo \"Skipping \$ds\"; continue; }
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 ~/.conda/envs/flatquant/bin/python eval_skill_flatquant.py \
      --dataset \$ds --scheme w8a8kv8 --n 99999 --max-tokens \$cap --seed 42 --batch-size 4 \
      --parquet ${FQ_DIR}/\$ds.parquet --output ${FQ_DIR}/\$ds.json
  done
"
