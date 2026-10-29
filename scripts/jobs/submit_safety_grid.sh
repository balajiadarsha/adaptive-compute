#!/usr/bin/env bash
# Submits the safety grid (see make_safety_table.py for the rows): one
# generation job per model group, each followed by a scoring job
# (score_safety.py over that group's parquets, run afterany so a partial
# group still gets scored). Everything is resumable: resubmitting skips
# existing parquets and already-scored jsons.
#
# Dependencies on jobs outside this script, as env vars (job IDs):
#   L1_7B_QUANT_JOB  quantize_l1_7b (L1-7B GPTQ/AWQ checkpoints)
#   FQ_CALIB_1P5B    fq_calib_l1_1p5b (L1-1.5B FlatQuant calibration)
#   FQ_CALIB_7B      fq_calib_l1_7b  (L1-7B FlatQuant calibration)
set -euo pipefail

REPO=/global/u2/a/abalaji/projects/agpt/adaptive-compute
GRID=${REPO}/scripts/jobs/run_safety_grid.sh
FQ=${REPO}/scripts/jobs/run_flatquant_safety.sh
LOGS=/pscratch/sd/a/abalaji/precision_sweep_results_data
CKPT=${REPO}/quantize/checkpoints
B=results/safety_t06
NL=results/safety_32k_t06
COMMON=(-A amsc010_g -C gpu -q shared -N 1 -G 1)

# submit <name> <hours> <dependency or ""> <score globs> <commands>
submit() {
  local name=$1 hours=$2 dep=$3 globs=$4 cmds=$5
  local dep_args=()
  [ -n "$dep" ] && dep_args=(--dependency="afterok:${dep}")
  local gen
  gen=$(sbatch --parsable "${COMMON[@]}" "${dep_args[@]}" -J "safety_${name}" -t "${hours}:00:00" \
    -o "${LOGS}/sbatch_safety_${name}_%j.log" --wrap "$cmds")
  local score
  score=$(sbatch --parsable "${COMMON[@]}" --dependency="afterany:${gen}" -J "score_${name}" -t 04:00:00 \
    -o "${LOGS}/sbatch_score_${name}_%j.log" --wrap "
      export HF_DATASETS_CACHE=/pscratch/sd/a/abalaji/hf_cache/datasets HF_HUB_OFFLINE=1
      cd ${REPO}/eval
      shopt -s nullglob
      ~/.conda/envs/reason/bin/python score_safety.py ${globs}")
  echo "${name}: generate ${gen}, score ${score}"
}

# Length control, BF16: budgets 512-4096 (8192 cap) and No limit (30000 cap).
for spec in "deepseek deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B DeepSeek-R1-Distill-Qwen-1.5B 24" \
            "deepscaler agentica-org/DeepScaleR-1.5B-Preview DeepScaleR-1.5B-Preview 24" \
            "l1_1p5b l3lab/L1-Qwen-1.5B-Exact L1-Qwen-1.5B-Exact 12" \
            "l1_7b l3lab/L1-Qwen-7B-Exact L1-Qwen-7B-Exact 36"; do
  set -- $spec
  submit "$1" "$4" "" "${B}/$3-budget*/*.parquet ${NL}/$3/*.parquet" "
    bash ${GRID} $2 $3 ${B} default 512 1024 2048 4096
    bash ${GRID} $2 $3 ${NL} 32k none"
done

# L1-1.5B GPTQ/AWQ across budgets.
cmds=""
for q in gptq-int8 gptq-int4 awq-w4a16_asym; do
  cmds+="bash ${GRID} ${CKPT}/L1-Qwen-1.5B-Exact-${q} L1-Qwen-1.5B-Exact-${q} ${B} default 512 1024 2048 4096; "
done
submit l1_1p5b_quant 24 "" "${B}/L1-Qwen-1.5B-Exact-*-budget*/*.parquet" "$cmds"

# DeepSeek-R1-1.5B GPTQ/AWQ, No limit only.
cmds=""
for q in gptq-int8 gptq-int4 awq-w4a16_asym; do
  cmds+="bash ${GRID} ${CKPT}/DeepSeek-R1-Distill-Qwen-1.5B-${q} DeepSeek-R1-Distill-Qwen-1.5B-${q} ${NL} 32k none; "
done
submit deepseek_quant 36 "" "${NL}/DeepSeek-R1-Distill-Qwen-1.5B-*/*.parquet" "$cmds"

# L1-7B GPTQ/AWQ across budgets, after the checkpoints exist.
cmds=""
for q in gptq-int8 gptq-int4 awq-w4a16_asym; do
  cmds+="bash ${GRID} ${CKPT}/L1-Qwen-7B-Exact-${q} L1-Qwen-7B-Exact-${q} ${B} default 512 1024 2048 4096; "
done
submit l1_7b_quant 47 "${L1_7B_QUANT_JOB:-}" "${B}/L1-Qwen-7B-Exact-*-budget*/*.parquet" "$cmds"

# FlatQuant (HF generate, slow): DeepSeek W8A8KV8 with no budget prompt, and
# L1 1.5B/7B at budget 1024 once their calibrations exist.
submit deepseek_fq 47 "" "${B}/DeepSeek-R1-Distill-Qwen-1.5B-flatquant-*/*.parquet" \
  "bash ${FQ} deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B w8a8kv8 none ${B} 16"
# Leave FQ_CALIB_* unset once the calibration has finished: Slurm rejects a
# dependency on a job that has aged out of its records. "-" stands in for "no
# dependency" so the positional fields below don't shift.
for spec in "1p5b l3lab/L1-Qwen-1.5B-Exact L1-Qwen-1.5B-Exact ${FQ_CALIB_1P5B:--} 16" \
            "7b l3lab/L1-Qwen-7B-Exact L1-Qwen-7B-Exact ${FQ_CALIB_7B:--} 8"; do
  set -- $spec
  dep=$4; [ "$dep" = "-" ] && dep=""
  for scheme in w8a8kv8 w4a4kv4; do
    submit "l1_$1_fq_${scheme}" 47 "$dep" "${B}/$3-flatquant-${scheme}-budget1024/*.parquet" \
      "bash ${FQ} $2 ${scheme} 1024 ${B} $5"
  done
done
