#!/usr/bin/env bash
# Quantize deepseek-ai/DeepSeek-R1-Distill-Qwen-7B for every quantization
# level already supported/used elsewhere in this repo for the 1.5B models:
# GPTQ-int8, GPTQ-int4, GPTQ-int3, and AWQ-W4A16_ASYM. (FlatQuant is
# excluded -- it needs its own vendored calibration/rotation-matrix
# pipeline that has only ever been run for the 1.5B DeepSeek-R1 model, a
# separate and heavier lift than the checkpoint-producing scripts here.)
#
# GPTQ-int8 may already be quantizing/done from an earlier interactive
# run -- --skip-existing-style guard below (checking for the output dir)
# means this job won't redo it if so.
#
# Same -C gpu / -q shared reasoning as the other batch jobs in
# results/precision_sweep/ -- this account only accepts sbatch via -C gpu,
# and -q shared queues much faster than -q regular here.
#
#SBATCH -J quantize_7b
#SBATCH -A amsc010_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -G 2
#SBATCH -t 04:00:00
#SBATCH -o /pscratch/sd/a/abalaji/precision_sweep_results_data/sbatch_quantize_7b_%j.log

cd /global/u2/a/abalaji/projects/agpt/adaptive-compute/quantize

MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

echo "=== job ${SLURM_JOB_ID} started on $(hostname) at $(date) ==="

quantize_gptq() {
  local bits="$1" outdir="$2"
  if [ -f "${outdir}/config.json" ]; then
    echo "Skipping GPTQ-int${bits} (already exists at ${outdir})"
    return 0
  fi
  echo "=== GPTQ-int${bits} starting at $(date) ==="
  ~/.conda/envs/reason/bin/python quantize_model.py \
    --model-id "$MODEL_ID" --bits "$bits" \
    --calib-dataset safechain --calib-samples 256 --seed 42 \
    --output-dir "$outdir"
  echo "=== GPTQ-int${bits} finished at $(date) (exit $?) ==="
}

quantize_awq() {
  local outdir="$1"
  if [ -f "${outdir}/config.json" ]; then
    echo "Skipping AWQ-W4A16_ASYM (already exists at ${outdir})"
    return 0
  fi
  echo "=== AWQ-W4A16_ASYM starting at $(date) ==="
  ~/.conda/envs/reason/bin/python quantize_model_awq.py \
    --model-id "$MODEL_ID" --scheme W4A16_ASYM \
    --calib-dataset safechain --calib-samples 256 --seed 42 \
    --output-dir "$outdir"
  echo "=== AWQ-W4A16_ASYM finished at $(date) (exit $?) ==="
}

# --- Lane 0 (GPU 0): GPTQ-int8, then GPTQ-int4 ---
(
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONUNBUFFERED=1
  export HF_HUB_OFFLINE=0   # 7B model may not be cached yet
  # HF_ENDPOINT defaults to hf-mirror.com in this account's shell env, which
  # just 308-redirects to the real huggingface.co -- huggingface_hub's
  # client doesn't follow that redirect cleanly (FileMetadataError:
  # "Distant resource does not seem to be on huggingface.co"), so point
  # directly at the real host instead (confirmed reachable from compute
  # nodes on this cluster, unlike the ALCF cluster KNOWN_ISSUES.md
  # describes). HF_DATASETS_CACHE also defaults to a broken path
  # (/home/abalaji/...) in this account's shell env that 403s on
  # os.makedirs -- override to the pscratch cache used everywhere else.
  export HF_ENDPOINT=https://huggingface.co
  export HF_DATASETS_CACHE=/pscratch/sd/a/abalaji/hf_cache/datasets
  quantize_gptq 8 "./checkpoints/DeepSeek-R1-Distill-Qwen-7B-gptq-int8"
  quantize_gptq 4 "./checkpoints/DeepSeek-R1-Distill-Qwen-7B-gptq-int4"
) > /pscratch/sd/a/abalaji/precision_sweep_results_data/quantize_7b_lane0.log 2>&1 &
LANE0_PID=$!

# --- Lane 1 (GPU 1): GPTQ-int3, then AWQ-W4A16_ASYM ---
(
  export CUDA_VISIBLE_DEVICES=1
  export PYTHONUNBUFFERED=1
  export HF_HUB_OFFLINE=0
  export HF_ENDPOINT=https://huggingface.co
  export HF_DATASETS_CACHE=/pscratch/sd/a/abalaji/hf_cache/datasets
  quantize_gptq 3 "./checkpoints/DeepSeek-R1-Distill-Qwen-7B-gptq-int3"
  quantize_awq "./checkpoints/DeepSeek-R1-Distill-Qwen-7B-awq-w4a16_asym"
) > /pscratch/sd/a/abalaji/precision_sweep_results_data/quantize_7b_lane1.log 2>&1 &
LANE1_PID=$!

wait $LANE0_PID
wait $LANE1_PID

echo "=== job ${SLURM_JOB_ID} all done at $(date) ==="
