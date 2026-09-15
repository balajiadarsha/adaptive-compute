#!/usr/bin/env bash
# Self-requeuing batch job for the DeepSeek-R1-Distill-Qwen-1.5B precision
# sweep. Intended for when a single interactive salloc window (1hr,
# gpu_interactive QOS) isn't enough to finish the remaining FlatQuant runs.
#
# NOTE: as of 2026-09-14, sbatch submission is rejected for this account
# (amsc010_g) on every GPU QOS tried -- gpu_regular, gpu_preempt,
# gpu_shared, and gpu_debug all return "Job request does not match any
# supported policy". Recent job history (sacct) shows only gpu_interactive
# jobs ever ran, suggesting this is an interactive-only allocation grant.
# Submit with `sbatch resume_sweep.sbatch` -- if it's rejected, batch access
# still isn't available and you'll need to keep resuming interactively
# (or ask NERSC support to enable batch QOS for this account).
#
#SBATCH -J deepseek_r1_precision_sweep
#SBATCH -A amsc010_g
#SBATCH -q gpu_debug
#SBATCH -C gpu
#SBATCH --gpus 1
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --signal=B:USR1@120
#SBATCH -o /global/u2/a/abalaji/projects/agpt/adaptive-compute/eval/results/precision_sweep/sbatch_%j.log

cd /global/u2/a/abalaji/projects/agpt/adaptive-compute/eval

DONE_MARKER=results/precision_sweep/.sweep_complete
CHAIN_COUNT_FILE=results/precision_sweep/.sweep_chain_count
MAX_CHAIN=40   # safety cap so a persistent failure (e.g. an unsupported
               # quantization config) can't resubmit forever

if [ -f "$DONE_MARKER" ]; then
  echo "Sweep already marked complete ($DONE_MARKER exists) -- nothing to do."
  exit 0
fi

count=0
[ -f "$CHAIN_COUNT_FILE" ] && count=$(cat "$CHAIN_COUNT_FILE")

echo "=== chain iteration ${count} (job ${SLURM_JOB_ID}) on $(hostname) ==="
date

# --signal=B:USR1@120 delivers SIGUSR1 ~120s before the walltime limit, so
# we can resubmit before SLURM kills this job outright (a walltime SIGKILL
# would terminate this whole script, leaving nothing to run `sbatch` again).
# Run the sweep in the background + `wait` so the trap fires promptly.
resubmit_and_exit() {
  echo "Caught warning signal near walltime limit -- resubmitting to continue."
  sbatch "$0"
  exit 0
}
trap resubmit_and_exit USR1

PYTHONUNBUFFERED=1 ~/.conda/envs/reason/bin/python run_skill_sweep.py \
  --models deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
           ../quantize/checkpoints/DeepSeek-R1-Distill-Qwen-1.5B-gptq-int8 \
           ../quantize/checkpoints/DeepSeek-R1-Distill-Qwen-1.5B-gptq-int4 \
           ../quantize/checkpoints/DeepSeek-R1-Distill-Qwen-1.5B-awq-w4a16_asym \
  --flatquant-schemes w8a8kv8 w4a4kv4 \
  --datasets aime gpqa math gsm8k \
  --n 50 \
  --tensor-parallel-size 1 \
  --out-dir results/precision_sweep \
  --skip-existing &
wait $!
status=$?

if [ $status -eq 0 ]; then
  echo "Sweep completed successfully."
  touch "$DONE_MARKER"
  exit 0
fi

echo "run_skill_sweep.py exited with status ${status} before the walltime signal fired."
count=$((count + 1))
echo "$count" > "$CHAIN_COUNT_FILE"

if [ "$count" -ge "$MAX_CHAIN" ]; then
  echo "Reached max chain count (${MAX_CHAIN}) without completing -- not resubmitting."
  echo "This likely means a real failure (not just a timeout) -- check the log above."
  exit 1
fi

echo "Resubmitting (chain count ${count}/${MAX_CHAIN})..."
sbatch "$0"
