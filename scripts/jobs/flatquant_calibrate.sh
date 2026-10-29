#!/usr/bin/env bash
# Calibrate FlatQuant W8A8KV8 and W4A4KV4 for one Qwen2-architecture model,
# with the same settings as the DeepSeek-R1-Distill-Qwen-1.5B calibration
# (quantize/flatquant/outputs/DeepSeek-R1-Distill-Qwen-1.5B/*/exp/log_rank0_*):
# 128 WikiText-2 samples, 15 epochs, flat_lr 5e-3, LWC + LAC, per-token
# symmetric activations, per-channel symmetric weights, asymmetric K/V with
# group size 128. lm_eval is skipped (the skill evals replace it).
#
# Writes outputs/<model basename>/w{bits}a{bits}/exp/flat_matrices.pth, the
# default path eval/eval_skill_flatquant.py loads for that --model-id.
# Skips a scheme whose flat_matrices.pth already exists.
#
# Usage: flatquant_calibrate.sh <HF_MODEL_ID>
set -uo pipefail

# The login environment points HF_DATASETS_CACHE at /home/abalaji/..., which
# does not exist on Perlmutter; datasets.load_dataset fails creating it.
export HF_DATASETS_CACHE=/pscratch/sd/a/abalaji/hf_cache/datasets

MODEL_ID="${1:?HF_MODEL_ID required}"
MODEL_NAME="${MODEL_ID##*/}"
failed=0

cd /global/u2/a/abalaji/projects/agpt/adaptive-compute/quantize/flatquant

for bits in 8 4; do
  matrices="outputs/${MODEL_NAME}/w${bits}a${bits}/exp/flat_matrices.pth"
  if [ -f "$matrices" ]; then
    echo "Skipping ${MODEL_NAME} W${bits}A${bits}KV${bits} (${matrices} exists)"
    continue
  fi
  echo ""
  echo "======================================================================"
  echo "  FlatQuant calibration | ${MODEL_ID} | W${bits}A${bits}KV${bits} | $(date)"
  echo "======================================================================"
  PYTHONUNBUFFERED=1 ~/.conda/envs/flatquant/bin/python main.py \
    --model "$MODEL_ID" \
    --w_bits "$bits" --a_bits "$bits" \
    --k_bits "$bits" --k_asym --k_groupsize 128 \
    --v_bits "$bits" --v_asym --v_groupsize 128 \
    --cali_dataset wikitext2 --nsamples 128 --cali_bsz 4 \
    --epochs 15 --flat_lr 5e-3 \
    --cali_trans --add_diag --lwc --lac \
    --seed 0 \
    --output_dir ./outputs --exp_name exp --save_matrix
  status=$?
  if [ $status -ne 0 ]; then
    echo "ERROR: ${MODEL_NAME} W${bits}A${bits}KV${bits} calibration failed with exit code ${status}"
    failed=1
  fi
done

echo ""
echo "=== ${MODEL_NAME} FlatQuant calibration finished at $(date) (failed=${failed}) ==="
# Nonzero exit so dependent eval jobs (--dependency=afterok) do not start.
exit $failed
