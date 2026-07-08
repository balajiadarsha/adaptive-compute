#!/usr/bin/env bash
# Submit all three model sweep jobs (WildGuard dataset) to PBS.
# Usage: bash submit_all_wildguard.sh

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting DeepSeek-R1-Distill-Qwen-1.5B sweep (WildGuard)..."
JOB1=$(qsub "${SCRIPTS_DIR}/sweep_deepseek_r1_wildguard.sh")
echo "  Job ID: ${JOB1}"

echo "Submitting DeepScaleR-1.5B-Preview sweep (WildGuard)..."
JOB2=$(qsub "${SCRIPTS_DIR}/sweep_deepscaler_wildguard.sh")
echo "  Job ID: ${JOB2}"

echo "Submitting L1-Qwen-1.5B-Exact sweep (WildGuard)..."
JOB3=$(qsub "${SCRIPTS_DIR}/sweep_l1_wildguard.sh")
echo "  Job ID: ${JOB3}"

echo ""
echo "All jobs submitted."
echo "Monitor with: qstat -u \$USER"
echo "Outputs:"
echo "  /lus/eagle/projects/argonne_tpc/abalaji/adaptive/output/DeepSeek-R1-Distill-Qwen-1.5B_wildguard/"
echo "  /lus/eagle/projects/argonne_tpc/abalaji/adaptive/output/DeepScaleR-1.5B-Preview_wildguard/"
echo "  /lus/eagle/projects/argonne_tpc/abalaji/adaptive/output/L1-Qwen-1.5B-Exact_wildguard/"
