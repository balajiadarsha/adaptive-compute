#!/usr/bin/env bash
# Submit both model sweep jobs to PBS.
# Usage: bash submit_all.sh

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting DeepSeek-R1-Distill-Qwen-1.5B sweep..."
JOB1=$(qsub "${SCRIPTS_DIR}/sweep_deepseek_r1.sh")
echo "  Job ID: ${JOB1}"

echo "Submitting DeepScaleR-1.5B-Preview sweep..."
JOB2=$(qsub "${SCRIPTS_DIR}/sweep_deepscaler.sh")
echo "  Job ID: ${JOB2}"

echo ""
echo "Both jobs submitted."
echo "Monitor with: qstat -u \$USER"
echo "Outputs:"
echo "  /lus/eagle/projects/argonne_tpc/abalaji/adaptive/output/DeepSeek-R1-Distill-Qwen-1.5B/"
echo "  /lus/eagle/projects/argonne_tpc/abalaji/adaptive/output/DeepScaleR-1.5B-Preview/"
