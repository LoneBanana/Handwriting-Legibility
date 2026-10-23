#!/bin/bash
# Evaluate a checkpoint on one GPU:  CHECKPOINT=runs/htr-large/best.pt sbatch scripts/slurm/eval_htr.sbatch
#SBATCH --job-name=legible-htr-eval
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
CHECKPOINT=${CHECKPOINT:-runs/htr-large/best.pt}
MANIFESTS=${MANIFESTS:-data/iam/test.jsonl}
legible-eval --checkpoint "$CHECKPOINT" --manifest $MANIFESTS --decoder accurate \
  --predictions "${CHECKPOINT%.pt}.predictions.jsonl"
