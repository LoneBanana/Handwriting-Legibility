#!/bin/bash
# Multi-node training of the handwriting model with torchrun on SLURM.
#
#   sbatch --nodes=4 scripts/slurm/train_htr.sbatch                      # configs/htr/large.yaml
#   CONFIG=configs/htr/finetune_dysgraphia.yaml sbatch scripts/slurm/train_htr.sbatch
#   sbatch scripts/slurm/train_htr.sbatch --set data.batch_size=32        # extra args go to the trainer
#
# Preemption/time limits: the trainer stops a few minutes before the job's time limit
# (or on SIGTERM), writes a checkpoint and a NEEDS_RESUME marker, and exits cleanly;
# this script then requeues the job, which resumes exactly where it stopped.
# Adjust the partition/account/GPU lines for your cluster.
#SBATCH --job-name=legible-htr
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

CONFIG=${CONFIG:-configs/htr/large.yaml}
RUN_DIR=$(python -c "from legible.htr.config import load_config; print(load_config('$CONFIG').train.output_dir)")
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}

# Stop with SAFETY_MINUTES to spare so the final checkpoint is written before the hard kill.
SAFETY_MINUTES=${SAFETY_MINUTES:-15}
LIMIT_MINUTES=$(squeue -h -j "$SLURM_JOB_ID" -o %l | awk -F'[-:]' '{
  if (NF==4) print $1*1440+$2*60+$3; else if (NF==3) print $1*60+$2; else if (NF==2) print $1; else print 0 }')
RUNTIME=$(( LIMIT_MINUTES > SAFETY_MINUTES ? LIMIT_MINUTES - SAFETY_MINUTES : 0 ))

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "job $SLURM_JOB_ID: $SLURM_NNODES node(s) x $GPUS_PER_NODE GPU(s), config $CONFIG, run dir $RUN_DIR, runtime ${RUNTIME}m"

srun --kill-on-bad-exit=1 --cpu-bind=none torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc-per-node="$GPUS_PER_NODE" \
  --rdzv-id="$SLURM_JOB_ID" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m legible.htr.train --config "$CONFIG" --set train.max_runtime_minutes="$RUNTIME" train.resume=auto "$@"

if [ -f "$RUN_DIR/NEEDS_RESUME" ]; then
  echo "time limit reached: requeueing $SLURM_JOB_ID to resume"
  scontrol requeue "$SLURM_JOB_ID"
fi
