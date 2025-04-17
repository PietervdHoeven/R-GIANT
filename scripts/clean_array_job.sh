#!/bin/bash
#SBATCH --job-name=clean_rgiant
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --constraint=scratch-node
#SBATCH --array=0-2                     # ← Adjust this range to match your list
#SBATCH --output=slurm_logs/slurm_%A_%a.out
#SBATCH --error=slurm_logs/slurm_%A_%a.err

set -euo pipefail

# ==== 1. SETUP ====

source $HOME/rgiant-venv/bin/activate

INPUT_LIST=$HOME/R-GIANT/rgiant/scripts/id_list.txt
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$INPUT_LIST" | tr -d '\r')

PARTICIPANT_ID=$(echo "$LINE" | cut -d'_' -f1)
SESSION_ID=$(echo "$LINE" | cut -d'_' -f2)

echo "Cleaning $PARTICIPANT_ID - $SESSION_ID"

LOG_DIR="$TEMPDIR/logs"
mkdir -p "$TEMPDIR/mr" "$TEMPDIR/fs" "$TEMPDIR/temp" "$TEMPDIR/clean" "$LOG_DIR"

# ==== 2. COPY INPUT DATA TO SCRATCH ====

rsync -ah "$HOME/R-GIANT/data/mr/${PARTICIPANT_ID}_${SESSION_ID}/" "$TEMPDIR/mr/${PARTICIPANT_ID}_${SESSION_ID}/"
rsync -ah "$HOME/R-GIANT/data/fs/${PARTICIPANT_ID}_${SESSION_ID}/" "$TEMPDIR/fs/${PARTICIPANT_ID}_${SESSION_ID}/"

# ==== 3. RUN PIPELINE ====

rgiant-cli clean \
  --participant-id "$PARTICIPANT_ID" \
  --session-id "$SESSION_ID" \
  --data-dir "$TEMPDIR" \
  --log-dir "$LOG_DIR" \
  --verbose

# ==== 4. COPY OUTPUT BACK TO HOME ====

DEST_CLEAN="$HOME/R-GIANT/data_test/${PARTICIPANT_ID}_${SESSION_ID}"
mkdir -p "$DEST_CLEAN"

rsync -ah "$TEMPDIR/clean/${PARTICIPANT_ID}_${SESSION_ID}/" "$DEST_CLEAN/"
rsync -ah "$LOG_DIR/" "$HOME/R-GIANT/logs/"

echo "Finished $PARTICIPANT_ID - $SESSION_ID"
