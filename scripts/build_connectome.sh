#!/bin/bash
#SBATCH --job-name=build_connectome
#SBATCH --partition=gpu_a100
#SBATCH --cpus-per-task=36
#SBATCH --mem=240G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

set -euo pipefail

# 1) Activate your environment
echo "[1/6] Activating environment"
source $HOME/rgiant-venv/bin/activate

# 2) Read the Nth line from id_list.txt
echo "[2/6] Reading ID from line $((SLURM_ARRAY_TASK_ID))"
INPUT_LIST=$HOME/R-GIANT/scripts/id_list.txt
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID))p" "$INPUT_LIST" | tr -d '\r')
PARTICIPANT_ID=$(echo "$LINE" | cut -d'_' -f1)
SESSION_ID=$(echo "$LINE" | cut -d'_' -f2)
echo "Participant: $PARTICIPANT_ID, Session: $SESSION_ID"

# 3) Prepare directory structure under $TMPDIR
echo "[3/6] Preparing scratch structure in \$TMPDIR"
mkdir -p \
    $TMPDIR/matrices/${PARTICIPANT_ID}_${SESSION_ID} \
    $TMPDIR/clean/${PARTICIPANT_ID}_${SESSION_ID} \
    $TMPDIR/logs \
    $TMPDIR/plots

# 4) Copy input data to scratch
echo "[4/6] Copying clean/ data to scratch"
rsync -ah \
  $HOME/R-GIANT/data/clean/${PARTICIPANT_ID}_${SESSION_ID}/ \
  $TMPDIR/clean/${PARTICIPANT_ID}_${SESSION_ID}/

# 5) Run the cleaning pipeline
echo "[5/6] Launching cleaning pipeline"
rgiant-cli connectome \
  --participant-id "$PARTICIPANT_ID" \
  --session-id "$SESSION_ID" \
  --data-dir "$TMPDIR" \
  --log-dir "$TMPDIR/logs" \
  --plot-dir "$TMPDIR/plots" \
  --verbose

# 6) Copy results back to home
echo "[6/6] Copying results back to home"

DEST_MTRX=$HOME/R-GIANT/data/matrices
DEST_LOGS=$HOME/R-GIANT/logs
DEST_PLOTS=$HOME/R-GIANT/plots

# Make sure target directories exist
mkdir -p "$DEST_MTRX" "$DEST_LOGS" "$DEST_PLOTS"

# Copy matrices (overwrite if name exists, preserve other files)
rsync -ah "$TMPDIR/matrices/" "$DEST_MTRX/"

# Copy logs and plots (optional, organize per subject/session)
rsync -ah "$TMPDIR/logs/" "$DEST_LOGS/"
rsync -ah "$TMPDIR/plots/" "$DEST_PLOTS/"

echo "Done with $PARTICIPANT_ID | $SESSION_ID"
