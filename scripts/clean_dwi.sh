#!/bin/bash
#SBATCH --job-name=clean_rgiant
#SBATCH --partition=gpu_mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=$HOME/R-GIANT/logs/slurm_%A_%a.out
#SBATCH --error =$HOME/R-GIANT/logs/slurm_%A_%a.err

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
echo "    → Participant: $PARTICIPANT_ID, Session: $SESSION_ID"

# 3) Prepare directory structure under $TMPDIR
echo "[3/6] Preparing scratch structure in \$TMPDIR"
mkdir -p \
    $TMPDIR/mr/${PARTICIPANT_ID}_${SESSION_ID} \
    $TMPDIR/fs/${PARTICIPANT_ID}_${SESSION_ID} \
    $TMPDIR/temp/${PARTICIPANT_ID}_${SESSION_ID} \
    $TMPDIR/clean/${PARTICIPANT_ID}_${SESSION_ID} \
    $TMPDIR/logs

# 4) Copy input data to scratch
echo "[4/6] Copying mr/ and fs/ data to scratch"
rsync -ah \
  $HOME/R-GIANT/data/mr/${PARTICIPANT_ID}_${SESSION_ID}/ \
  $TMPDIR/mr/${PARTICIPANT_ID}_${SESSION_ID}/
rsync -ah \
  $HOME/R-GIANT/data/fs/${PARTICIPANT_ID}_${SESSION_ID}/ \
  $TMPDIR/fs/${PARTICIPANT_ID}_${SESSION_ID}/

# 5) Run the cleaning pipeline
echo "[5/6] Launching cleaning pipeline"
rgiant-cli clean \
  --participant-id "$PARTICIPANT_ID" \
  --session-id    "$SESSION_ID"    \
  --data-dir      "$TMPDIR"        \
  --log-dir       "$TMPDIR/logs"   \
  --verbose

# 6) Copy results back to home
echo "[6/6] Copying temp/ and clean/ back to home"
DEST_TEMP=$HOME/R-GIANT/data/temp/${PARTICIPANT_ID}_${SESSION_ID}
DEST_CLEAN=$HOME/R-GIANT/data/clean/${PARTICIPANT_ID}_${SESSION_ID}
mkdir -p "$DEST_TEMP" "$DEST_CLEAN"
rsync -ah $TMPDIR/temp/${PARTICIPANT_ID}_${SESSION_ID}/ "$DEST_TEMP/"
rsync -ah $TMPDIR/clean/${PARTICIPANT_ID}_${SESSION_ID}/ "$DEST_CLEAN/"

echo "Done with $PARTICIPANT_ID | $SESSION_ID"