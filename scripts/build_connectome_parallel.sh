#!/bin/bash
#SBATCH --job-name=build_connectome
#SBATCH --partition=gpu_a100
#SBATCH --cpus-per-task=72
#SBATCH --mem=480G
#SBATCH --gres=gpu:1
#SBATCH --time=03:30:00
#SBATCH --output=logs/slurm_%A.out
#SBATCH --error=logs/slurm_%A.err

set -euo pipefail

# 1) Activate your environment
echo "[1/8] Activating environment"
source $HOME/rgiant-venv/bin/activate

# 2) Create directories
echo "[2/8] Creating directories"
REPO=$HOME/R-GIANT
IN_LIST=$REPO/scripts/id_list.txt
IN_DATA_CLEAN=$REPO/data/clean
TMP_LOGS=$TMPDIR/logs
TMP_PLOTS=$TMPDIR/plots
TMP_MTRX=$TMPDIR/matrices
TMP_CLEAN=$TMPDIR/clean
TMP_LIST=$TMPDIR/id_list.txt

MAX_JOBS=72

# make all the scratch dirs we’ll use
mkdir -p \
    "$TMP_LOGS" \
    "$TMP_PLOTS" \
    "$TMP_MTRX" \
    "$TMP_CLEAN"

# 3) Copy data to scratch
echo "[3/8] Copying id_list and clean data to scratch"

# copy the single file to exactly TMP_LIST
rsync -ah "$IN_LIST" "$TMP_LIST"

# copy everything _inside_ $IN_DATA_CLEAN into $TMP_CLEAN/
# (trailing slash on source = “contents only”)
rsync -ah "${IN_DATA_CLEAN}/" "$TMP_CLEAN/"

# 4) Flood CPUs
echo "[4/6] Launch connectome pipelines"
wait_for_cpu() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 1
    done
}

# launch up to MAX_JOBS pipelines, then stop
COUNTER=0
while IFS=_ read -r P_ID S_ID; do
    if [ "$COUNTER" -ge "$MAX_JOBS" ]; then
        echo "Reached maximum of $MAX_JOBS jobs, not adding any more jobs."
        break
    fi

    wait_for_cpu
    echo "Launching: $P_ID $S_ID"
    rgiant-cli connectome \
        --participant-id "$P_ID" \
        --session-id "$S_ID" \
        --data-dir "$TMPDIR" \
        --log-dir "$TMP_LOGS" \
        --plot-dir "$TMP_PLOTS" \
        --verbose &
    COUNTER=$((COUNTER + 1))
done < "$TMP_LIST"

wait

# 5) move output data to home

rsync -ah "${TMP_LOGS}/"  "$REPO/logs/"
rsync -ah "${TMP_PLOTS}/" "$REPO/plots/"
rsync -ah "${TMP_MTRX}/"  "$REPO/data/matrices/"