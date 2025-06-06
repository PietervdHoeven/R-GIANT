#!/bin/bash
#SBATCH --job-name=test_connectome_array
#SBATCH --partition=gpu_a100
#SBATCH --cpus-per-task=70
#SBATCH --mem=480G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

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

BATCH_SIZE=70

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

TASK_ID=${SLURM_ARRAY_TASK_ID}     # now 0…10
# For TASK_ID=0 you want lines 1…72,
# for TASK_ID=1 lines 73…144, etc.
START=$(( TASK_ID * BATCH_SIZE + 1 ))
END=$(( (TASK_ID + 1) * BATCH_SIZE ))

if (( END > 742 )); then
  END=742
fi


# 4) Flood CPUs
echo "[4/6] Launch connectome pipelines"
echo "Launching from line $START up to and including line $END"
wait_for_cpu() {
    while [ "$(jobs -rp | wc -l)" -ge "$BATCH_SIZE" ]; do
        sleep 1
    done
}

LINE_NO=0
while IFS=_ read -r P_ID S_ID; do
    LINE_NO=$(( LINE_NO + 1 ))

    # skip any IDs before our START or after our END
    if (( LINE_NO < START || LINE_NO > END )); then
        continue
    fi

    wait_for_cpu
    echo "Launching: $P_ID $S_ID from line $LINE_NO"
    # echo "--subject $P_ID"
    # echo "--session $S_ID"
    # echo "--data-dir $TMPDIR"
    # echo "--log-dir $TMP_LOGS"
    # echo "--plot-dir $TMP_PLOTS"
    rgiant-cli connectome \
        --participant-id "$P_ID" \
        --session-id "$S_ID" \
        --data-dir "$TMPDIR" \
        --log-dir "$TMP_LOGS" \
        --plot-dir "$TMP_PLOTS" \
        --verbose &
    done < "$TMP_LIST"

wait

# 5) move output data to home

rsync -ah "${TMP_LOGS}/"  "$REPO/logs/"
rsync -ah "${TMP_PLOTS}/" "$REPO/plots/"
rsync -ah "${TMP_MTRX}/"  "$REPO/data/matrices/"