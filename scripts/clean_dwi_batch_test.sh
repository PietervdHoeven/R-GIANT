#!/bin/bash
#SBATCH --job-name=clean_rgiant
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/slurm_%j.out
#SBATCH --error=slurm_logs/slurm_%j.err

# ==== 1. SETUP ====

module purge
source $HOME/rgiant-venv/bin/activate

# Working directories
SRC_DATA_DIR=$HOME/R-GIANT/data
DEST_DATA_DIR=$HOME/R-GIANT/data_test_clean
INPUT_LIST=$HOME/R-GIANT/scripts/id_list.txt

mkdir -p $TMPDIR/logs

# ==== 2. COPY INPUT DATA TO SCRATCH ====

echo "Copying input data to scratch..."
cp -r "$SRC_DATA_DIR" "$TMPDIR"

# ==== 3. RUN CLEANING PIPELINE IN PARALLEL ====

echo "Launching cleaning jobs..."

for i in $(seq 1 10); do
    LINE=$(sed -n "${i}p" "$INPUT_LIST" | tr -d '\r')
    PARTICIPANT_ID=$(echo $LINE | cut -d'_' -f1)
    SESSION_ID=$(echo $LINE | cut -d'_' -f2)

    echo "Parsed: [$PARTICIPANT_ID] - [$SESSION_ID]"

    rgiant-cli clean \
    --participant-id $PARTICIPANT_ID \
    --session-id $SESSION_ID \
    --data-dir $TMPDIR/data \
    --log-dir $TMPDIR/logs \
    --verbose &

done

wait
echo "All cleaning jobs completed."

# ==== 4. COPY RESULTS BACK TO HOME ====

echo "Copying results back to home..."
mkdir -p $DEST_DATA_DIR
cp -r $TMPDIR/data/* $DEST_DATA_DIR/
cp -r $TMPDIR/logs $HOME/R-GIANT/

echo "finished!"