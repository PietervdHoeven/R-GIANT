#!/bin/bash
#SBATCH --job-name=clean_rgiant
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --constraint=scratch-node
#SBATCH --output=slurm_logs/slurm_%j.out
#SBATCH --error=slurm_logs/slurm_%j.err

# ==== 1. SETUP ====

module purge
source $HOME/rgiant-venv/bin/activate

# Working directories
SRC_DATA_DIR=$HOME/R-GIANT/data
SCRATCH_BASE=/scratch-shared/$USER
DEST_DATA_DIR=$HOME/R-GIANT/data_test_clean
INPUT_LIST=$HOME/R-GIANT/rgiant/scripts/id_list.txt

mkdir -p $SCRATCH_BASE/data $SCRATCH_BASE/logs $SCRATCH_BASE/slurm_logs

# ==== 2. COPY INPUT DATA TO SCRATCH ====

echo "Copying input data to scratch..."
cp -r $SRC_DATA_DIR/* $SCRATCH_BASE/data/

# ==== 3. RUN CLEANING PIPELINE IN PARALLEL ====

echo "Launching cleaning jobs..."

for i in $(seq 1 10); do
    LINE=$(sed -n "${i}p" $INPUT_LIST)
    PARTICIPANT_ID=$(echo $LINE | cut -d'_' -f1)
    SESSION_ID=$(echo $LINE | cut -d'_' -f2)

    rgiant-cli clean \
    --participant-id $PARTICIPANT_ID \
    --session-id $SESSION_ID \
    --data-dir $SCRATCH_BASE/data \
    --log-dir $SCRATCH_BASE/logs \
    --verbose &

done

wait
echo "All cleaning jobs completed."

# ==== 4. COPY RESULTS BACK TO HOME ====

echo "Copying results back to home..."
mkdir -p $DEST_DATA_DIR
cp -r $SCRATCH_BASE/data/* $DEST_DATA_DIR/
cp -r $SCRATCH_BASE/logs $HOME/R-GIANT/
cp -r $SCRATCH_BASE/slurm_logs $HOME/R-GIANT/

echo "finished!"