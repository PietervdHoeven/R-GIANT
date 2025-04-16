#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --time=00:30:00
#SBATCH --output=array_%A_%a.out
#SBATCH --error=array_%A_%a.err
#SBATCH --array=1-10    # Replace <N> with the total number of lines in your ids_list.txt

# Optional: Load modules if needed. For example, if your venv requires a module:
# module load python/3.9.5

# Activate your virtual environment (adjust the path as needed)
source /path/to/rgiant-venv/bin/activate

# Retrieve the corresponding line from ids_list.txt using SLURM_ARRAY_TASK_ID
ID_LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ids_list.txt)

# Extract patient and session IDs from the line (assuming the format patient_session)
p_id=$(echo "$ID_LINE" | cut -d'_' -f1)
s_id=$(echo "$ID_LINE" | cut -d'_' -f2)

# Execute the CLI command using srun
srun rgiant-cli clean \
  --patient_id $p_id \
  --session_id $s_id \
  --data-dir R-GIANT/data \
  --log-dir logs \
  --verbose