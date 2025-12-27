#!/bin/bash -l
#SBATCH -A plgar2025-cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=33
#SBATCH -p plgrid
#SBATCH -t 04:00:00
#SBATCH --job-name=branch_and_bound_scaling
#SBATCH --output=branch_and_bound_scaling_%j.out

# Load necessary modules
module load scipy-bundle/2021.10-intel-2021b


export I_MPI_SPAWN=on

PYTHON_FILE="classic_ising_futures.py"
OUTPUT_FILE="results.csv"

# Create or overwrite results file with header
echo "n_proc;time" > "$OUTPUT_FILE"

# Define parameter sets
cores=(1 2 4 8 16 32)

for c in "${cores[@]}"; do
  n_procs=$((c + 1))  # Add 1 for master process
  mpiexec -n $n_procs python -m mpi4py.futures $PYTHON_FILE
done

echo "All runs completed. Results saved to $OUTPUT_FILE"
