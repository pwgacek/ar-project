#!/bin/bash -l
#SBATCH -A plgar2025-cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH -p plgrid
#SBATCH -t 04:00:00
#SBATCH --job-name=classic_ising_cpp
#SBATCH --output=classic_ising_cpp_%j.out

# Load necessary modules
module load scipy-bundle/2021.10-intel-2021b

export I_MPI_SPAWN=on

CPP_FILE="classic_ising.cpp"
EXECUTABLE="classic_ising"
OUTPUT_FILE="results.csv"

# Compile the C++ code
echo "Compiling $CPP_FILE..."
mpicc -O3 -std=c++17 $CPP_FILE -o $EXECUTABLE -lstdc++ -lm
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi
echo "Compilation successful."

# Create or overwrite results file with header
echo "n_proc,sweeps,temperature,time" > "$OUTPUT_FILE"

# Define parameter sets
cores=(1 2 4 8 16 32)
SWEEPS=1000
TEMPERATURE=2.0
# MAGNETIZATION_FLAG="--save-magnetization"  # Uncomment to enable magnetization output
MAGNETIZATION_FLAG=""  # Empty by default (disabled)
# RANDOM_SEED_FLAG="--random-seed"  # Uncomment to use random seed instead of fixed seed
RANDOM_SEED_FLAG=""  # Empty by default (uses fixed seed)
# OUTPUT_FLAG="--save-output"  # Uncomment to save grid every 1000 sweeps
OUTPUT_FLAG=""  # Empty by default (disabled)

for c in "${cores[@]}"; do
  for i in {1..3}; do
    mpiexec -n $c ./$EXECUTABLE $SWEEPS $TEMPERATURE $MAGNETIZATION_FLAG $RANDOM_SEED_FLAG $OUTPUT_FLAG
  done
done

echo "All runs completed. Results saved to $OUTPUT_FILE"
