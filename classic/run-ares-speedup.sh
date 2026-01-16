#!/bin/bash -l
#SBATCH -A plgar2025-cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH -p plgrid
#SBATCH -t 04:00:00
#SBATCH --job-name=speedup_experiment
#SBATCH --output=speedup_experiment_%j.out

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

# Experiment 1: Speedup analysis
# T=2.0, 5000 sweeps, varying cores, no repetitions
cores=(1 2 4 8 16 32)
SWEEPS=5000
TEMPERATURE=2.0

echo "Starting speedup experiment..."
for c in "${cores[@]}"; do
  echo "Running with $c cores..."
  mpiexec -n $c ./$EXECUTABLE $SWEEPS $TEMPERATURE
done

echo "Speedup experiment completed. Results saved to $OUTPUT_FILE"
