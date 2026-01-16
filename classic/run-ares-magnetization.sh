#!/bin/bash -l
#SBATCH -A plgar2025-cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH -p plgrid
#SBATCH -t 12:00:00
#SBATCH --job-name=magnetization_experiment
#SBATCH --output=magnetization_experiment_%j.out

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

# Experiment 2: Magnetization analysis
# T=2.0,2.5,1.0, 20000 sweeps, 32 cores, 5 repetitions, random seed
temperatures=(2.0 2.5 1.0)
SWEEPS=20000
CORES=32
REPETITIONS=5

echo "Starting magnetization experiment..."
for T in "${temperatures[@]}"; do
  echo "Temperature T=$T"
  for i in $(seq 1 $REPETITIONS); do
    echo "  Repetition $i/$REPETITIONS"
    mpiexec -n $CORES ./$EXECUTABLE $SWEEPS $T --save-magnetization --random-seed
  done
done

echo "Magnetization experiment completed. Results saved to $OUTPUT_FILE"
