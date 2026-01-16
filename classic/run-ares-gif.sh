#!/bin/bash -l
#SBATCH -A plgar2025-cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH -p plgrid
#SBATCH -t 08:00:00
#SBATCH --job-name=gif_experiment
#SBATCH --output=gif_experiment_%j.out

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

# Experiment 3: GIF generation (output snapshots every 1000 sweeps)
# T=2.0,2.5,1.0, 50000 sweeps, 32 cores, no repetitions
# Critical temperature: 2.269
temperatures=(2.0 2.5 1.0)
SWEEPS=50000
CORES=32

echo "Starting GIF generation experiment..."
for T in "${temperatures[@]}"; do
  echo "Running simulation for T=$T (50 snapshots every 1000 sweeps)"
  mpiexec -n $CORES ./$EXECUTABLE $SWEEPS $T --save-output
done

echo "GIF experiment completed. Results saved to $OUTPUT_FILE"
echo "Output directories created: output_2.00/, output_2.50/, output_1.00/"
echo "Use plot_output.py to generate GIF animations from these directories"
