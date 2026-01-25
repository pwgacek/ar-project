#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --partition=plgrid
#SBATCH --account=plgar2025-cpu
#SBATCH --job-name=ising_model_rnd_method
#SBATCH --output=out_ising_random_%j.out

module load gcc/13

# Seeds for different temperatures
declare -a seeds_T_1_0=(901665902 -1395521703 -2099478297 -200519273 1981539244)
declare -a seeds_T_2_0=(616128922 801385867 998172480 1570159009 1579357537)
declare -a seeds_T_2_5=(-1472978566 -1951579887 457145835 132120809 -211868607)

# Temperature array
declare -a T_array=(1.0 2.0 2.5)

# Number of processors array
declare -a n_procs=(1 2 4 8 16 32)

# Compile the program once
echo "Compiling ising.cpp..."
g++ -fopenmp -std=c++23 -O3 ising.cpp -o ising
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi
echo "Compilation successful!"

#########################################
# EXPERIMENT 1: Magnetization studies
# 5 repetitions per temperature, 50k sweeps, 32 procs, save magnetization only
#########################################
echo "==================== EXPERIMENT 1: Magnetization ===================="
EXP1_DIR="experiment_1_magnetization"
mkdir -p "$EXP1_DIR"
cd "$EXP1_DIR"

for T in "${T_array[@]}"; do
    T_DIR="T_${T}"
    mkdir -p "$T_DIR"
    cd "$T_DIR"
    
    echo "Running Experiment 1 for T=${T}..."
    
    # Select seeds based on temperature
    if [ "$T" == "1.0" ]; then
        seeds=("${seeds_T_1_0[@]}")
    elif [ "$T" == "2.0" ]; then
        seeds=("${seeds_T_2_0[@]}")
    else
        seeds=("${seeds_T_2_5[@]}")
    fi
    
    # 5 repetitions with different seeds
    for i in {0..4}; do
        seed=${seeds[$i]}
        echo "  Repetition $((i+1))/5 with seed=$seed"
        
        # Copy input.csv from main directory
        cp ../../input.csv .
        
        # Run: T, sweeps=50000, seed, save_mag=1, save_snapshots=0
        OMP_NUM_THREADS=32 ../../ising "$T" 50000 "$seed" 1 0
        
        # Rename output files to include repetition number
        mv output_last.csv "output_last_rep${i}.csv" 2>/dev/null || true
        mv magnetization_${seed}.txt "magnetization_rep${i}_seed${seed}.txt" 2>/dev/null || true
    done
    
    cd ..
done

cd .. # Back to main directory

#########################################
# EXPERIMENT 2: Snapshot studies
# 1 repetition per temperature, 50k sweeps, 32 procs, save snapshots only
#########################################
echo "==================== EXPERIMENT 2: Snapshots ===================="
EXP2_DIR="experiment_2_snapshots"
mkdir -p "$EXP2_DIR"
cd "$EXP2_DIR"

for T in "${T_array[@]}"; do
    T_DIR="T_${T}"
    mkdir -p "$T_DIR"
    cd "$T_DIR"
    
    echo "Running Experiment 2 for T=${T}..."
    
    # Copy input.csv
    cp ../../input.csv .
    
    # Run with arbitrary seed (using 999), save snapshots only
    # T, sweeps=50000, seed=999, save_mag=0, save_snapshots=1
    OMP_NUM_THREADS=32 ../../ising "$T" 50000 999 0 1
    
    cd ..
done

cd .. # Back to main directory

#########################################
# EXPERIMENT 3: Scaling study
# All n_procs, T=2.0, 5k sweeps, no saving
#########################################
echo "==================== EXPERIMENT 3: Scaling ===================="
EXP3_DIR="experiment_3_scaling"
mkdir -p "$EXP3_DIR"
cd "$EXP3_DIR"

# Copy input.csv once
cp ../input.csv .

for nproc in "${n_procs[@]}"; do
    echo "Running Experiment 3 with n_procs=${nproc}..."
    
    # Run: T=2.0, sweeps=5000, seed=123, save_mag=0, save_snapshots=0
    OMP_NUM_THREADS=$nproc ../ising 2.0 5000 123 0 0
    
    # Rename metrics to avoid overwriting
    if [ -f "metrics.csv" ]; then
        if [ $nproc -eq 1 ]; then
            # Keep header for first run
            cp metrics.csv metrics_backup.csv
        else
            # Append without header
            tail -n 1 metrics.csv >> metrics_backup.csv
        fi
    fi
done

# Replace with accumulated metrics
mv metrics_backup.csv metrics.csv 2>/dev/null || true

cd .. # Back to main directory

echo "==================== ALL EXPERIMENTS COMPLETED ===================="