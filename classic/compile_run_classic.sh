#!/bin/bash

# Check if required arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <number_of_processes> <sweeps> <temperature> [--save-magnetization] [--random-seed] [--save-output]"
    exit 1
fi

NP=$1
SWEEPS=$2
TEMPERATURE=$3
shift 3  # Remove first 3 arguments

# Remaining arguments (flags)
FLAGS="$@"

# Compile the C++ Ising model with MPI
mpic++ -O3 -std=c++17 -o classic_ising classic_ising.cpp

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    
    # Run with MPI
    echo "Running simulation with $NP processes, $SWEEPS sweeps, T=$TEMPERATURE $FLAGS..."
    mpirun --use-hwthread-cpus -np $NP ./classic_ising $SWEEPS $TEMPERATURE $FLAGS
else
    echo "Compilation failed!"
    exit 1
fi
