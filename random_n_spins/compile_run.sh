#!/bin/bash

if [ "$#" -ne 7 ]; then
    echo "Usage: $0 <program_name> <num_threads> <temperature> <sweeps> <seed> <save_mag> <save_snapshots>"
    echo "  save_mag: 0=false, 1=true"
    echo "  save_snapshots: 0=false, 1=true"
    exit 1
fi

PROGRAM="$1"
THREADS="$2"
TEMP="$3"
SWEEPS="$4"
SEED="$5"
SAVE_MAG="$6"
SAVE_SNAPSHOTS="$7"

SOURCE="${PROGRAM}.cpp"

g++ -fopenmp -std=c++23 "$SOURCE" -o "$PROGRAM"

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

OMP_NUM_THREADS="$THREADS" ./"$PROGRAM" "$TEMP" "$SWEEPS" "$SEED" "$SAVE_MAG" "$SAVE_SNAPSHOTS"