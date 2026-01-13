#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <program_name> <num_threads>"
    exit 1
fi

PROGRAM="$1"
THREADS="$2"

SOURCE="${PROGRAM}.cpp"

g++ -fopenmp -std=c++23 "$SOURCE" -o "$PROGRAM"

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

OMP_NUM_THREADS="$THREADS" ./"$PROGRAM" 