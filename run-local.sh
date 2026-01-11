PYTHON_FILE="classic_ising.py"
OUTPUT_FILE="results.csv"

# Create or overwrite results file with header
echo "n_proc;time" > "$OUTPUT_FILE"

# Define parameter sets
cores=(1 2 4 8 16)

for c in "${cores[@]}"; do
  for i in {1..3}; do
    mpiexec --use-hwthread-cpus -n $c python $PYTHON_FILE 
  done
done

echo "All runs completed. Results saved to $OUTPUT_FILE"
