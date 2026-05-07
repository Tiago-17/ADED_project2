#!/bin/bash
#SBATCH --job-name=llm_benchmark
#SBATCH --partition=normal-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

# ============================================
# LLM Inference Benchmark - Track A1
# ============================================

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Load required modules
module load GCC
module load CMake

# Project directories
MODELS_DIR="$PROJECT_ROOT/models"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
RESULTS_DIR="$PROJECT_ROOT/results"
SERVER_BIN="$PROJECT_ROOT/llama.cpp/build/bin/llama-server"
PYTHON_BIN="$PROJECT_ROOT/env-spark/bin/python"

# Create directories
mkdir -p "$RESULTS_DIR" "$PROJECT_ROOT/logs"

if [[ ! -x "$SERVER_BIN" ]]; then
    if command -v llama-server >/dev/null 2>&1; then
        SERVER_BIN="$(command -v llama-server)"
    else
        echo "ERROR: llama-server nao encontrado. Execute scripts/run_llama.sh primeiro ou garanta que o binario esta no PATH."
        exit 1
    fi
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3"
fi

# Model and configuration
MODEL_FILE="meta-llama-3.1-8b-instruct-q4_k_m.gguf"
MODEL_NAME="llama-3.1-8b"
NUM_THREADS=16
SERVER_PORT=8080

echo "============================================"
echo "Starting benchmark for: $MODEL_NAME"
echo "Threads: $NUM_THREADS"
echo "============================================"

# Step 1: Start llama.cpp server in background
echo "[1/3] Starting llama.cpp server..."
"$SERVER_BIN" \
    -m "$MODELS_DIR/$MODEL_FILE" \
    -c 4096 \
    -t $NUM_THREADS \
    --port $SERVER_PORT \
    --host 0.0.0.0 \
    > "$PROJECT_ROOT/logs/server_${SLURM_JOB_ID}.log" 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Give server time to load the model
sleep 10

# Step 2: Run benchmark
echo "[2/3] Running benchmark..."
cd "$PROJECT_ROOT"

"$PYTHON_BIN" "$SCRIPTS_DIR/benchmark_llm.py" \
    --model "$MODEL_NAME" \
    --project-root "$PROJECT_ROOT" \
    --threads $NUM_THREADS \
    --port $SERVER_PORT \
    --trials 3

BENCHMARK_EXIT_CODE=$?

# Step 3: Cleanup
echo "[3/3] Stopping server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo "============================================"
if [ $BENCHMARK_EXIT_CODE -eq 0 ]; then
    echo "✓ Benchmark completed successfully!"
    echo "Results: $RESULTS_DIR/$MODEL_NAME/"
else
    echo "✗ Benchmark failed with exit code: $BENCHMARK_EXIT_CODE"
fi
echo "============================================"

exit $BENCHMARK_EXIT_CODE