#!/bin/bash
#SBATCH --job-name=llm_master_bench
#SBATCH --partition=normal-arm
#SBATCH --account=f202500010hpcvlabuminhoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --output=logs/master_bench_%j.out
#SBATCH --error=logs/master_bench_%j.err

set -euo pipefail

if [ -d "$SLURM_SUBMIT_DIR/llama.cpp" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"
fi
SCRIPT_DIR="$PROJECT_ROOT/scripts"
mkdir -p "$PROJECT_ROOT/results" "$PROJECT_ROOT/logs"

ENV_NAME="env-spark"
ENV_DIR="$PROJECT_ROOT/$ENV_NAME"

echo "================================================="
echo "Starting Master Benchmark Pipeline"
echo "Project Root: $PROJECT_ROOT"
echo "================================================="

echo "Loading modules..."
if command -v module >/dev/null 2>&1; then
    module purge
    module load GCCcore/13.2.0 || module load GCC || true
    module load CMake || true
    module load OpenSSL/3 || module load OpenSSL || true
    module load Python/3.11.5-GCCcore-13.2.0
fi

OPENSSL_FALLBACK_DIR="/eb/aarch64/software/OpenSSL/3/lib"
if [[ -d "$OPENSSL_FALLBACK_DIR" ]]; then
    export LD_LIBRARY_PATH="$OPENSSL_FALLBACK_DIR:${LD_LIBRARY_PATH:-}"
fi

if [ ! -f "$PROJECT_ROOT/env-spark/bin/activate" ]; then
    echo "ERROR: Virtual environment not found. Did you run setup_project.sh?"
    exit 1
fi
source "$PROJECT_ROOT/env-spark/bin/activate"

SERVER_BIN="$PROJECT_ROOT/llama.cpp/build/bin/llama-server"
if [[ ! -x "$SERVER_BIN" ]]; then
    echo "ERROR: $SERVER_BIN not found or not executable. Compile llama.cpp first!"
    exit 1
fi

SERVER_PORT=8080

run_experiment() {
    local model_file=$1
    local run_name=$2
    local threads=$3

    local model_path="$PROJECT_ROOT/models/$model_file"

    echo ""
    echo "-------------------------------------------------"
    echo " RUNNING EXPERIMENT: $run_name (Threads: $threads)"
    echo "-------------------------------------------------"

    if [[ ! -f "$model_path" ]]; then
        echo "  WARNING: Model $model_file not found. Skipping $run_name..."
        return 0
    fi

    mkdir -p "$PROJECT_ROOT/results/$run_name"

    # 1. Start Resource Monitor
    python "$SCRIPT_DIR/monitor_resources.py" \
        --output-dir "$PROJECT_ROOT/results/$run_name" \
        --prefix "$run_name" &
    local monitor_pid=$!

    # 2. Start llama-server
    echo "Starting server on port $SERVER_PORT..."
    "$SERVER_BIN" -m "$model_path" -c 4096 -t $threads --port $SERVER_PORT --host 0.0.0.0 \
        > "$PROJECT_ROOT/logs/server_${run_name}.log" 2>&1 &
    local server_pid=$!

    echo "Waiting 20 seconds for server to load the model into memory..."
    sleep 20

    # 3. Verify server didn't crash
    if ! kill -0 $server_pid 2>/dev/null; then
        echo "ERROR: llama-server crashed! Check logs/server_${run_name}.log"
        kill $monitor_pid 2>/dev/null || true
        return 1
    fi

    # 4. Run Python Benchmark Client
    echo "Server is healthy. Starting benchmark client..."
    cd "$PROJECT_ROOT"
    python "$SCRIPT_DIR/benchmark_llm.py" \
        --model "$run_name" \
        --project-root "$PROJECT_ROOT" \
        --threads $threads \
        --port $SERVER_PORT \
        --trials 3 || echo "Benchmark script encountered a non-fatal error."

    # 5. Cleanup
    echo "Cleaning up processes for $run_name..."
    kill $server_pid 2>/dev/null || true
    kill $monitor_pid 2>/dev/null || true
    wait $server_pid 2>/dev/null || true
    wait $monitor_pid 2>/dev/null || true
    
    # Wait to ensure the port is completely freed before the next loop
    sleep 5 
    echo "Finished $run_name"
}


# Test 1: Threading Dimension (Llama 3.1 8B Q4)
THREADS_TO_TEST=(4 8 16 32 48)
for t in "${THREADS_TO_TEST[@]}"; do
    run_experiment "meta-llama-3.1-8b-instruct-q4_k_m.gguf" "llama-3.1-8b-threads-$t" $t
done

# Test 2: Other Models (Qwen & TinyLlama)
run_experiment "qwen2.5-0.5b-instruct-q4_k_m.gguf" "qwen2.5-0.5b" 32
run_experiment "tinyllama-1.1b-chat-v1.0-q4_k_m.gguf" "tinyllama-1.1b" 32


# Test 3: Quantization Dimension (Llama 3.1 8B Q8)
run_experiment "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf" "llama-3.1-8b-Q8" 32

# Analysis & Graph Generation
echo ""
echo "================================================="
echo " GENERATING PLOTS AND PERFORMANCE MODEL"
echo "================================================="

source "$ENV_DIR/bin/activate"

# Assuming ~4.6GB for the baseline Q4_K_M model and ~120GB/s bandwidth for Deucalion ARM
python "$SCRIPT_DIR/analyze_results.py" \
    --results-dir "$PROJECT_ROOT/results" \
    --all \
    --model-size-gb 4.6 \
    --memory-bw-gbs 1024.0

echo "================================================="
echo "ALL BENCHMARKS AND PLOTS COMPLETE!"
echo "Check the results/plots/ directory for your graphs."
echo "================================================="