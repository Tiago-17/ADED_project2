#!/bin/bash
#SBATCH --job-name=llm_decode_length_study
#SBATCH --partition=normal-arm
#SBATCH --account=f202500010hpcvlabuminhoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --output=logs/decode_length_study_%j.out
#SBATCH --error=logs/decode_length_study_%j.err

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
echo "Starting Decode Length Sensitivity Study"
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

if [[ -f "$ENV_DIR/bin/activate" ]]; then
    source "$ENV_DIR/bin/activate"
else
    echo "⚠️  WARNING: Environment not found at $ENV_DIR"
fi

echo "Building llama.cpp if needed..."
"$SCRIPT_DIR/run_llama.sh" > /dev/null 2>&1 || echo "⚠️  llama.cpp build skipped or already built"

SERVER_BIN="$PROJECT_ROOT/llama.cpp/build/bin/llama-server"
if [[ ! -x "$SERVER_BIN" ]]; then
    echo "ERROR: $SERVER_BIN not found or not executable. Compile llama.cpp first!"
    exit 1
fi

SERVER_PORT=8080
MODEL_FILE="meta-llama-3.1-8b-instruct-q4_k_m.gguf"
MODEL_PATH="$PROJECT_ROOT/models/$MODEL_FILE"
THREADS=32

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

run_experiment_with_max_tokens() {
    local max_tokens=$1
    local run_name="llama-3.1-8b-max_tokens_${max_tokens}"

    echo ""
    echo "-------------------------------------------------"
    echo "🚀 RUNNING: $run_name (max_tokens=$max_tokens, threads=$THREADS)"
    echo "-------------------------------------------------"

    mkdir -p "$PROJECT_ROOT/results/$run_name"

    # 1. Start Resource Monitor
    python "$SCRIPT_DIR/monitor_resources.py" \
        --output-dir "$PROJECT_ROOT/results/$run_name" \
        --prefix "$run_name" &
    local monitor_pid=$!

    # 2. Start llama-server
    echo "Starting server on port $SERVER_PORT..."
    "$SERVER_BIN" -m "$MODEL_PATH" -c 4096 -t $THREADS --port $SERVER_PORT --host 0.0.0.0 \
        > "$PROJECT_ROOT/logs/server_${run_name}.log" 2>&1 &
    local server_pid=$!

    # 3. Wait for server to load
    echo "Waiting 20 seconds for server to load the model into memory..."
    sleep 20

    # 4. Verify server didn't crash
    if ! kill -0 $server_pid 2>/dev/null; then
        echo "❌ ERROR: llama-server crashed! Check logs/server_${run_name}.log"
        kill $monitor_pid 2>/dev/null || true
        return 1
    fi

    # 5. Run Python Benchmark Client with max_tokens parameter
    echo "Server is healthy. Starting benchmark client with max_tokens=$max_tokens..."
    cd "$PROJECT_ROOT"
    python "$SCRIPT_DIR/benchmark_llm.py" \
        --model "$run_name" \
        --project-root "$PROJECT_ROOT" \
        --threads $THREADS \
        --max-tokens $max_tokens \
        --port $SERVER_PORT \
        --trials 3 || echo "⚠️  Benchmark script encountered a non-fatal error."

    # 6. Cleanup
    echo "Cleaning up processes for $run_name..."
    kill $server_pid 2>/dev/null || true
    kill $monitor_pid 2>/dev/null || true
    wait $server_pid 2>/dev/null || true
    wait $monitor_pid 2>/dev/null || true
    
    # Wait to ensure the port is completely freed before the next loop
    sleep 5 
    echo "✅ Finished $run_name"
}

MAX_TOKENS_TO_TEST=(64 128 256 512)
for max_tokens in "${MAX_TOKENS_TO_TEST[@]}"; do
    run_experiment_with_max_tokens $max_tokens
done

echo ""
echo "================================================="
echo "📊 GENERATING ANALYSIS AND PLOTS"
echo "================================================="]

# Collect all decode-length result directories and generate plots
python "$SCRIPT_DIR/analyze_results.py" \
    --results-dir "$PROJECT_ROOT/results" \
    --all \
    --model-size-gb 4.6 \
    --memory-bw-gbs 1.0

echo "================================================="
echo "🎉 DECODE LENGTH STUDY COMPLETE! 🎉"
echo "Check the results/plots/ directory for your graphs."
echo "================================================="
