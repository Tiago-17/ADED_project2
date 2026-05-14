#!/bin/bash
#SBATCH --job-name=start_models
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --account=f202500010hpcvlabuminhoa
#SBATCH --partition=normal-arm

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
LLAMA_DIR="$PROJECT_ROOT/llama.cpp"
SERVER_BIN="$LLAMA_DIR/build/bin/llama-server"

if [[ -f /share/env/module_select.sh ]]; then
	source /share/env/module_select.sh
fi

if command -v module >/dev/null 2>&1; then
	module purge
	module load GCCcore/15.2.0 || true
	module load OpenSSL/3 || module load OpenSSL || true
fi

OPENSSL_FALLBACK_DIR="/eb/aarch64/software/OpenSSL/3/lib"

if [[ -d "$OPENSSL_FALLBACK_DIR" ]]; then
	export LD_LIBRARY_PATH="$OPENSSL_FALLBACK_DIR:${LD_LIBRARY_PATH:-}"
fi

if [[ ! -x "$SERVER_BIN" ]]; then
	echo "ERROR: $SERVER_BIN nao existe. Execute primeiro scripts/run_llama.sh"
	exit 1
fi

mkdir -p "$PROJECT_ROOT/logs"

declare -a pids=()

cleanup() {
	if [[ ${#pids[@]} -gt 0 ]]; then
		kill "${pids[@]}" 2>/dev/null || true
		wait "${pids[@]}" 2>/dev/null || true
	fi
}

trap cleanup EXIT INT TERM

wait_for_port() {
	local port="$1"
	local attempts=0

	while ! (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; do
		sleep 1
		attempts=$((attempts + 1))
		if [[ $attempts -ge 300 ]]; then
			echo "ERROR: timeout a esperar pelo porto $port"
			exit 1
		fi
	done
	exec 3>&-
	exec 3<&-
}

start_server() {
	local model_path="$1"
	local port="$2"
	local log_file="$3"

	echo "Starting $(basename "$model_path") on port $port"
	"$SERVER_BIN" \
		-m "$model_path" \
		--port "$port" \
		--host 0.0.0.0 \
		>"$log_file" 2>&1 &
	pids+=("$!")
}

start_server "$PROJECT_ROOT/models/qwen2.5-0.5b-instruct-q4_k_m.gguf" 8080 "$PROJECT_ROOT/logs/qwen2.5-0.5b.log"
wait_for_port 8080
start_server "$PROJECT_ROOT/models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf" 8081 "$PROJECT_ROOT/logs/tinyllama-1.1b.log"
wait_for_port 8081
start_server "$PROJECT_ROOT/models/meta-llama-3.1-8b-instruct-q4_k_m.gguf" 8082 "$PROJECT_ROOT/logs/meta-llama-3.1-8b.log"
wait_for_port 8082

PYTHON_ENV="$PROJECT_ROOT/env-spark"
source "$PYTHON_ENV/bin/activate"

python "$SCRIPT_DIR/test_all_models.py"
