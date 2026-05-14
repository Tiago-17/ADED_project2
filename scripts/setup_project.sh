#!/bin/bash
#SBATCH --job-name=setup_project
#SBATCH --partition=dev-arm
#SBATCH --account=f202500010hpcvlabuminhoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

# Exit immediately if a command exits with a non-zero status
set -euo pipefail

# Get the absolute path to the project root
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Define paths
ENV_NAME="env-spark"
ENV_DIR="$PROJECT_ROOT/$ENV_NAME"
MODELS_DIR="$PROJECT_ROOT/models"

mkdir -p "$PROJECT_ROOT/logs"

echo "================================================="
echo " Setting up Python Virtual Environment"
echo "================================================="

if command -v module >/dev/null 2>&1; then
    echo "Purging existing modules to prevent conflicts..."
    module purge
    
    echo "Loading Python 3.11.5 module..."
    module load Python/3.11.5-GCCcore-13.2.0
fi

if [ ! -d "$ENV_DIR" ]; then
    echo "Creating virtual environment at $ENV_DIR..."
    python3 -m venv "$ENV_DIR"
else
    echo "Virtual environment already exists. Updating packages..."
fi

echo "Activating virtual environment..."
source "$ENV_DIR/bin/activate"

echo "Upgrading pip and installing required packages..."
pip install --upgrade pip
pip install --no-cache-dir -r "$PROJECT_ROOT/requirements_arm.txt"

echo "================================================="
echo " Downloading Models"
echo "================================================="

mkdir -p "$MODELS_DIR"

# Helper function to download only if the file doesn't exist
download_model() {
    local url=$1
    local filename=$2
    
    if [ -f "$MODELS_DIR/$filename" ]; then
        echo "✓ Model $filename already exists. Skipping download."
    else
        echo "⬇ Downloading $filename..."
        # -c continues a partially downloaded file
        # -O specifies the output location
        wget -c "$url" -O "$MODELS_DIR/$filename"
    fi
}

# 1. Llama 3.1 8B (Q4_K_M)
download_model "https://huggingface.co/joshnader/Meta-Llama-3.1-8B-Instruct-Q4_K_M-GGUF/resolve/main/meta-llama-3.1-8b-instruct-q4_k_m.gguf" "meta-llama-3.1-8b-instruct-q4_k_m.gguf"

# 2. Qwen 2.5 0.5B (Q4_K_M)
download_model "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf" "qwen2.5-0.5b-instruct-q4_k_m.gguf"

# 3. TinyLlama 1.1B (Q4_K_M)
download_model "https://huggingface.co/hieupt/TinyLlama-1.1B-Chat-v1.0-Q4_K_M-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf" "tinyllama-1.1b-chat-v1.0-q4_k_m.gguf"

# 4. Llama 3.1 8B (Q8_0)
download_model "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf" "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"

echo "================================================="
echo "Project Setup Complete!"
echo "All models are saved in: $MODELS_DIR"
echo "To run your benchmark, execute:"
echo "sbatch scripts/run_benchmark.sh"
echo "================================================="