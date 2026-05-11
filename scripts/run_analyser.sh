#!/bin/bash
#SBATCH --job-name=llm_analyser
#SBATCH --partition=normal-arm
#SBATCH --account=f202500010hpcvlabuminhoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:30:00
#SBATCH --output=logs/analyser_%j.out
#SBATCH --error=logs/analyser_%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Define paths
ENV_NAME="env-spark"
ENV_DIR="$PROJECT_ROOT/$ENV_NAME"
MODELS_DIR="$PROJECT_ROOT/models"

echo "Loading modules..."
if command -v module >/dev/null 2>&1; then
    module purge
    module load GCCcore/13.2.0 || module load GCC || true
    module load CMake || true
    module load OpenSSL/3 || module load OpenSSL || true
    module load Python/3.11.5-GCCcore-13.2.0
fi

source "$ENV_DIR/bin/activate"

python "$SCRIPT_DIR/analyze_results.py" \
    --results-dir "$PROJECT_ROOT/results" \
    --all \
    --model-size-gb 4.6 \
    --memory-bw-gbs 120.0