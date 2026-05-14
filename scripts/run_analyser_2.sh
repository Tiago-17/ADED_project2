#!/bin/bash
#SBATCH --job-name=llm_analyser_2
#SBATCH --partition=normal-arm
#SBATCH --account=f202500010hpcvlabuminhoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:05:00
#SBATCH --output=logs/llm_analyser_2_%j.out
#SBATCH --error=logs/llm_analyser_2_%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Define paths
ENV_NAME="env-spark"
ENV_DIR="$PROJECT_ROOT/$ENV_NAME"

echo "Loading modules..."
if command -v module >/dev/null 2>&1; then
    module purge
    module load GCCcore/13.2.0 || module load GCC || true
    module load CMake || true
    module load OpenSSL/3 || module load OpenSSL || true
    module load Python/3.11.5-GCCcore-13.2.0
fi

source "$ENV_DIR/bin/activate"

cd "$PROJECT_ROOT" 

echo "A iniciar o cálculo..."
python "$SCRIPT_DIR/analyze_results_2.py"
echo "Concluído!"