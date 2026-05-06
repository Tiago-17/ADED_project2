#!/bin/bash
#SBATCH --job-name=test_all_models
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --account=f202500010hpcvlabuminhoa
#SBATCH --partition=normal-arm

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

module load "Python/3.12.3-GCCcore-13.3.0"

PYTHON_ENV="$PROJECT_ROOT/env-spark"
source "$PYTHON_ENV/bin/activate"

python "$SCRIPT_DIR/test_all_models.py"
