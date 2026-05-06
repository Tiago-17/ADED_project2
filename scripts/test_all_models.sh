#!/bin/bash
#SBATCH --job-name=test_all_models
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --account=f202500010hpcvlabuminhoa
#SBATCH --partition=normal-arm

module load "Python/3.12.3-GCCcore-13.3.0"

PYTHON_ENV=env-spark
source ${PYTHON_ENV}/bin/activate

python test_all_models.py
