#!/bin/bash
#SBATCH --job-name=run_llama_servers
#SBATCH --nodes=1
#SBATCH --time=0:30:00
#SBATCH --account=f202500010hpcvlabuminhoa
#SBATCH --partition=normal-arm

set -euo pipefail

# Carrega toolchain consistente (compilador + assembler)
module purge
module load CMake
module load binutils/2.45-GCCcore-15.2.0 || module load binutils

cd llama.cpp

# Compila com CMake
echo "Building llama.cpp with CMake..."

rm -rf build
mkdir -p build

echo "Configuring CMake..."
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON \
    -DLLAMA_BUILD_SERVER=ON

echo "Building..."
cmake --build build --target llama-server -j"$(nproc)"

if [[ ! -x build/bin/llama-server ]]; then
    echo "ERROR: build/bin/llama-server nao foi gerado"
    exit 1
fi

echo "Build completed."
