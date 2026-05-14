# LLM Inference Benchmark - ADED Project2 Track A1

## Quick Start

### Prerequisites

1. **Setup Project** (one-time):
   ```bash
   sbatch scripts/setup_project.sh
   ```
   
   This will:
   - Create Python virtual environment (`env-spark`)
   - Install all dependencies from `requirements_arm.txt`
   - Download all required GGUF models from HuggingFace

---

## Running Experiments

### Experiment 1: Complete Benchmark Suite (All Dimensions)

Runs all experiments including threading, quantization, and multiple models:

```bash
sbatch scripts/run_benchmark.sh
```

**What it does**:
- Thread sweep for llama-3.1-8b (4, 8, 16, 32, 48 threads)
- Other models (qwen2.5-0.5b, tinyllama-1.1b) at 32 threads
- Quantization comparison (Q4_K_M vs Q8_0) at 32 threads
- Generates plots and performance model analysis
- Results stored in `results/<model-name>/`

---

### Experiment 2: Thread Scaling Analysis (Threading Dimension)

Measure how performance varies with thread count (4, 8, 16, 32, 48):

```bash
sbatch scripts/run_benchmark.sh
```

This is part of the complete suite. Individual runs are orchestrated by the main script.

**Output directories**:
```
results/llama-3.1-8b-threads-4/statistics.json
results/llama-3.1-8b-threads-8/statistics.json
results/llama-3.1-8b-threads-16/statistics.json
results/llama-3.1-8b-threads-32/statistics.json
results/llama-3.1-8b-threads-48/statistics.json
```

**Key metrics** (per prompt):
- `ttft_mean_ms`: Time to first token (ms)
- `tpot_mean_ms`: Time per output token (ms)
- `throughput_tps`: Tokens per second
- `memory_peak_mb`: Peak memory usage

---

### Experiment 3: Decode Length Sensitivity Study

Measure TPOT stability across different max generation lengths (64, 128, 256, 512 tokens):

```bash
sbatch scripts/run_decode_length_study.sh
```

**What it does**:
- Runs llama-3.1-8b with 32 threads (fastest config)
- Varies `--max-tokens` parameter: 64, 128, 256, 512
- 3 trials per configuration for statistical robustness
- Generates TPOT stability plots
- Results stored in `results/llama-3.1-8b-max_tokens_<SIZE>/`

**Output directories**:
```
results/llama-3.1-8b-max_tokens_64/statistics.json
results/llama-3.1-8b-max_tokens_128/statistics.json
results/llama-3.1-8b-max_tokens_256/statistics.json
results/llama-3.1-8b-max_tokens_512/statistics.json
```

---

### Experiment 4: Model Comparison (Multi-Model Analysis)

Part of the complete suite. Compares llama-3.1-8b, qwen2.5-0.5b, and tinyllama-1.1b:

```bash
sbatch scripts/run_benchmark.sh
```

**Models tested**:
- Meta-Llama-3.1-8B-Instruct (Q4_K_M)
- Qwen2.5-0.5B-Instruct (Q4_K_M)
- TinyLlama-1.1B (Q4_K_M)

**Configuration**: 32 threads, 3 trials per model

---

## Analyzing Results

### Generate Plots and Analysis

```bash
python scripts/analyze_results.py \
    --results-dir results \
    --all \
    --model-size-gb 4.6 \
    --memory-bw-gbs 1.0
```

**Output plots** (in `results/plots/`):
- `*_ttft_vs_threads.png`: Time to first token scaling
- `*_throughput_vs_threads.png`: Throughput vs. thread count
- `*_tpot_vs_threads.png`: Token generation latency scaling
- `*_performance_model.png`: Predicted vs. observed TPOT
- `*_resource_dashboard.png`: Memory and CPU utilization
- `*_memory_vs_threads.png`: Peak memory scaling

### Analyze Single Model

```bash
python scripts/analyze_results.py \
    --results-dir results \
    --model llama-3.1-8b-threads-32 \
    --model-size-gb 4.6 \
    --memory-bw-gbs 1.0
```

---

## File Structure

```
.
├── README.md                          # This file
├── requirements_arm.txt               # Python dependencies for ARM
├── models/                            # GGUF model files (download manually)
├── prompts/
│   └── benchmark_prompts.json         # 30 benchmark prompts (short/medium/long)
├── scripts/
│   ├── run_benchmark.sh               # Main SLURM job: all experiments
│   ├── run_decode_length_study.sh     # Decode length sensitivity study
│   ├── run_llama.sh                   # Build llama.cpp (auto-called)
│   ├── benchmark_llm.py               # Core benchmark client (30 prompts, 3 trials)
│   ├── monitor_resources.py           # Resource monitoring (CPU, memory)
│   └── analyze_results.py             # Plot generation and analysis
├── results/                           # Benchmark outputs
│   ├── llama-3.1-8b-threads-4/
│   │   ├── statistics.json            # Aggregated metrics per prompt
│   │   ├── raw_results.json           # Individual request data
│   │   └── llama-3.1-8b-threads-4_*.csv  # CPU/memory traces
│   └── plots/                         # Generated analysis plots
└── logs/                              # SLURM job logs and server logs
```
