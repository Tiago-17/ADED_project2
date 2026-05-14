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

### Obtain and build `llama.cpp`

This project requires `llama.cpp` (the inference server). We include a script to build the server in this repository.

Build using the project's helper script:

```bash
git submodule update --init --recursive
sbatch scripts/run_llama.sh
```

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

### Generate Plots and Tables

To automatically generate all comparison plots and tables, use the configured SLURM job:

```bash
sbatch scripts/run_analyser_2.sh
```

**Output plots** (in `results/plots/report/`):
- `thread_scaling.png`: Scaling of TTFT, TPOT, Throughput, and Memory based on thread count.
- `model_comparison.png`: Comparison of various metrics across different models (Llama-3.1, Qwen2.5, TinyLlama).
- `quantization_comparison.png`: Effect of quantization (Q4_K_M vs Q8_0) on TPOT and Memory usage.
- `tpot_observed_vs_predicted.png`: Validation of the theoretical performance model against actual results.
- `memory_overview.png`: Horizontal bar chart showing peak memory per model.

**Output tables** (in `results/tables/`):
- The script generates representations in `.csv`, `.md`, and `.typ` (Typst) code so they can be easily included in reports.
- Generated tables: `thread_scaling`, `model_comparison`, `quantization`, `model_summary`, and `tpot_validation`.

---

## File Structure

```
.
├── requirements_arm.txt               # Python dependencies for ARM
├── models/                            # GGUF model files (download manually)
├── prompts/
│   └── benchmark_prompts.json         # 30 benchmark prompts (short/medium/long)
├── scripts/
│   ├── run_benchmark.sh               # Main SLURM job: all experiments
│   ├── run_decode_length_study.sh     # Decode length sensitivity study
│   ├── run_llama.sh                   # Build llama.cpp (auto-called)
│   ├── run_analyser_2.sh              # Run analysis and plot generation
│   ├── benchmark_llm.py               # Core benchmark client (30 prompts, 3 trials)
│   ├── monitor_resources.py           # Resource monitoring (CPU, memory)
│   ├── analyze_results.py             # Legacy plot generation
│   └── analyze_results_2.py           # Core results analyzer (plots & tables)
├── results/                           # Benchmark outputs
│   ├── llama-3.1-8b-threads-4/
│   │   ├── statistics.csv             # Aggregated metrics per prompt
│   │   ├── raw_results.json           # Individual request data
│   │   └── llama-3.1-8b-threads-4_*.csv  # CPU/memory traces
│   ├── plots/
│   │   └── report/                    # Generated analysis plots
│   └── tables/                        # Generated tables (.csv, .md, .typ)
└── logs/                              # SLURM job logs and server logs
```
