#!/usr/bin/env python3
import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-darkgrid")
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

MODEL_LABELS = {
    "llama-3.1-8b-threads-4": "Llama-3.1-8B (4 threads)",
    "llama-3.1-8b-threads-8": "Llama-3.1-8B (8 threads)",
    "llama-3.1-8b-threads-16": "Llama-3.1-8B (16 threads)",
    "llama-3.1-8b-threads-32": "Llama-3.1-8B",
    "llama-3.1-8b-threads-48": "Llama-3.1-8B (48 threads)",
    "llama-3.1-8b-Q8": "Llama-3.1-8B Q8",
    "tinyllama-1.1b": "TinyLlama-1.1B",
    "qwen2.5-0.5b": "Qwen2.5-0.5B",
}

MODEL_SUMMARY = [
    {
        "display_name": "Meta-Llama-3.1-8B",
        "quantization": "Q4_K_M",
        "model_file": "meta-llama-3.1-8b-instruct-q4_k_m.gguf",
        "fallback_size_gb": 4.60,
    },
    {
        "display_name": "TinyLlama-1.1B",
        "quantization": "Q8_0",
        "model_file": "tinyllama-1.1b-chat-v1.0-q4_k_m.gguf",
        "fallback_size_gb": 1.26,
    },
    {
        "display_name": "Qwen2.5-0.5B",
        "quantization": "Q5_K_M",
        "model_file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "fallback_size_gb": 1.08,
    },
]

def safe_mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0

def safe_std(values: Sequence[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0

def format_mean_std(mean_value: float, std_value: float, digits: int = 2) -> str:
    return f"{mean_value:.{digits}f} ± {std_value:.{digits}f}"

def format_signed_percent(value: float, digits: int = 1) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{digits}f}%"

def load_statistics(model_dir: Path) -> List[Dict]:
    stats_file = model_dir / "statistics.csv"
    if not stats_file.exists():
        return []
    with stats_file.open(encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        return list(reader)

def discover_model_dirs(results_dir: Path) -> List[Path]:
    if not results_dir.exists():
        return []
    model_dirs = []
    for path in sorted(results_dir.iterdir()):
        if path.is_dir() and path.name not in {"plots", "tables"} and (path / "statistics.csv").exists():
            model_dirs.append(path)
    return model_dirs

def resolve_results_dir(requested_dir: Path) -> Path:
    requested_dir = requested_dir.resolve()
    current = requested_dir
    while True:
        if discover_model_dirs(current):
            return current
        if current.parent == current:
            return requested_dir
        current = current.parent

def mean_metric(stats: Sequence[Dict], metric: str, category: Optional[str] = None) -> Tuple[float, float]:
    filtered = [row for row in stats if category is None or row.get("category") == category]
    values = [float(row.get(metric, 0) or 0) for row in filtered if float(row.get(metric, 0) or 0) > 0]
    return safe_mean(values), safe_std(values)

def category_summary(stats: Sequence[Dict], category: Optional[str] = None) -> Dict[str, Tuple[float, float]]:
    filtered = [row for row in stats if category is None or row.get("category") == category]
    metric_values: Dict[str, List[float]] = defaultdict(list)
    for row in filtered:
        for metric in ("ttft_mean_ms", "tpot_mean_ms", "decode_tps_mean", "memory_mb_mean"):
            value = float(row.get(metric, 0) or 0)
            if value > 0:
                metric_values[metric].append(value)
    return {metric: (safe_mean(values), safe_std(values)) for metric, values in metric_values.items()}

def label_for_model(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)

def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_thread_scaling(results_dir: Path, output_dir: Path) -> Optional[Path]:
    thread_pattern = re.compile(r"llama-3\.1-8b-threads-(\d+)$")
    thread_rows: List[Tuple[int, List[Dict]]] = []

    for model_dir in discover_model_dirs(results_dir):
        match = thread_pattern.match(model_dir.name)
        if match:
            thread_rows.append((int(match.group(1)), load_statistics(model_dir)))

    thread_rows.sort(key=lambda item: item[0])
    if not thread_rows: return None

    threads = [row[0] for row in thread_rows]
    short_stats = [category_summary(row[1], "short") for row in thread_rows]

    metrics = [
        ("ttft_mean_ms", "TTFT short (s)", lambda value: value / 1000.0),
        ("tpot_mean_ms", "TPOT short (ms)", lambda value: value),
        ("decode_tps_mean", "Throughput short (tok/s)", lambda value: value),
        ("memory_mb_mean", "Peak memory short (MB)", lambda value: value),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for idx, (metric, title, transform) in enumerate(metrics):
        means = []
        stds = []
        for row in short_stats:
            mean, std = row.get(metric, (0.0, 0.0))
            means.append(transform(mean))
            stds.append(transform(std))

        ax = axes[idx]
        ax.errorbar(threads, means, yerr=stds, fmt="o-", linewidth=2, capsize=4, color=COLORS[idx])
        ax.set_xlabel("Threads")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(threads)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Llama-3.1-8B thread scaling", fontsize=15, fontweight="bold")
    return save_figure(fig, output_dir, "thread_scaling.png")

def plot_model_comparison(results_dir: Path, output_dir: Path) -> Optional[Path]:
    preferred_order = ["llama-3.1-8b-threads-32", "tinyllama-1.1b", "qwen2.5-0.5b"]
    available = {path.name: path for path in discover_model_dirs(results_dir)}
    selected = [name for name in preferred_order if name in available]
    if not selected: return None

    categories = ["short", "medium", "long"]
    metrics = [
        ("ttft_mean_ms", "TTFT (s)", lambda value: value / 1000.0),
        ("tpot_mean_ms", "TPOT (ms)", lambda value: value),
        ("decode_tps_mean", "Throughput (tok/s)", lambda value: value),
        ("memory_mb_mean", "Peak memory (MB)", lambda value: value),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    bar_width = 0.8 / max(len(selected), 1)

    for metric_index, (metric, title, transform) in enumerate(metrics):
        ax = axes[metric_index]
        x = np.arange(len(categories))

        for model_index, model_name in enumerate(selected):
            stats = load_statistics(available[model_name])
            means = []
            stds = []
            for category in categories:
                mean, std = mean_metric(stats, metric, category)
                means.append(transform(mean))
                stds.append(transform(std))

            offset = (model_index - (len(selected) - 1) / 2) * bar_width
            ax.bar(
                x + offset, means, bar_width, yerr=stds, capsize=4, alpha=0.85,
                color=COLORS[model_index % len(COLORS)],
                label=label_for_model(model_name) if metric_index == 0 else None,
            )

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([name.capitalize() for name in categories])
        ax.set_xlabel("Prompt category")
        ax.set_ylabel(title)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("Comparison across models", fontsize=15, fontweight="bold")
    return save_figure(fig, output_dir, "model_comparison.png")

def plot_quantization_comparison(results_dir: Path, output_dir: Path) -> Optional[Path]:
    quantization_variants = [("llama-3.1-8b-threads-32", "Q4_K_M"), ("llama-3.1-8b-Q8", "Q8_0")]
    available = {path.name: path for path in discover_model_dirs(results_dir)}

    labels, tpot_means, tpot_stds, memory_means, memory_stds = [], [], [], [], []

    for model_key, quant_label in quantization_variants:
        model_dir = available.get(model_key)
        if model_dir is None: continue

        stats = load_statistics(model_dir)
        tpot_mean, tpot_std = mean_metric(stats, "tpot_mean_ms", "short")
        memory_mean, memory_std = mean_metric(stats, "memory_mb_mean", "short")

        if tpot_mean <= 0 and memory_mean <= 0: continue

        labels.append(quant_label)
        tpot_means.append(tpot_mean)
        tpot_stds.append(tpot_std)
        memory_means.append(memory_mean)
        memory_stds.append(memory_std)

    if len(labels) < 2: return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    x = np.arange(len(labels))

    axes[0].bar(x, tpot_means, yerr=tpot_stds, capsize=4, alpha=0.85, color=[COLORS[0], COLORS[1]])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("TPOT short (ms)")
    axes[0].set_title("TPOT by quantization")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, memory_means, yerr=memory_stds, capsize=4, alpha=0.85, color=[COLORS[2], COLORS[3]])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Peak memory short (MB)")
    axes[1].set_title("Memory by quantization")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Llama-3.1-8B quantization comparison", fontsize=15, fontweight="bold")
    return save_figure(fig, output_dir, "quantization_comparison.png")

def plot_observed_vs_predicted_tpot(
    results_dir: Path, output_dir: Path, model_size_gb: float, effective_bandwidth_gbs: float,
) -> Optional[Path]:
    thread_pattern = re.compile(r"llama-3\.1-8b-threads-(\d+)$")
    thread_rows: List[Tuple[int, List[Dict]]] = []

    for model_dir in discover_model_dirs(results_dir):
        match = thread_pattern.match(model_dir.name)
        if match:
            thread_rows.append((int(match.group(1)), load_statistics(model_dir)))

    thread_rows.sort(key=lambda item: item[0])
    if not thread_rows: return None

    threads = [row[0] for row in thread_rows]
    observed, observed_std = [], []

    for _, stats in thread_rows:
        mean, std = mean_metric(stats, "tpot_mean_ms", "short")
        observed.append(mean)
        observed_std.append(std)

    predicted_single = (model_size_gb / effective_bandwidth_gbs) * 1000.0 if effective_bandwidth_gbs else 0.0
    predicted = [predicted_single for _ in threads]
    relative_error = [((obs - predicted_single) / predicted_single) * 100.0 if predicted_single else 0.0 for obs in observed]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].errorbar(threads, observed, yerr=observed_std, fmt="o", color=COLORS[0], capsize=4, label="Observed")
    axes[0].plot(threads, predicted, linestyle="--", linewidth=2, color=COLORS[1], label=f"Predicted ({predicted_single:.1f} ms)")
    axes[0].set_xlabel("Threads")
    axes[0].set_ylabel("TPOT (ms)")
    axes[0].set_title("Observed vs predicted TPOT")
    axes[0].set_xticks(threads)
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].bar(threads, relative_error, color=COLORS[2], alpha=0.85)
    axes[1].set_xlabel("Threads")
    axes[1].set_ylabel("Relative error (%)")
    axes[1].set_title("Relative error of the model")
    axes[1].set_xticks(threads)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("TPOT performance model", fontsize=15, fontweight="bold")
    return save_figure(fig, output_dir, "tpot_observed_vs_predicted.png")

def plot_memory_overview(results_dir: Path, output_dir: Path) -> Optional[Path]:
    selected = ["llama-3.1-8b-threads-32", "tinyllama-1.1b", "qwen2.5-0.5b"]
    available = {path.name: path for path in discover_model_dirs(results_dir)}
    chosen = [available[name] for name in selected if name in available]
    
    labels, memory_means, memory_stds = [], [], []

    for model_dir in chosen:
        stats = load_statistics(model_dir)
        mean, std = mean_metric(stats, "memory_mb_mean")
        if mean > 0:
            labels.append(label_for_model(model_dir.name))
            memory_means.append(mean)
            memory_stds.append(std)

    if not labels: return None

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(labels))
    ax.barh(y, memory_means, xerr=memory_stds, color=COLORS[: len(labels)], alpha=0.85, capsize=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Peak memory (MB)")
    ax.set_title("Peak memory by model")
    ax.grid(True, axis="x", alpha=0.3)
    return save_figure(fig, output_dir, "memory_overview.png")

def write_csv_table(path: Path, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)

def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator, *body]) + "\n"

def build_thread_scaling_rows(results_dir: Path) -> List[List[str]]:
    thread_pattern = re.compile(r"llama-3\.1-8b-threads-(\d+)$")
    rows: List[List[str]] = []
    for model_dir in discover_model_dirs(results_dir):
        match = thread_pattern.match(model_dir.name)
        if not match: continue
        threads = int(match.group(1))
        stats = load_statistics(model_dir)
        ttft_mean, ttft_std = mean_metric(stats, "ttft_mean_ms", "short")
        tpot_mean, tpot_std = mean_metric(stats, "tpot_mean_ms", "short")
        tps_mean, tps_std = mean_metric(stats, "decode_tps_mean", "short")
        mem_mean, _ = mean_metric(stats, "memory_mb_mean", "short")
        rows.append([
            str(threads),
            format_mean_std(ttft_mean / 1000.0, ttft_std / 1000.0),
            format_mean_std(tpot_mean, tpot_std),
            format_mean_std(tps_mean, tps_std),
            f"{mem_mean:.1f}",
        ])
    rows.sort(key=lambda row: int(row[0]))
    return rows

def build_model_comparison_rows(results_dir: Path) -> List[List[str]]:
    selected = ["llama-3.1-8b-threads-32", "qwen2.5-0.5b", "tinyllama-1.1b"]
    available = {path.name: path for path in discover_model_dirs(results_dir)}
    rows: List[List[str]] = []
    for model_key in selected:
        if model_key not in available: continue
        stats = load_statistics(available[model_key])
        for category in ("short", "medium", "long"):
            ttft_mean, ttft_std = mean_metric(stats, "ttft_mean_ms", category)
            tpot_mean, tpot_std = mean_metric(stats, "tpot_mean_ms", category)
            tps_mean, tps_std = mean_metric(stats, "decode_tps_mean", category)
            mem_mean, _ = mean_metric(stats, "memory_mb_mean", category)
            rows.append([
                label_for_model(model_key),
                category.capitalize(),
                format_mean_std(ttft_mean / 1000.0, ttft_std / 1000.0),
                format_mean_std(tpot_mean, tpot_std),
                format_mean_std(tps_mean, tps_std),
                f"{mem_mean:.1f}",
            ])
    return rows

def build_quantization_comparison_rows(results_dir: Path) -> List[List[str]]:
    quants = [("llama-3.1-8b-threads-32", "Q4_K_M"), ("llama-3.1-8b-Q8", "Q8_0")]
    available = {path.name: path for path in discover_model_dirs(results_dir)}
    rows = []
    for model_key, quant_label in quants:
        if model_key not in available: continue
        stats = load_statistics(available[model_key])
        for category in ("short", "medium", "long"):
            ttft_mean, ttft_std = mean_metric(stats, "ttft_mean_ms", category)
            tpot_mean, tpot_std = mean_metric(stats, "tpot_mean_ms", category)
            tps_mean, tps_std = mean_metric(stats, "decode_tps_mean", category)
            mem_mean, _ = mean_metric(stats, "memory_mb_mean", category)
            rows.append([
                quant_label, category.capitalize(),
                format_mean_std(ttft_mean / 1000.0, ttft_std / 1000.0),
                format_mean_std(tpot_mean, tpot_std),
                format_mean_std(tps_mean, tps_std),
                f"{mem_mean:.1f}"
            ])
    return rows

def model_file_size_gb(models_dir: Path, file_name: str) -> float:
    file_path = models_dir / file_name
    if not file_path.exists(): return 0.0
    return file_path.stat().st_size / 1_000_000_000.0

def build_model_summary_rows(models_dir: Path, effective_bandwidth_gbs: float) -> List[List[str]]:
    rows: List[List[str]] = []
    for item in MODEL_SUMMARY:
        size_gb = model_file_size_gb(models_dir, item["model_file"])
        if size_gb <= 0: size_gb = float(item["fallback_size_gb"])
        predicted_tpot_ms = (size_gb / effective_bandwidth_gbs) * 1000.0 if effective_bandwidth_gbs else 0.0
        rows.append([
            item["display_name"], item["quantization"], f"{size_gb:.2f}",
            f"{effective_bandwidth_gbs:.1f}", f"{predicted_tpot_ms:.1f}",
        ])
    return rows

def build_tpot_validation_rows(results_dir: Path, model_size_gb: float, effective_bandwidth_gbs: float) -> List[List[str]]:
    thread_pattern = re.compile(r"llama-3\.1-8b-threads-(\d+)$")
    observed_by_threads: Dict[int, float] = {}
    for model_dir in discover_model_dirs(results_dir):
        match = thread_pattern.match(model_dir.name)
        if not match: continue
        stats = load_statistics(model_dir)
        mean, _ = mean_metric(stats, "tpot_mean_ms", "short")
        observed_by_threads[int(match.group(1))] = mean

    predicted = (model_size_gb / effective_bandwidth_gbs) * 1000.0 if effective_bandwidth_gbs else 0.0
    rows = []
    for threads in sorted(observed_by_threads):
        observed = observed_by_threads[threads]
        error = ((observed - predicted) / predicted) * 100.0 if predicted else 0.0
        rows.append([str(threads), f"{observed:.2f}", f"{predicted:.1f}", format_signed_percent(error)])
    return rows

def generate_typst_thread_scaling(rows: List[List[str]]) -> str:
    lines = [
        '#figure(',
        '  text(size: 9pt,',
        '    table(',
        '      columns: (auto, auto, auto, auto, auto),',
        '      align: (center, center, center, center, center),',
        '      ',
        '      table.header(',
        '        [*Threads*],',
        '        [*TTFT short (s)*],',
        '        [*TPOT (ms)*],',
        '        [*Throughput\\ (tok/s)*],',
        '        [*Peak Mem.\\ (MB)*],',
        '      ),'
    ]
    for r in rows:
        if r[0] == "32":
            lines.append(f'      table.cell(fill: luma(220))[*{r[0]}*], table.cell(fill: luma(220))[*{r[1]}*], table.cell(fill: luma(220))[*{r[2]}*], table.cell(fill: luma(220))[*{r[3]}*], table.cell(fill: luma(220))[*{r[4]}*],')
        else:
            lines.append(f'      [{r[0]}],  [{r[1]}],  [{r[2]}], [{r[3]}],  [{r[4]}],')
    lines.append('    )')
    lines.append('  ),')
    lines.append('  caption: [Resultados obtidos pelo modelo Llama-3.1-8B com diferentes threads.],')
    lines.append(') <tab:thread-scaling>')
    return "\n".join(lines) + "\n"


def generate_typst_model_comparison(rows: List[List[str]]) -> str:
    lines = [
        '#figure(',
        '  text(size: 9pt,',
        '    table(',
        '      columns: (auto, auto, auto, auto, auto, auto),',
        '      align: (left, center, center, center, center, center),',
        '',
        '      table.header(',
        '        [*Model*],',
        '        [*Category*],',
        '        [*TTFT (s)*],',
        '        [*TPOT (ms)*],',
        '        [*Throughput\\ (tok/s)*],',
        '        [*Peak Mem.\\ (MB)*],',
        '      ),'
    ]
    grouped = {}
    for r in rows:
        grouped.setdefault(r[0], []).append(r)
        
    for model, m_rows in grouped.items():
        lines.append(f'\n      // {model}')
        if "Llama-3.1-8B" in model and "Q8" not in model:
            model_label = "Meta-Llama-3.1-8B\\\n        (Q4\\_K\\_M)"
        elif "TinyLlama" in model:
            model_label = "TinyLlama-1.1B"
        elif "Qwen" in model:
            model_label = "Qwen2.5-0.5B"
        else:
            model_label = model.replace("_", "\\_")
            
        lines.append(f'      table.cell(rowspan: {len(m_rows)})[{model_label}],')
        for r in m_rows:
            lines.append(f'      [{r[1]}],  [{r[2]}], [{r[3]}], [{r[4]}], [{r[5]}],')
            
    lines.append('    )')
    lines.append('  ),')
    lines.append('  caption: [Comparação de métricas de inferência entre modelos a 32 threads.],')
    lines.append(') <tab:model-comparison>')
    return "\n".join(lines) + "\n"


def generate_typst_quantization(rows: List[List[str]]) -> str:
    lines = [
        '#figure(',
        '  text(size: 9pt,',
        '    table(',
        '      columns: (auto, auto, auto, auto, auto, auto),',
        '      align: (left, center, center, center, center, center),',
        '',
        '      table.header(',
        '        [*Quantização*],',
        '        [*Category*],',
        '        [*TTFT (s)*],',
        '        [*TPOT (ms)*],',
        '        [*Throughput\\ (tok/s)*],',
        '        [*Peak Mem.\\ (MB)*],',
        '      ),'
    ]
    grouped = {}
    for r in rows:
        grouped.setdefault(r[0], []).append(r)
        
    for quant, m_rows in grouped.items():
        lines.append(f'\n      // {quant}')
        quant_escaped = quant.replace("_", "\\_")
        lines.append(f'      table.cell(rowspan: {len(m_rows)})[{quant_escaped}],')
        for r in m_rows:
            lines.append(f'      [{r[1]}],  [{r[2]}],  [{r[3]}], [{r[4]}], [{r[5]}],')
            
    lines.append('    )')
    lines.append('  ),')
    lines.append('  caption: [Comparação entre quantizações Q4\\_K\\_M e Q8\\_0 para o modelo \n             Meta-Llama-3.1-8B a 32 threads.],')
    lines.append(') <tab:quantization>')
    return "\n".join(lines) + "\n"


def generate_typst_model_summary(rows: List[List[str]]) -> str:
    lines = [
        '#figure(',
        '  text(size: 9pt,',
        '    table(',
        '      columns: (auto, auto, auto, auto, auto),',
        '      align: (left, center, center, center, center),',
        '',
        '      table.header(',
        '        [*Modelo*],',
        '        [*Quantização*],',
        '        [*Tamanho (GB)*],',
        '        [*$B_"efetiva"$ (GB/s)*],',
        '        [*TPOT previsto*],',
        '      ),'
    ]
    for r in rows:
        name = "Llama-3.1-8B-Instruct" if "Meta-Llama" in r[0] else r[0]
        quant = r[1].replace("_", "\\_")
        lines.append(f'      [{name}], [{quant}], [{r[2]}], [{r[3]}], [{r[4]}],')
    lines.append('    )')
    lines.append('  ),')
    lines.append('  caption: [Comparação entre modelos, quantização, tamanho e TPOT previsto.],')
    lines.append(') <tab:model-summary>')
    return "\n".join(lines) + "\n"


def generate_typst_tpot_validation(rows: List[List[str]]) -> str:
    lines = [
        '#figure(',
        '  text(size: 9pt,',
        '    table(',
        '      columns: (auto, auto, auto, auto),',
        '      align: (center, center, center, center),',
        '',
        '      table.header(',
        '        [*Threads*],',
        '        [*TPOT observado*],',
        '        [*TPOT previsto*],',
        '        [*Erro relativo*],',
        '      ),'
    ]
    for r in rows:
        error = r[3].replace("-", "−")
        lines.append(f'      [{r[0]}],  [{r[1]}], [{r[2]}], [{error}],')
    lines.append('    )')
    lines.append('  ),')
    lines.append('  caption: [Comparação entre TPOT observado e previsto para diferentes números de threads.],')
    lines.append(') <tab:tpot-threads>')
    return "\n".join(lines) + "\n"



def write_report_tables(results_dir: Path, tables_dir: Path, models_dir: Path, model_size_gb: float, effective_bandwidth_gbs: float) -> List[Path]:
    generated: List[Path] = []
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Thread Scaling
    thread_rows = build_thread_scaling_rows(results_dir)
    if thread_rows:
        csv_path = tables_dir / "thread_scaling.csv"
        md_path = tables_dir / "thread_scaling.md"
        write_csv_table(csv_path, ["Threads", "TTFT short (s)", "TPOT (ms)", "Throughput (tok/s)", "Peak Mem. (MB)"], thread_rows)
        md_path.write_text(markdown_table(["Threads", "TTFT short (s)", "TPOT (ms)", "Throughput (tok/s)", "Peak Mem. (MB)"], thread_rows), encoding="utf-8")
        
        # O novo ficheiro .typ customizado
        typ_path = tables_dir / "thread_scaling.typ"
        typ_path.write_text(generate_typst_thread_scaling(thread_rows), encoding="utf-8")
        generated.extend([csv_path, md_path, typ_path])

    # 2. Model Comparison
    model_rows = build_model_comparison_rows(results_dir)
    if model_rows:
        csv_path = tables_dir / "model_comparison.csv"
        md_path = tables_dir / "model_comparison.md"
        write_csv_table(csv_path, ["Model", "Category", "TTFT (s)", "TPOT (ms)", "Throughput (tok/s)", "Peak Mem. (MB)"], model_rows)
        md_path.write_text(markdown_table(["Model", "Category", "TTFT (s)", "TPOT (ms)", "Throughput (tok/s)", "Peak Mem. (MB)"], model_rows), encoding="utf-8")
        
        typ_path = tables_dir / "model_comparison.typ"
        typ_path.write_text(generate_typst_model_comparison(model_rows), encoding="utf-8")
        generated.extend([csv_path, md_path, typ_path])

    # 3. Quantization Comparison
    quant_rows = build_quantization_comparison_rows(results_dir)
    if quant_rows:
        csv_path = tables_dir / "quantization.csv"
        md_path = tables_dir / "quantization.md"
        write_csv_table(csv_path, ["Quantização", "Category", "TTFT (s)", "TPOT (ms)", "Throughput (tok/s)", "Peak Mem. (MB)"], quant_rows)
        md_path.write_text(markdown_table(["Quantização", "Category", "TTFT (s)", "TPOT (ms)", "Throughput (tok/s)", "Peak Mem. (MB)"], quant_rows), encoding="utf-8")
        
        typ_path = tables_dir / "quantization.typ"
        typ_path.write_text(generate_typst_quantization(quant_rows), encoding="utf-8")
        generated.extend([csv_path, md_path, typ_path])

    # 4. Model Summary
    summary_rows = build_model_summary_rows(models_dir, effective_bandwidth_gbs)
    if summary_rows:
        csv_path = tables_dir / "model_summary.csv"
        md_path = tables_dir / "model_summary.md"
        write_csv_table(csv_path, ["Model", "Quantization", "Size (GB)", "Effective BW (GB/s)", "Predicted TPOT (ms)"], summary_rows)
        md_path.write_text(markdown_table(["Model", "Quantization", "Size (GB)", "Effective BW (GB/s)", "Predicted TPOT (ms)"], summary_rows), encoding="utf-8")
        
        typ_path = tables_dir / "model_summary.typ"
        typ_path.write_text(generate_typst_model_summary(summary_rows), encoding="utf-8")
        generated.extend([csv_path, md_path, typ_path])

    # 5. TPOT Validation
    validation_rows = build_tpot_validation_rows(results_dir, model_size_gb, effective_bandwidth_gbs)
    if validation_rows:
        csv_path = tables_dir / "tpot_validation.csv"
        md_path = tables_dir / "tpot_validation.md"
        write_csv_table(csv_path, ["Threads", "TPOT observed (ms)", "TPOT predicted (ms)", "Relative error"], validation_rows)
        md_path.write_text(markdown_table(["Threads", "TPOT observed (ms)", "TPOT predicted (ms)", "Relative error"], validation_rows), encoding="utf-8")
        
        typ_path = tables_dir / "tpot_validation.typ"
        typ_path.write_text(generate_typst_tpot_validation(validation_rows), encoding="utf-8")
        generated.extend([csv_path, md_path, typ_path])

    manifest_path = tables_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"results_dir": str(results_dir), "tables": [str(p) for p in generated]}, indent=2), encoding="utf-8")
    generated.append(manifest_path)
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report-ready plots and tables from benchmark results.")
    parser.add_argument("--results-dir", default="results", help="Directory containing model result folders")
    parser.add_argument("--output-dir", default=None, help="Directory where plots will be written")
    parser.add_argument("--tables-dir", default=None, help="Directory where tables will be written")
    parser.add_argument("--model-size-gb", type=float, default=4.60, help="Model size in GB")
    parser.add_argument("--effective-bandwidth-gbs", type=float, default=18.0, help="Effective memory bandwidth in GB/s")
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    requested_results_dir = Path(args.results_dir).resolve()
    results_dir = resolve_results_dir(requested_results_dir)

    if not results_dir.exists() or not discover_model_dirs(results_dir):
        print(f"Results directory not found or no statistics.csv files found under: {requested_results_dir}")
        return 1

    output_dir = Path(args.output_dir).resolve() if args.output_dir else results_dir / "plots" / "report"
    tables_dir = Path(args.tables_dir).resolve() if args.tables_dir else results_dir / "tables"
    models_dir = results_dir.parent / "models"
    if not models_dir.exists():
        models_dir = requested_results_dir.parent / "models"

    generated: List[Path] = []

    for plotter in (
        lambda: plot_thread_scaling(results_dir, output_dir),
        lambda: plot_model_comparison(results_dir, output_dir),
        lambda: plot_quantization_comparison(results_dir, output_dir),
        lambda: plot_observed_vs_predicted_tpot(results_dir, output_dir, args.model_size_gb, args.effective_bandwidth_gbs),
        lambda: plot_memory_overview(results_dir, output_dir),
    ):
        path = plotter()
        if path is not None:
            generated.append(path)

    generated.extend(write_report_tables(results_dir, tables_dir, models_dir, args.model_size_gb, args.effective_bandwidth_gbs))

    print("Generated assets:")
    for path in generated:
        print(f" - {path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())