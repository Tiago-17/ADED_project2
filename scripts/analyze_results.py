#!/usr/bin/env python3
"""
Results Analyzer for LLM Inference Benchmark - Track A1
Generates all required plots and tables from benchmark results.

Usage:
    python scripts/analyze_results.py --results-dir results/llama-3.1-8b
    python scripts/analyze_results.py --results-dir results --compare-models
    python scripts/analyze_results.py --results-dir results --all
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import statistics
import math

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#9C27B0', '#00BCD4']


class ResultsAnalyzer:
    """Analyze and visualize LLM benchmark results."""
    
    def __init__(self, results_dir: str, project_root: str = "."):
        """
        Initialize analyzer.
        
        Args:
            results_dir: Directory containing benchmark results
            project_root: Project root directory
        """
        self.results_dir = Path(results_dir).resolve()
        self.project_root = Path(project_root).resolve()
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.all_results = {}
        self.aggregated_stats = {}
        
        print(f"Results directory: {self.results_dir}")
        print(f"Plots directory: {self.plots_dir}")
    
    def load_results(self, model_name: str = None) -> Dict:
        """
        Load benchmark results for a specific model or all models.
        
        Args:
            model_name: Optional model name to load
        
        Returns:
            Dictionary with loaded results
        """
        results = {}
        
        if model_name:
            # Load specific model
            model_dir = self.results_dir / model_name
            if model_dir.exists():
                stats_file = model_dir / "statistics.json"
                if stats_file.exists():
                    with open(stats_file) as f:
                        results[model_name] = json.load(f)
                    print(f"Loaded {model_name}: {len(results[model_name])} prompts")
        else:
            # Load all models in results directory
            for model_dir in self.results_dir.iterdir():
                if model_dir.is_dir() and model_dir.name != "plots":
                    stats_file = model_dir / "statistics.json"
                    if stats_file.exists():
                        model = model_dir.name
                        with open(stats_file) as f:
                            results[model] = json.load(f)
                        print(f"Loaded {model}: {len(results[model])} prompts")
        
        self.aggregated_stats = results
        return results
    
    def load_raw_results(self, model_name: str) -> List[Dict]:
        """Load raw (non-aggregated) results for detailed analysis."""
        model_dir = self.results_dir / model_name
        raw_file = model_dir / "raw_results.json"
        
        if raw_file.exists():
            with open(raw_file) as f:
                return json.load(f)
        return []
    
    def load_monitor_data(self, model_name: str) -> Dict:
        """Load resource monitoring data."""
        model_dir = self.results_dir / model_name
        monitor_data = {
            'cpu': None,
            'memory': None,
            'energy': None,
            'summary': None
        }
        
        # Find latest monitor files
        for file in model_dir.glob("*_cpu_*.csv"):
            monitor_data['cpu'] = self._read_csv(file)
        for file in model_dir.glob("*_memory_*.csv"):
            monitor_data['memory'] = self._read_csv(file)
        for file in model_dir.glob("*_energy_*.csv"):
            if file.stat().st_size > 0:
                monitor_data['energy'] = self._read_csv(file)
        for file in model_dir.glob("*_summary_*.txt"):
            with open(file) as f:
                monitor_data['summary'] = f.read()
        
        return monitor_data
    
    def _read_csv(self, filepath: Path) -> List[Dict]:
        """Read CSV file into list of dictionaries."""
        import csv
        rows = []
        with open(filepath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows
    
    # ============================================
    # PLOT 1: TTFT vs Prompt Length (by category)
    # ============================================
    def plot_ttft_vs_prompt_length(self, model_name: str = None):
        """
        Generate TTFT vs prompt length plot.
        This is a REQUIRED plot for the report.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        models_to_plot = [model_name] if model_name else list(self.aggregated_stats.keys())
        
        # Plot 1: TTFT by category (bar chart)
        ax1 = axes[0]
        categories = ['short', 'medium', 'long']
        x_positions = np.arange(len(categories))
        bar_width = 0.8 / len(models_to_plot) if len(models_to_plot) > 1 else 0.5
        
        for i, model in enumerate(models_to_plot):
            stats = self.aggregated_stats.get(model, [])
            if not stats:
                continue
            
            # Calculate mean TTFT per category
            cat_ttft = defaultdict(list)
            for s in stats:
                cat_ttft[s['category']].append(s['ttft_mean_ms'])
            
            means = []
            stds = []
            for cat in categories:
                vals = cat_ttft.get(cat, [0])
                means.append(statistics.mean(vals) if vals else 0)
                stds.append(statistics.stdev(vals) if len(vals) > 1 else 0)
            
            offset = i * bar_width
            bars = ax1.bar(x_positions + offset, means, bar_width, 
                          yerr=stds, label=model, color=COLORS[i % len(COLORS)],
                          capsize=5, alpha=0.8)
            
            # Add value labels
            for bar, mean in zip(bars, means):
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(stds) * 0.1,
                        f'{mean:.0f}', ha='center', va='bottom', fontsize=9)
        
        ax1.set_xlabel('Prompt Category', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Time to First Token (ms)', fontsize=12, fontweight='bold')
        ax1.set_title('TTFT by Prompt Category', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_positions + bar_width * (len(models_to_plot) - 1) / 2)
        ax1.set_xticklabels([c.capitalize() for c in categories])
        ax1.legend(loc='upper left')
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: TTFT distribution (box plot)
        ax2 = axes[1]
        
        # Collect all TTFT values from raw results
        box_data = []
        box_labels = []
        
        for model in models_to_plot:
            raw_results = self.load_raw_results(model)
            if not raw_results:
                continue
            
            model_cat_data = defaultdict(list)
            for r in raw_results:
                if not r.get('error') and r['ttft_ms'] > 0:
                    model_cat_data[r['category']].append(r['ttft_ms'])
            
            for cat in categories:
                vals = model_cat_data.get(cat, [])
                if vals:
                    box_data.append(vals)
                    box_labels.append(f"{model}\n{cat}")
        
        if box_data:
            bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True,
                            showfliers=True, showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
            
            for patch, color in zip(bp['boxes'], COLORS * (len(box_data) // len(COLORS) + 1)):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        
        ax2.set_ylabel('Time to First Token (ms)', fontsize=12, fontweight='bold')
        ax2.set_title('TTFT Distribution by Model and Category', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"ttft_analysis_{model_name or 'all'}.png"
        filepath = self.plots_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {filepath}")
    
    # ============================================
    # PLOT 2: Throughput vs Thread Count
    # ============================================
    def plot_throughput_vs_threads(self, model_name: str = None):
        """
        Generate throughput analysis plot.
        REQUIRED for Track A1 when thread count is varied.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        models_to_plot = [model_name] if model_name else list(self.aggregated_stats.keys())
        
        # Plot 1: Decode throughput by category
        ax1 = axes[0]
        categories = ['short', 'medium', 'long']
        
        for i, model in enumerate(models_to_plot):
            stats = self.aggregated_stats.get(model, [])
            if not stats:
                continue
            
            cat_tps = defaultdict(list)
            for s in stats:
                if s['decode_tps_mean'] > 0:
                    cat_tps[s['category']].append(s['decode_tps_mean'])
            
            means = []
            for cat in categories:
                vals = cat_tps.get(cat, [])
                means.append(statistics.mean(vals) if vals else 0)
            
            x = np.arange(len(categories))
            ax1.plot(x, means, 'o-', linewidth=2, markersize=8, 
                    label=model, color=COLORS[i])
        
        ax1.set_xlabel('Prompt Category', fontweight='bold')
        ax1.set_ylabel('Decode Throughput (tokens/s)', fontweight='bold')
        ax1.set_title('Decode Throughput by Category', fontweight='bold')
        ax1.set_xticks(range(len(categories)))
        ax1.set_xticklabels([c.capitalize() for c in categories])
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Throughput components (prefill vs decode)
        ax2 = axes[1]
        
        if models_to_plot:
            model = models_to_plot[0]
            stats = self.aggregated_stats.get(model, [])
            
            if stats:
                # Average across all prompts
                prefill_vals = []
                decode_vals = []
                for s in stats:
                    raw = self.load_raw_results(model)
                    for r in raw:
                        if not r.get('error'):
                            if r.get('prefill_throughput_tps', 0) > 0:
                                prefill_vals.append(r['prefill_throughput_tps'])
                            if r.get('decode_throughput_tps', 0) > 0:
                                decode_vals.append(r['decode_throughput_tps'])
                
                components = ['Prefill\n(prompt processing)', 'Decode\n(token generation)']
                means = [
                    statistics.mean(prefill_vals) if prefill_vals else 0,
                    statistics.mean(decode_vals) if decode_vals else 0
                ]
                
                bars = ax2.bar(components, means, color=[COLORS[0], COLORS[1]], alpha=0.8)
                for bar, val in zip(bars, means):
                    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                            f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax2.set_ylabel('Throughput (tokens/s)', fontweight='bold')
        ax2.set_title('Prefill vs Decode Throughput', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Overall throughput comparison
        ax3 = axes[2]
        
        model_throughputs = []
        model_names = []
        
        for model in models_to_plot:
            stats = self.aggregated_stats.get(model, [])
            if stats:
                overall_vals = []
                for s in stats:
                    raw = self.load_raw_results(model)
                    for r in raw:
                        if not r.get('error') and r.get('overall_throughput_tps', 0) > 0:
                            overall_vals.append(r['overall_throughput_tps'])
                
                if overall_vals:
                    model_throughputs.append(statistics.mean(overall_vals))
                    model_names.append(model)
        
        if model_throughputs:
            bars = ax3.barh(model_names, model_throughputs, 
                           color=COLORS[:len(model_names)], alpha=0.8)
            for bar, val in zip(bars, model_throughputs):
                ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2.,
                        f'{val:.1f}', va='center', fontweight='bold')
        
        ax3.set_xlabel('Overall Throughput (tokens/s)', fontweight='bold')
        ax3.set_title('Overall Throughput Comparison', fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        filename = f"throughput_analysis_{model_name or 'all'}.png"
        plt.savefig(self.plots_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {self.plots_dir / filename}")
    
    # ============================================
    # PLOT 3: Memory vs Quantization Level
    # ============================================
    def plot_memory_analysis(self):
        """
        Generate memory usage analysis plot.
        REQUIRED for Track A1.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Memory per model
        ax1 = axes[0]
        
        model_memory = {}
        for model in self.aggregated_stats:
            stats = self.aggregated_stats[model]
            mem_vals = [s.get('memory_mb_mean', 0) for s in stats if s.get('memory_mb_mean', 0) > 0]
            if mem_vals:
                model_memory[model] = statistics.mean(mem_vals)
        
        if model_memory:
            models = list(model_memory.keys())
            memories = [model_memory[m] for m in models]
            
            bars = ax1.barh(models, memories, color=COLORS[:len(models)], alpha=0.8)
            
            for bar, mem in zip(bars, memories):
                # Show in GB if > 1024
                if mem > 1024:
                    label = f'{mem/1024:.1f} GB'
                else:
                    label = f'{mem:.0f} MB'
                ax1.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2.,
                        label, va='center', fontweight='bold')
        
        ax1.set_xlabel('Memory Usage (MB)', fontweight='bold')
        ax1.set_title('Peak Memory Usage by Model', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Plot 2: Memory breakdown from monitor data
        ax2 = axes[1]
        
        # Try to load monitor data for the first model
        if self.aggregated_stats:
            first_model = list(self.aggregated_stats.keys())[0]
            monitor_data = self.load_monitor_data(first_model)
            
            if monitor_data.get('memory'):
                mem_rows = monitor_data['memory']
                if mem_rows:
                    timestamps = list(range(len(mem_rows)))
                    used_mb = [float(r.get('used_mb', 0)) for r in mem_rows]
                    cached_mb = [float(r.get('cached_mb', 0)) for r in mem_rows]
                    free_mb = [float(r.get('free_mb', 0)) for r in mem_rows]
                    
                    ax2.fill_between(timestamps, 0, used_mb, alpha=0.7, 
                                    color=COLORS[0], label='Used')
                    ax2.fill_between(timestamps, used_mb, 
                                    [u + c for u, c in zip(used_mb, cached_mb)],
                                    alpha=0.7, color=COLORS[1], label='Cached')
                    
                    ax2.set_xlabel('Time (samples)', fontweight='bold')
                    ax2.set_ylabel('Memory (MB)', fontweight='bold')
                    ax2.set_title(f'Memory Usage Over Time - {first_model}', fontweight='bold')
                    ax2.legend(loc='upper right')
                    ax2.grid(alpha=0.3)
        
        if not monitor_data.get('memory'):
            ax2.text(0.5, 0.5, 'No monitor data available\nRun monitor_resources.sh during benchmark',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        
        plt.tight_layout()
        filepath = self.plots_dir / "memory_analysis.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {filepath}")
    
    # ============================================
    # PLOT 4: TPOT Analysis
    # ============================================
    def plot_tpot_analysis(self, model_name: str = None):
        """
        Generate TPOT (Time Per Output Token) analysis.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: TPOT by category
        ax1 = axes[0]
        categories = ['short', 'medium', 'long']
        
        models_to_plot = [model_name] if model_name else list(self.aggregated_stats.keys())
        
        x = np.arange(len(categories))
        width = 0.8 / len(models_to_plot) if len(models_to_plot) > 1 else 0.5
        
        for i, model in enumerate(models_to_plot):
            stats = self.aggregated_stats.get(model, [])
            if not stats:
                continue
            
            cat_tpot = defaultdict(list)
            for s in stats:
                if s.get('tpot_mean_ms', 0) > 0:
                    cat_tpot[s['category']].append(s['tpot_mean_ms'])
            
            means = []
            stds = []
            for cat in categories:
                vals = cat_tpot.get(cat, [0])
                means.append(statistics.mean(vals) if vals else 0)
                stds.append(statistics.stdev(vals) if len(vals) > 1 else 0)
            
            offset = i * width
            ax1.bar(x + offset, means, width, yerr=stds, 
                   label=model, color=COLORS[i], capsize=5, alpha=0.8)
        
        ax1.set_xlabel('Prompt Category', fontweight='bold')
        ax1.set_ylabel('Average TPOT (ms)', fontweight='bold')
        ax1.set_title('Time Per Output Token by Category', fontweight='bold')
        ax1.set_xticks(x + width * (len(models_to_plot) - 1) / 2)
        ax1.set_xticklabels([c.capitalize() for c in categories])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: TPOT distribution over generation
        ax2 = axes[1]
        
        # Load raw results to get per-token TPOT values
        for model in models_to_plot:
            raw_results = self.load_raw_results(model)
            if not raw_results:
                continue
            
            all_tpots = []
            for r in raw_results:
                if not r.get('error') and r.get('tpot_values_ms'):
                    all_tpots.extend(r['tpot_values_ms'])
            
            if all_tpots:
                # Sample for histogram
                sampled = all_tpots[:1000] if len(all_tpots) > 1000 else all_tpots
                ax2.hist(sampled, bins=50, alpha=0.5, label=f"{model}\n(μ={statistics.mean(all_tpots):.1f}ms)",
                        color=COLORS[models_to_plot.index(model) % len(COLORS)], density=True)
        
        ax2.set_xlabel('TPOT (ms)', fontweight='bold')
        ax2.set_ylabel('Density', fontweight='bold')
        ax2.set_title('TPOT Distribution During Decoding', fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        filename = f"tpot_analysis_{model_name or 'all'}.png"
        plt.savefig(self.plots_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {self.plots_dir / filename}")
    
    # ============================================
    # PLOT 5: Performance Model - Predicted vs Observed
    # ============================================
    def plot_performance_model(self, model_name: str, 
                               model_size_gb: float = None, 
                               memory_bandwidth_gbs: float = None):
        """
        Generate predicted vs observed TPOT plot.
        REQUIRED for the performance model section.
        
        Args:
            model_name: Name of the model
            model_size_gb: Model size in GB (for prediction)
            memory_bandwidth_gbs: Measured memory bandwidth in GB/s
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Load raw results
        raw_results = self.load_raw_results(model_name)
        if not raw_results:
            print(f"No raw results for {model_name}")
            return
        
        # Plot 1: Predicted vs Observed TPOT
        ax1 = axes[0]
        
        # Extract observed TPOT values
        observed_tpots = []
        for r in raw_results:
            if not r.get('error') and r.get('tpot_avg_ms', 0) > 0:
                observed_tpots.append(r['tpot_avg_ms'])
        
        if observed_tpots:
            # If we have model size and bandwidth, compute prediction
            if model_size_gb and memory_bandwidth_gbs:
                # Theoretical minimum: time to read all weights from memory
                predicted_tpot_ms = (model_size_gb / memory_bandwidth_gbs) * 1000
                
                ax1.axhline(y=predicted_tpot_ms, color=COLORS[0], linestyle='--', 
                           linewidth=2, label=f'Predicted (BW-bound): {predicted_tpot_ms:.1f} ms')
            
            avg_observed = statistics.mean(observed_tpots)
            ax1.axhline(y=avg_observed, color=COLORS[1], linestyle='-', 
                       linewidth=2, label=f'Observed (mean): {avg_observed:.1f} ms')
            
            # Scatter plot of individual observations
            ax1.scatter(range(len(observed_tpots)), observed_tpots, 
                       alpha=0.5, s=20, color=COLORS[2], label='Individual measurements')
            
            # Add ±1 std band
            std_observed = statistics.stdev(observed_tpots) if len(observed_tpots) > 1 else 0
            ax1.axhspan(avg_observed - std_observed, avg_observed + std_observed, 
                       alpha=0.2, color=COLORS[1], label=f'±1σ band')
        
        ax1.set_xlabel('Measurement Index', fontweight='bold')
        ax1.set_ylabel('TPOT (ms)', fontweight='bold')
        ax1.set_title(f'Performance Model: Predicted vs Observed TPOT\n{model_name}', fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(alpha=0.3)
        
        # Plot 2: Efficiency analysis
        ax2 = axes[1]
        
        if model_size_gb and memory_bandwidth_gbs and observed_tpots:
            predicted_tpot_ms = (model_size_gb / memory_bandwidth_gbs) * 1000
            avg_observed = statistics.mean(observed_tpots)
            
            # Memory bandwidth utilization efficiency
            efficiency = (predicted_tpot_ms / avg_observed) * 100 if avg_observed > 0 else 0
            
            # Create efficiency breakdown
            categories_bar = ['Theoretical\nMinimum', 'Observed\nAverage', 'Overhead']
            values = [predicted_tpot_ms, avg_observed - predicted_tpot_ms, 
                     max(0, avg_observed - predicted_tpot_ms)]
            
            colors_bar = [COLORS[0], COLORS[1], '#FF5722']
            
            # Waterfall-like chart
            bottom = 0
            for i, (cat, val, color) in enumerate(zip(categories_bar, values, colors_bar)):
                if i == 0:
                    bottom = 0
                    ax2.bar(cat, val, color=color, alpha=0.8, bottom=bottom)
                    bottom = val
                else:
                    ax2.bar(cat, val, color=color, alpha=0.8, bottom=bottom)
                    bottom += val
            
            ax2.set_ylabel('TPOT (ms)', fontweight='bold')
            ax2.set_title(f'TPOT Breakdown - Efficiency: {efficiency:.1f}%', fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            
            # Add efficiency annotation
            ax2.text(0.5, 0.95, f'Memory BW Efficiency: {efficiency:.1f}%\n'
                    f'Pred: {predicted_tpot_ms:.1f}ms | Obs: {avg_observed:.1f}ms',
                    transform=ax2.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=10)
        
        plt.tight_layout()
        filepath = self.plots_dir / f"performance_model_{model_name}.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {filepath}")
    
    # ============================================
    # PLOT 6: CPU & Memory Monitoring Dashboard
    # ============================================
    def plot_resource_dashboard(self, model_name: str):
        """
        Generate comprehensive resource utilization dashboard.
        """
        monitor_data = self.load_monitor_data(model_name)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        
        # CPU Usage
        ax1 = axes[0, 0]
        if monitor_data.get('cpu'):
            cpu_data = monitor_data['cpu']
            timestamps = list(range(len(cpu_data)))
            
            # Extract CPU metrics
            user = [float(r.get('user', 0)) for r in cpu_data]
            system = [float(r.get('system', 0)) for r in cpu_data]
            iowait = [float(r.get('iowait', 0)) for r in cpu_data]
            
            ax1.stackplot(timestamps, user, system, iowait,
                         labels=['User', 'System', 'IOWait'],
                         colors=COLORS[:3], alpha=0.7)
            ax1.set_xlabel('Time (samples)', fontweight='bold')
            ax1.set_ylabel('CPU Usage (%)', fontweight='bold')
            ax1.set_title(f'CPU Utilization - {model_name}', fontweight='bold')
            ax1.legend(loc='upper right')
            ax1.set_ylim(0, 100)
        
        # Memory Usage
        ax2 = axes[0, 1]
        if monitor_data.get('memory'):
            mem_data = monitor_data['memory']
            timestamps = list(range(len(mem_data)))
            
            used = [float(r.get('used_mb', 0)) for r in mem_data]
            cached = [float(r.get('cached_mb', 0)) for r in mem_data]
            
            ax2.plot(timestamps, used, color=COLORS[0], linewidth=1.5, label='Used')
            ax2.fill_between(timestamps, 0, used, alpha=0.3, color=COLORS[0])
            
            if any(c > 0 for c in cached):
                ax2.plot(timestamps, cached, color=COLORS[1], linewidth=1, 
                        linestyle='--', label='Cached')
            
            ax2.set_xlabel('Time (samples)', fontweight='bold')
            ax2.set_ylabel('Memory (MB)', fontweight='bold')
            ax2.set_title(f'Memory Usage - {model_name}', fontweight='bold')
            ax2.legend(loc='upper right')
        
        # CPU/Memory correlation
        ax3 = axes[1, 0]
        if monitor_data.get('cpu') and monitor_data.get('memory'):
            cpu_data = monitor_data['cpu']
            mem_data = monitor_data['memory']
            
            min_len = min(len(cpu_data), len(mem_data))
            cpu_vals = [float(cpu_data[i].get('user', 0)) + float(cpu_data[i].get('system', 0)) 
                       for i in range(min_len)]
            mem_vals = [float(mem_data[i].get('used_mb', 0)) for i in range(min_len)]
            
            scatter = ax3.scatter(cpu_vals, mem_vals, c=range(min_len), 
                                 cmap='viridis', alpha=0.6, s=30)
            ax3.set_xlabel('CPU Usage (%)', fontweight='bold')
            ax3.set_ylabel('Memory Used (MB)', fontweight='bold')
            ax3.set_title('CPU vs Memory Correlation', fontweight='bold')
            plt.colorbar(scatter, ax=ax3, label='Time progression')
        
        # Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
        RESOURCE MONITORING SUMMARY
        ═══════════════════════════
        Model: {model_name}
        
        CPU Statistics:
        • Avg User:   {statistics.mean(user) if 'user' in dir() else 'N/A':.1f}%
        • Avg System: {statistics.mean(system) if 'system' in dir() else 'N/A':.1f}%
        
        Memory Statistics:
        • Peak Used:  {max(used) if 'used' in dir() else 'N/A':.0f} MB
        • Avg Used:   {statistics.mean(used) if 'used' in dir() else 'N/A':.0f} MB
        """
        
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
                fontsize=11, family='monospace', verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle(f'Resource Monitoring Dashboard - {model_name}', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        filepath = self.plots_dir / f"resource_dashboard_{model_name}.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {filepath}")
    
    # ============================================
    # Generate all plots
    # ============================================
    def generate_all_plots(self, model_name: str = None, 
                          model_params: Dict = None):
        """
        Generate all required plots.
        
        Args:
            model_name: Specific model or None for all
            model_params: Dict with model_size_gb and memory_bandwidth_gbs for performance model
        """
        print("\n" + "="*60)
        print("GENERATING ALL PLOTS")
        print("="*60 + "\n")
        
        # Load results
        self.load_results(model_name)
        
        if not self.aggregated_stats:
            print("No results found to plot!")
            return
        
        models = [model_name] if model_name else list(self.aggregated_stats.keys())
        
        for model in models:
            print(f"\n--- {model} ---")
            
            # Plot 1: TTFT vs Prompt Length
            self.plot_ttft_vs_prompt_length(model)
            
            # Plot 2: Throughput analysis
            self.plot_throughput_vs_threads(model)
            
            # Plot 3: TPOT analysis
            self.plot_tpot_analysis(model)
            
            # Plot 4: Performance model
            if model_params and model in model_params:
                params = model_params[model]
                self.plot_performance_model(
                    model,
                    model_size_gb=params.get('model_size_gb'),
                    memory_bandwidth_gbs=params.get('memory_bandwidth_gbs')
                )
            else:
                self.plot_performance_model(model)
            
            # Plot 5: Resource dashboard
            self.plot_resource_dashboard(model)
        
        # Plot 6: Memory analysis (all models)
        if len(models) > 1:
            self.plot_memory_analysis()
        
        # Print summary
        print("\n" + "="*60)
        print(f"All plots saved to: {self.plots_dir}")
        print("="*60)
        
        # List all generated files
        print("\nGenerated files:")
        for file in sorted(self.plots_dir.glob("*.png")):
            size_kb = file.stat().st_size / 1024
            print(f"  • {file.name} ({size_kb:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize LLM benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single model
  python scripts/analyze_results.py --results-dir results/llama-3.1-8b --model llama-3.1-8b
  
  # Compare all models
  python scripts/analyze_results.py --results-dir results --compare-models
  
  # Generate performance model with known parameters
  python scripts/analyze_results.py --results-dir results --model llama-3.1-8b \\
      --model-size-gb 4.0 --memory-bw-gbs 50.0
        """
    )
    
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing benchmark results')
    parser.add_argument('--model', type=str, default=None,
                       help='Specific model to analyze')
    parser.add_argument('--compare-models', action='store_true',
                       help='Compare all models in results directory')
    parser.add_argument('--model-size-gb', type=float, default=None,
                       help='Model size in GB (for performance model)')
    parser.add_argument('--memory-bw-gbs', type=float, default=None,
                       help='Measured memory bandwidth in GB/s')
    parser.add_argument('--all', action='store_true',
                       help='Generate all possible plots')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ResultsAnalyzer(args.results_dir)
    
    # 1. Package the performance parameters if provided
    model_params = None
    if args.model_size_gb and args.memory_bw_gbs:
        # Create a dictionary that applies these params to any model analyzed
        model_params = {
            args.model: {
                'model_size_gb': args.model_size_gb, 
                'memory_bandwidth_gbs': args.memory_bw_gbs
            }
        } if args.model else defaultdict(lambda: {
            'model_size_gb': args.model_size_gb, 
            'memory_bandwidth_gbs': args.memory_bw_gbs
        })

    # 2. Trigger the appropriate plotting functions based on arguments
    if args.all:
        analyzer.generate_all_plots(model_name=args.model, model_params=model_params)
    elif args.compare_models:
        analyzer.plot_memory_analysis()
        analyzer.plot_throughput_vs_threads()
        print(f"Comparison plots saved to: {analyzer.plots_dir}")
    elif args.model:
        analyzer.generate_all_plots(model_name=args.model, model_params=model_params)
    else:
        parser.print_help()
        print("\nERROR: Please specify an action like --all, --compare-models, or --model <name>")

# 3. Actually execute the main function
if __name__ == "__main__":
    main()