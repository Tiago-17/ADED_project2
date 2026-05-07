#!/usr/bin/env python3
"""
LLM Inference Benchmark Script for llama.cpp
Track A1 - Single-Engine Deep Dive

Usage (from project root):
    python scripts/benchmark_llm.py --model llama-3.1-8b --threads 16
    python scripts/benchmark_llm.py --model qwen2.5-1.5b --threads 8 --trials 3
"""

import argparse
import json
import os
import sys
import time
import csv
import psutil
import requests
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging


class LLMBenchmark:
    """Main benchmark class for LLM inference performance measurement."""
    
    def __init__(self, 
                 model_name: str,
                 project_root: str = ".",
                 server_url: str = "http://localhost:8080",
                 num_threads: int = 16,
                 max_tokens: int = 512,
                 temperature: float = 0.7):
        """
        Initialize benchmark configuration.
        
        Args:
            model_name: Name of the model being tested (e.g., 'llama-3.1-8b')
            project_root: Root directory of the project
            server_url: URL of the llama.cpp server
            num_threads: Number of CPU threads used by server
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.model_name = model_name
        self.project_root = Path(project_root).resolve()
        self.server_url = server_url
        self.num_threads = num_threads
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Define paths relative to project root
        self.prompts_dir = self.project_root / "prompts"
        self.results_dir = self.project_root / "results" / model_name
        self.logs_dir = self.project_root / "logs"
        
        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Load prompts
        self.prompts = self._load_prompts()
        
        # Results storage
        self.results = []
        
        self.logger.info(f"Project root: {self.project_root}")
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Threads: {num_threads}")
        self.logger.info(f"Loaded {len(self.prompts)} prompts")
    
    def _setup_logging(self):
        """Configure logging to file and console."""
        log_file = self.logs_dir / f"benchmark_{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Create logger
        self.logger = logging.getLogger(f"benchmark_{self.model_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging to: {log_file}")
    
    def _load_prompts(self) -> List[Dict]:
        """Load prompts from the prompts directory."""
        prompt_candidates = [
            self.prompts_dir / "benchmark_prompts.json",
            self.prompts_dir / "prompts.json",
        ]
        prompts_file = next((path for path in prompt_candidates if path.exists()), None)

        if prompts_file is None:
            expected = ", ".join(str(path) for path in prompt_candidates)
            self.logger.error(f"Prompts file not found. Looked for: {expected}")
            sys.exit(1)
        
        try:
            with open(prompts_file, 'r') as f:
                data = json.load(f)
            
            prompts = data.get('prompts', [])
            
            # Apply generation params from file if present
            if 'generation_params' in data:
                params = data['generation_params']
                self.temperature = params.get('temperature', self.temperature)
                if 'max_tokens' in params:
                    self.max_tokens = params['max_tokens']
                
            return prompts
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in prompts file: {e}")
            sys.exit(1)

    def _extract_response_text(self, payload: Dict) -> str:
        """Extract generated text from a llama.cpp response payload."""
        if not isinstance(payload, dict):
            return ""

        for key in ("content", "response", "text", "completion", "token"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value

        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                message = first_choice.get("message")
                if isinstance(message, dict):
                    value = message.get("content")
                    if isinstance(value, str) and value:
                        return value

                value = first_choice.get("text")
                if isinstance(value, str) and value:
                    return value

        return ""
    
    def check_server_health(self, max_retries: int = 30, retry_delay: int = 2) -> bool:
        """Check if the llama.cpp server is running and healthy."""
        health_url = f"{self.server_url}/health"
        self.logger.info(f"Checking server health at {health_url}...")
        
        for attempt in range(max_retries):
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    self.logger.info("✓ Server is healthy and ready")
                    return True
            except requests.exceptions.ConnectionError:
                if attempt % 5 == 0:
                    self.logger.info(f"Waiting for server... (attempt {attempt + 1}/{max_retries})")
            except requests.exceptions.Timeout:
                self.logger.warning(f"Health check timeout (attempt {attempt + 1})")
            
            time.sleep(retry_delay)
        
        self.logger.error("Server failed to become healthy")
        return False
    
    def get_memory_usage(self) -> Dict:
        """Get current memory usage of the llama.cpp server process."""
        memory_info = {
            'rss_mb': 0,
            'vms_mb': 0,
            'percent': 0
        }
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and any('llama' in str(arg).lower() for arg in cmdline):
                        mem = proc.memory_info()
                        memory_info['rss_mb'] = round(mem.rss / (1024 * 1024), 2)
                        memory_info['vms_mb'] = round(mem.vms / (1024 * 1024), 2)
                        memory_info['percent'] = round(proc.memory_percent(), 2)
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            self.logger.debug(f"Could not get memory usage: {e}")
        
        return memory_info
    
    def run_completion(self, prompt: Dict) -> Dict:
        """
        Run a single completion request and measure performance.
        
        Args:
            prompt: Prompt dictionary
        
        Returns:
            Metrics dictionary
        """
        prompt_id = prompt.get('id', 'unknown')
        prompt_text = prompt.get('text', '')
        category = prompt.get('category', 'unknown')
        mandatory = prompt.get('mandatory', False)
        
        # Prepare request
        payload = {
            "prompt": prompt_text,
            "n_predict": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
            "cache_prompt": True
        }
        
        # Initialize metrics
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'threads': self.num_threads,
            'prompt_id': prompt_id,
            'category': category,
            'mandatory': mandatory,
            'prompt_text_preview': prompt_text[:150] + "...",
            'input_tokens_est': len(prompt_text.split()),
            'output_tokens': 0,
            'ttft_ms': 0,
            'tpot_values_ms': [],
            'tpot_avg_ms': 0,
            'tpot_median_ms': 0,
            'tpot_std_ms': 0,
            'total_time_ms': 0,
            'prefill_throughput_tps': 0,
            'decode_throughput_tps': 0,
            'overall_throughput_tps': 0,
            'memory_rss_mb': 0,
            'memory_vms_mb': 0,
            'generated_text': '',
            'error': None
        }
        
        # Timing
        request_start = time.perf_counter()
        first_token_time = None
        last_token_time = None
        token_intervals = []
        generated_text = ""
        
        try:
            response = requests.post(
                f"{self.server_url}/completion",
                json=payload,
                stream=True,
                timeout=300
            )
            
            if response.status_code != 200:
                metrics['error'] = f"HTTP {response.status_code}"
                return metrics
            
            # Process streaming tokens
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                
                try:
                    if line.startswith('data: '):
                        data = json.loads(line[6:])
                    else:
                        data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                if data.get('stop', False):
                    break

                chunk_text = self._extract_response_text(data)
                if chunk_text:
                    generated_text += chunk_text
                
                current_time = time.perf_counter()
                
                if first_token_time is None:
                    first_token_time = current_time
                    metrics['ttft_ms'] = round((first_token_time - request_start) * 1000, 2)
                else:
                    interval = (current_time - last_token_time) * 1000
                    token_intervals.append(interval)
                
                last_token_time = current_time
            
            # Calculate final metrics
            end_time = time.perf_counter()
            metrics['total_time_ms'] = round((end_time - request_start) * 1000, 2)
            metrics['output_tokens'] = len(token_intervals) + 1
            
            # Calculate TPOT statistics
            if token_intervals:
                metrics['tpot_values_ms'] = [round(t, 2) for t in token_intervals]
                metrics['tpot_avg_ms'] = round(statistics.mean(token_intervals), 2)
                metrics['tpot_median_ms'] = round(statistics.median(token_intervals), 2)
                metrics['tpot_std_ms'] = round(statistics.stdev(token_intervals), 2) if len(token_intervals) > 1 else 0
                
                decode_time_sec = sum(token_intervals) / 1000
                metrics['decode_throughput_tps'] = round(metrics['output_tokens'] / decode_time_sec, 2) if decode_time_sec > 0 else 0
            
            # Prefill throughput
            if metrics['ttft_ms'] > 0:
                prefill_time_sec = metrics['ttft_ms'] / 1000
                metrics['prefill_throughput_tps'] = round(metrics['input_tokens_est'] / prefill_time_sec, 2)
            
            # Overall throughput
            if metrics['total_time_ms'] > 0:
                total_time_sec = metrics['total_time_ms'] / 1000
                total_tokens = metrics['input_tokens_est'] + metrics['output_tokens']
                metrics['overall_throughput_tps'] = round(total_tokens / total_time_sec, 2)
            
            # Memory
            mem_info = self.get_memory_usage()
            metrics['memory_rss_mb'] = mem_info['rss_mb']
            metrics['memory_vms_mb'] = mem_info['vms_mb']
            
            # Store truncated output
            metrics['generated_text'] = generated_text[:300]
            
            self.logger.info(f"✓ {prompt_id}: TTFT={metrics['ttft_ms']}ms, "
                           f"TPOT={metrics['tpot_avg_ms']}ms, "
                           f"Tokens={metrics['output_tokens']}")
            
        except requests.exceptions.Timeout:
            metrics['error'] = "Timeout"
            self.logger.error(f"✗ {prompt_id}: Timeout")
        except Exception as e:
            metrics['error'] = str(e)[:200]
            self.logger.error(f"✗ {prompt_id}: {e}")
        
        return metrics
    
    def run_warmup(self):
        """Run warmup requests."""
        self.logger.info("Running warmup...")
        
        warmup_prompts = [
            {"text": "Hello, how are you?"},
            {"text": "What is 2+2?"},
            {"text": "Tell me a short joke."}
        ]
        
        for i, prompt in enumerate(warmup_prompts):
            self.logger.info(f"Warmup {i+1}/{len(warmup_prompts)}")
            prompt['id'] = f'warmup_{i}'
            prompt['category'] = 'warmup'
            prompt['mandatory'] = False
            self.run_completion(prompt)
            time.sleep(0.3)
        
        self.logger.info("Warmup complete\n")
    
    def run_benchmark(self, num_trials: int = 3):
        """Run the complete benchmark."""
        self.logger.info(f"{'='*60}")
        self.logger.info(f"BENCHMARK: {self.model_name} ({num_trials} trials)")
        self.logger.info(f"{'='*60}\n")
        
        # Get memory before starting
        mem_before = self.get_memory_usage()
        self.logger.info(f"Memory before benchmark: {mem_before['rss_mb']:.1f} MB RSS")
        
        all_results = []
        
        for trial in range(num_trials):
            self.logger.info(f"\n--- Trial {trial + 1}/{num_trials} ---")
            
            trial_results = []
            
            for i, prompt in enumerate(self.prompts):
                # Run completion
                metrics = self.run_completion(prompt)
                metrics['trial'] = trial + 1
                trial_results.append(metrics)
                
                # Brief pause between requests
                time.sleep(0.3)
            
            # Save trial results
            trial_file = self.results_dir / f"trial_{trial + 1}.json"
            with open(trial_file, 'w') as f:
                json.dump(trial_results, f, indent=2)
            self.logger.info(f"Trial {trial + 1} saved to {trial_file}")
            
            all_results.extend(trial_results)
        
        # Save all raw results
        raw_file = self.results_dir / "raw_results.json"
        with open(raw_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Compute and save aggregated statistics
        self.results = all_results
        self._compute_statistics()
        
        self.logger.info(f"\n✓ Benchmark complete. Results in: {self.results_dir}")
    
    def _compute_statistics(self):
        """Compute and save aggregated statistics."""
        from collections import defaultdict
        
        # Group by prompt_id
        grouped = defaultdict(list)
        for r in self.results:
            if not r.get('error'):
                grouped[r['prompt_id']].append(r)
        
        # Compute statistics per prompt
        stats = []
        for prompt_id, results in grouped.items():
            if not results:
                continue
            
            cat = results[0]['category']
            mandatory = results[0].get('mandatory', False)
            
            # Extract values
            ttft_vals = [r['ttft_ms'] for r in results if r['ttft_ms'] > 0]
            tpot_vals = [r['tpot_avg_ms'] for r in results if r['tpot_avg_ms'] > 0]
            decode_vals = [r['decode_throughput_tps'] for r in results if r['decode_throughput_tps'] > 0]
            mem_vals = [r['memory_rss_mb'] for r in results if r['memory_rss_mb'] > 0]
            tokens_vals = [r['output_tokens'] for r in results]
            
            stat = {
                'prompt_id': prompt_id,
                'category': cat,
                'mandatory': mandatory,
                'num_trials': len(results),
                'ttft_mean_ms': round(statistics.mean(ttft_vals), 2) if ttft_vals else 0,
                'ttft_std_ms': round(statistics.stdev(ttft_vals), 2) if len(ttft_vals) > 1 else 0,
                'tpot_mean_ms': round(statistics.mean(tpot_vals), 2) if tpot_vals else 0,
                'tpot_std_ms': round(statistics.stdev(tpot_vals), 2) if len(tpot_vals) > 1 else 0,
                'decode_tps_mean': round(statistics.mean(decode_vals), 2) if decode_vals else 0,
                'decode_tps_std': round(statistics.stdev(decode_vals), 2) if len(decode_vals) > 1 else 0,
                'memory_mb_mean': round(statistics.mean(mem_vals), 2) if mem_vals else 0,
                'output_tokens_mean': round(statistics.mean(tokens_vals), 0),
            }
            stats.append(stat)
        
        # Save as JSON
        stats_file = self.results_dir / "statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save as CSV
        csv_file = self.results_dir / "statistics.csv"
        if stats:
            fieldnames = [
                'prompt_id', 'category', 'mandatory', 'num_trials',
                'ttft_mean_ms', 'ttft_std_ms',
                'tpot_mean_ms', 'tpot_std_ms',
                'decode_tps_mean', 'decode_tps_std',
                'memory_mb_mean', 'output_tokens_mean'
            ]
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(stats)
        
        # Print summary
        self._print_summary(stats)
        
        self.logger.info(f"Statistics saved to {stats_file} and {csv_file}")
    
    def _print_summary(self, stats: List[Dict]):
        """Print formatted summary."""
        from collections import defaultdict
        
        by_category = defaultdict(list)
        for s in stats:
            by_category[s['category']].append(s)
        
        print(f"\n{'='*75}")
        print(f"BENCHMARK SUMMARY - {self.model_name} ({self.num_threads} threads)")
        print(f"{'='*75}")
        print(f"{'Category':<12} {'TTFT(ms)':<15} {'TPOT(ms)':<15} {'Decode(t/s)':<15} {'Memory(MB)':<12}")
        print("-" * 69)
        
        for cat in ['short', 'medium', 'long']:
            if cat in by_category:
                items = by_category[cat]
                ttft = statistics.mean([s['ttft_mean_ms'] for s in items])
                tpot = statistics.mean([s['tpot_mean_ms'] for s in items])
                decode = statistics.mean([s['decode_tps_mean'] for s in items])
                mem = statistics.mean([s['memory_mb_mean'] for s in items])
                print(f"{cat:<12} {ttft:<15.1f} {tpot:<15.1f} {decode:<15.1f} {mem:<12.1f}")
        
        print(f"{'='*75}\n")


def main():
    """Main entry point - called from SLURM script."""
    parser = argparse.ArgumentParser(
        description="LLM Inference Benchmark for llama.cpp - Track A1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From project root
  python scripts/benchmark_llm.py --model llama-3.1-8b --threads 16
  
  # With specific trials and custom server
  python scripts/benchmark_llm.py --model qwen2.5-1.5b --threads 8 --trials 3 --port 8081
  
  # Minimal run for testing
  python scripts/benchmark_llm.py --model tinyllama-1.1b --threads 4 --trials 1 --no-warmup
        """
    )
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                       help='Model name (e.g., llama-3.1-8b, qwen2.5-1.5b)')
    
    # Optional arguments with sensible defaults
    parser.add_argument('--project-root', type=str, default='.',
                       help='Project root directory (default: current dir)')
    parser.add_argument('--threads', type=int, default=16,
                       help='Number of CPU threads used by server (default: 16)')
    parser.add_argument('--trials', type=int, default=3,
                       help='Number of trials per prompt (default: 3)')
    parser.add_argument('--port', type=int, default=8080,
                       help='Server port (default: 8080)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Server host (default: localhost)')
    parser.add_argument('--max-tokens', type=int, default=512,
                       help='Maximum tokens to generate (default: 512)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature (default: 0.7)')
    parser.add_argument('--no-warmup', action='store_true',
                       help='Skip warmup phase')
    
    args = parser.parse_args()
    
    # Build server URL
    server_url = f"http://{args.host}:{args.port}"
    
    # Resolve project root
    project_root = Path(args.project_root).resolve()
    
    print(f"\n{'='*60}")
    print(f"LLM Inference Benchmark - Track A1")
    print(f"{'='*60}")
    print(f"Project root: {project_root}")
    print(f"Model:        {args.model}")
    print(f"Server:       {server_url}")
    print(f"Threads:      {args.threads}")
    print(f"Trials:       {args.trials}")
    print(f"Max tokens:   {args.max_tokens}")
    print(f"{'='*60}\n")
    
    # Create and run benchmark
    benchmark = LLMBenchmark(
        model_name=args.model,
        project_root=str(project_root),
        server_url=server_url,
        num_threads=args.threads,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    # Check server
    if not benchmark.check_server_health():
        print("\nERROR: llama.cpp server is not running!")
        print(f"Start it with: llama-server -m models/<model>.gguf -c 4096 -t {args.threads}")
        sys.exit(1)
    
    # Warmup
    if not args.no_warmup:
        benchmark.run_warmup()
    
    # Run benchmark
    try:
        benchmark.run_benchmark(num_trials=args.trials)
    except KeyboardInterrupt:
        print("\nInterrupted. Saving partial results...")
        if benchmark.results:
            benchmark._compute_statistics()
        sys.exit(0)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()