#!/usr/bin/env python3
import psutil
import time
import csv
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Background Resource Monitor")
    parser.add_argument('--output-dir', required=True, help="Directory to save CSVs")
    parser.add_argument('--prefix', required=True, help="Prefix for CSV files")
    parser.add_argument('--interval', type=float, default=1.0, help="Sampling interval in seconds")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cpu_file = open(out_dir / f"{args.prefix}_cpu_1.csv", 'w', newline='')
    mem_file = open(out_dir / f"{args.prefix}_memory_1.csv", 'w', newline='')

    cpu_writer = csv.writer(cpu_file)
    mem_writer = csv.writer(mem_file)

    # Write headers exactly as expected by analyze_results.py
    cpu_writer.writerow(['user', 'system', 'iowait'])
    mem_writer.writerow(['used_mb', 'cached_mb', 'free_mb'])

    print(f"Monitoring resources every {args.interval}s. Saving to {out_dir}...")
    try:
        while True:
            cpu = psutil.cpu_times_percent(interval=args.interval)
            mem = psutil.virtual_memory()

            iowait = getattr(cpu, 'iowait', 0.0)
            
            cpu_writer.writerow([cpu.user, cpu.system, iowait])
            mem_writer.writerow([mem.used / 1024**2, mem.cached / 1024**2, mem.free / 1024**2])

            cpu_file.flush()
            mem_file.flush()
    except KeyboardInterrupt:
        pass
    finally:
        cpu_file.close()
        mem_file.close()

if __name__ == '__main__':
    main()