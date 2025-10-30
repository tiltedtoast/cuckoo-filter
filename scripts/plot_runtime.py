#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
# ]
# ///

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr)
        sys.exit(1)

    benchmark_data = defaultdict(dict)

    for benchmark in data.get("benchmarks", []):
        name = benchmark.get("name", "")
        if "/" not in name:
            continue

        base_name, size_str = name.rsplit("/", 1)

        if "FalsePositiveRate" in base_name or "InsertQueryDelete" in base_name:
            continue

        try:
            size = int(size_str)
            real_time = benchmark.get("real_time", 0)
            benchmark_data[base_name][size] = real_time
        except ValueError:
            continue

    if not benchmark_data:
        print("No benchmark data found in JSON", file=sys.stderr)
        sys.exit(1)

    def get_last_value(bench_name):
        sizes = sorted(benchmark_data[bench_name].keys())
        if sizes:
            return benchmark_data[bench_name][sizes[-1]]
        return 0

    benchmark_names = sorted(benchmark_data.keys(), key=get_last_value, reverse=True)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle("Runtime Comparison", fontsize=16)

    for bench_name in benchmark_names:
        sizes = sorted(benchmark_data[bench_name].keys())
        times = [benchmark_data[bench_name][size] for size in sizes]

        ax.plot(sizes, times, "o-", label=bench_name, linewidth=2, markersize=6)

    ax.set_xlabel("Input Size", fontsize=12)
    ax.set_ylabel("Runtime (ms)", fontsize=12)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()

    script_dir = Path(__file__).parent
    build_dir = script_dir.parent / "build"
    build_dir.mkdir(exist_ok=True)

    output_file = build_dir / "benchmark_runtime.png"
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to {output_file}")


if __name__ == "__main__":
    main()
