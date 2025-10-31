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
import re
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
        suffix = base_name.rsplit("_", 1)[-1]
        if not re.fullmatch(r"(?:Query|Insert)(<\d+>)?", suffix):
            continue

        try:
            size = int(size_str)
            items_per_second = benchmark.get("items_per_second")
            if items_per_second is not None:
                throughput_mops = items_per_second / 1_000_000
                benchmark_data[base_name][size] = throughput_mops
        except ValueError:
            continue

    if not benchmark_data:
        print("No throughput data found in JSON", file=sys.stderr)
        sys.exit(1)

    script_dir = Path(__file__).parent
    build_dir = script_dir.parent / "build"
    build_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 8))

    def get_last_throughput(bench_name):
        sizes = sorted(benchmark_data[bench_name].keys())
        if sizes:
            return benchmark_data[bench_name][sizes[-1]]
        return 0

    benchmark_names = sorted(
        benchmark_data.keys(), key=get_last_throughput, reverse=True
    )

    for bench_name in benchmark_names:
        sizes = sorted(benchmark_data[bench_name].keys())
        throughput = [benchmark_data[bench_name][size] for size in sizes]
        ax.plot(sizes, throughput, "o-", label=bench_name, linewidth=2, markersize=6)

    ax.set_xlabel("Input Size", fontsize=12)
    ax.set_ylabel("Throughput (Million ops/sec)", fontsize=12)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=10, loc="best", ncol=2)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.set_title("Throughput Comparison", fontsize=14)
    plt.tight_layout()

    output_file = build_dir / "benchmark_throughput.png"
    plt.savefig(output_file, dpi=150)
    print(f"Throughput plot saved to {output_file}")


if __name__ == "__main__":
    main()
