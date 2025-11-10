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

    memory_data = defaultdict(dict)
    bits_per_item_data = defaultdict(dict)

    for benchmark in data.get("benchmarks", []):
        name = benchmark.get("name", "")
        if "/" not in name:
            continue

        base_name, size_str = name.rsplit("/", 1)
        if (
            "Insert" not in base_name
            or "InsertAndQuery" in base_name
            or "InsertQueryDelete" in base_name
            or "FalsePositiveRate" in base_name
        ):
            continue

        try:
            size = int(size_str)

            memory_bytes = None
            bits_per_item = None

            for counter_name, counter_value in benchmark.items():
                if counter_name == "memory_bytes":
                    memory_bytes = counter_value
                elif counter_name == "bits_per_item":
                    bits_per_item = counter_value

            if memory_bytes is not None:
                memory_data[base_name][size] = memory_bytes
            if bits_per_item is not None:
                bits_per_item_data[base_name][size] = bits_per_item

        except ValueError:
            continue

    if not memory_data and not bits_per_item_data:
        print("No memory data found in JSON", file=sys.stderr)
        sys.exit(1)

    script_dir = Path(__file__).parent
    build_dir = script_dir.parent / "build"
    build_dir.mkdir(exist_ok=True)

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    if memory_data:

        def get_last_memory_value(bench_name):
            sizes = sorted(memory_data[bench_name].keys())
            if sizes:
                return memory_data[bench_name][sizes[-1]]
            return 0

        benchmark_names = sorted(
            memory_data.keys(), key=get_last_memory_value, reverse=True
        )
        for bench_name in benchmark_names:
            sizes = sorted(memory_data[bench_name].keys())
            memory = [memory_data[bench_name][size] for size in sizes]

            ax1.plot(sizes, memory, "o-", label=bench_name, linewidth=2, markersize=6)

        ax1.set_xlabel("Input Size", fontsize=12)
        ax1.set_ylabel("Memory Usage (MiB)", fontsize=12)
        ax1.set_xscale("log", base=2)
        ax1.set_yscale("log")
        ax1.legend(fontsize=10, loc="best")
        ax1.grid(True, which="both", ls="--", alpha=0.5)
        ax1.set_title("Total Memory Usage", fontsize=14)

    if bits_per_item_data:

        def get_last_bpi_value(bench_name):
            sizes = sorted(bits_per_item_data[bench_name].keys())
            if sizes:
                return bits_per_item_data[bench_name][sizes[-1]]
            return 0

        benchmark_names = sorted(
            bits_per_item_data.keys(), key=get_last_bpi_value, reverse=True
        )
        for bench_name in benchmark_names:
            sizes = sorted(bits_per_item_data[bench_name].keys())
            bpi = [bits_per_item_data[bench_name][size] for size in sizes]

            ax2.plot(sizes, bpi, "o-", label=bench_name, linewidth=2, markersize=6)

        ax2.set_xlabel("Input Size", fontsize=12)
        ax2.set_ylabel("Bits Per Item", fontsize=12)
        ax2.set_xscale("log", base=2)
        ax2.legend(fontsize=10, loc="best")
        ax2.grid(True, which="both", ls="--", alpha=0.5)
        ax2.set_title("Memory Efficiency (Bits Per Item)", fontsize=14)

    plt.tight_layout()

    output_file = build_dir / "benchmark_memory.png"
    plt.savefig(output_file, dpi=150)
    print(f"Memory plot saved to {output_file}")


if __name__ == "__main__":
    main()
