#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
# ]
# ///
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def normalize_benchmark_name(name):
    """Convert FixtureName/BenchmarkName/... to FixtureName_BenchmarkName/..."""
    parts = name.split("/")
    if len(parts) >= 2 and "Fixture" in parts[0]:
        # Convert "CFFixture/Insert/..." to "CF_Insert/..."
        fixture_name = parts[0].replace("Fixture", "")
        bench_name = parts[1]
        parts[0] = f"{fixture_name}_{bench_name}"
        parts.pop(1)  # Remove the benchmark name since it's now in parts[0]
    return "/".join(parts)


def main():
    try:
        df = pd.read_csv(sys.stdin)
    except Exception as e:
        print(f"Error parsing CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # Filter for median records only
    df = df[df["name"].str.endswith("_median")]

    benchmark_data = defaultdict(dict)
    for _, row in df.iterrows():
        name = normalize_benchmark_name(row["name"])
        if "/" not in name:
            continue

        # Remove _median suffix and extract base_name and size
        # Format: "CF_Insert/65536/min_time:0.500/repeats:10_median"
        parts = name.split("/")
        if len(parts) < 2:
            continue

        base_name = parts[0]
        size_str = parts[1]

        suffix = base_name.rsplit("_", 1)[-1]
        if not re.fullmatch(r"(?:Query|Insert)(?:AddSub)?(<\d+>)?", suffix):
            continue

        try:
            size = int(size_str)
            items_per_second = row.get("items_per_second")
            if pd.notna(items_per_second):
                throughput_mops = items_per_second / 1_000_000
                benchmark_data[base_name][size] = throughput_mops
        except (ValueError, KeyError):
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
    ax.set_ylabel("Throughput (MOPS)", fontsize=12)
    ax.set_xscale("log", base=2)
    # ax.set_yscale("log")
    ax.legend(fontsize=10, loc="best", ncol=2)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.set_title("Throughput Comparison", fontsize=14)
    plt.tight_layout()

    output_file = build_dir / "benchmark_throughput.png"
    plt.savefig(output_file, dpi=150)
    print(f"Throughput plot saved to {output_file}")


if __name__ == "__main__":
    main()
