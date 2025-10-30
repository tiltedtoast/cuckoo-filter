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

    fpr_data = defaultdict(dict)
    false_positives_data = defaultdict(dict)

    for benchmark in data.get("benchmarks", []):
        name = benchmark.get("name", "")
        if "/" not in name:
            continue

        base_name, size_str = name.rsplit("/", 1)

        if "FalsePositiveRate" not in base_name:
            continue

        try:
            size = int(size_str)
            fpr_percentage = None
            false_positives = None

            for counter_name, counter_value in benchmark.items():
                if counter_name == "fpr_percentage":
                    fpr_percentage = counter_value
                elif counter_name == "false_positives":
                    false_positives = counter_value

            if fpr_percentage is not None:
                fpr_data[base_name][size] = fpr_percentage
            if false_positives is not None:
                false_positives_data[base_name][size] = false_positives

        except ValueError:
            continue

    if not fpr_data and not false_positives_data:
        print("No false positive rate data found in JSON", file=sys.stderr)
        sys.exit(1)

    script_dir = Path(__file__).parent
    build_dir = script_dir.parent / "build"
    build_dir.mkdir(exist_ok=True)

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    if fpr_data:

        def get_last_fpr_value(bench_name):
            sizes = sorted(fpr_data[bench_name].keys())
            if sizes:
                return fpr_data[bench_name][sizes[-1]]
            return 0

        benchmark_names = sorted(fpr_data.keys(), key=get_last_fpr_value, reverse=True)

        for bench_name in benchmark_names:
            sizes = sorted(fpr_data[bench_name].keys())
            fpr = [fpr_data[bench_name][size] for size in sizes]
            ax1.plot(sizes, fpr, "o-", label=bench_name, linewidth=2, markersize=6)

        ax1.set_xlabel("Input Size", fontsize=12)
        ax1.set_ylabel("False Positive Rate (%)", fontsize=12)
        ax1.set_xscale("log", base=2)
        ax1.legend(fontsize=10, loc="best")
        ax1.grid(True, which="both", ls="--", alpha=0.5)
        ax1.set_title("False Positive Rate Percentage", fontsize=14)

    if false_positives_data:
        def get_last_fp_value(bench_name):
            sizes = sorted(false_positives_data[bench_name].keys())
            if sizes:
                return false_positives_data[bench_name][sizes[-1]]
            return 0

        benchmark_names = sorted(
            false_positives_data.keys(), key=get_last_fp_value, reverse=True
        )

        for bench_name in benchmark_names:
            sizes = sorted(false_positives_data[bench_name].keys())
            fp = [false_positives_data[bench_name][size] for size in sizes]
            ax2.plot(sizes, fp, "o-", label=bench_name, linewidth=2, markersize=6)

        ax2.set_xlabel("Input Size", fontsize=12)
        ax2.set_ylabel("Total False Positives", fontsize=12)
        ax2.set_xscale("log", base=2)
        ax2.set_yscale("log")
        ax2.legend(fontsize=10, loc="best")
        ax2.grid(True, which="both", ls="--", alpha=0.5)
        ax2.set_title("Total False Positives Count", fontsize=14)

    plt.tight_layout()
    output_file = build_dir / "benchmark_false_positives.png"
    plt.savefig(output_file, dpi=150)
    print(f"False positive plot saved to {output_file}")


if __name__ == "__main__":
    main()
