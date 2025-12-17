#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "typer",
# ]
# ///
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer

app = typer.Typer(help="Plot benchmark throughput results")


@app.command()
def main(
    csv_file: Path = typer.Argument(
        "-",
        help="Path to CSV file, or '-' to read from stdin (default: stdin)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots (default: build/)",
    ),
):
    """
    Generate throughput comparison plots from benchmark CSV results.

    Plots throughput [M ops/s] vs input size for various benchmarks.

    Examples:
        cat results.csv | plot_throughput.py
        plot_throughput.py < results.csv
        plot_throughput.py results.csv
        plot_throughput.py results.csv -o custom/dir
    """
    try:
        if str(csv_file) == "-":
            import sys

            df = pd.read_csv(sys.stdin)
        else:
            df = pd.read_csv(csv_file)
    except Exception as e:
        typer.secho(f"Error parsing CSV: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # Filter for median records only
    df = df[df["name"].str.endswith("_median")]

    benchmark_data = defaultdict(dict)
    for _, row in df.iterrows():
        name = pu.normalize_benchmark_name(row["name"])
        if "/" not in name:
            continue

        # Remove _median suffix and extract base_name and size
        # Format: "CF_Insert/65536/min_time:0.500/repeats:10_median"
        parts = name.split("/")
        if len(parts) < 2:
            continue

        base_name = parts[0]
        size_str = parts[1]

        # Strip CF_ prefix for sorted/unsorted insertion benchmarks
        if base_name in ("CF_InsertSorted", "CF_InsertUnsorted"):
            base_name = base_name[3:]

        suffix = base_name.rsplit("_", 1)[-1]
        if not re.fullmatch(
            r"(?:Query|Insert|Delete)(?:Sorted|Unsorted)?(?:AddSub)?(<\d+>)?",
            suffix,
        ):
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
        typer.secho("No throughput data found in CSV", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))

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
        ax.plot(sizes, throughput, "o-", label=bench_name, linewidth=2.5, markersize=8)

    ax.set_xlabel("Input Size", fontsize=14, fontweight="bold")
    ax.set_ylabel("Throughput [M ops/s]", fontsize=14, fontweight="bold")
    ax.set_xscale("log", base=2)
    # ax.set_yscale("log")
    ax.legend(fontsize=10, loc="best", ncol=2, framealpha=0)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.set_title("Throughput Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()

    output_file = output_dir / "benchmark_throughput.pdf"
    pu.save_figure(None, output_file, f"Throughput plot saved to {output_file}")


if __name__ == "__main__":
    app()
