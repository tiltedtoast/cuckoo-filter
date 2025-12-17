#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "typer",
# ]
# ///
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer

app = typer.Typer(help="Plot memory usage benchmark results")


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
    Generate memory usage plots from benchmark CSV results.

    Creates two plots: total memory usage and bits per item efficiency metric.

    Examples:
        cat results.csv | plot_memory.py
        plot_memory.py < results.csv
        plot_memory.py results.csv
        plot_memory.py results.csv -o custom/dir
    """
    df = pu.load_csv(csv_file)

    # Filter for median records only
    df = df[df["name"].str.endswith("_median")]

    memory_data = defaultdict(dict)
    bits_per_item_data = defaultdict(dict)

    for _, row in df.iterrows():
        name = pu.normalize_benchmark_name(row["name"])
        if "/" not in name:
            continue

        # Extract base_name and size from name
        parts = name.split("/")
        if len(parts) < 2:
            continue

        base_name = parts[0]
        size_str = parts[1]

        if (
            "Insert" not in base_name
            or "InsertAndQuery" in base_name
            or "InsertQueryDelete" in base_name
            or "FalsePositiveRate" in base_name
        ):
            continue

        try:
            size = int(size_str)

            memory_bytes = row.get("memory_bytes")
            bits_per_item = row.get("bits_per_item")

            if pd.notna(memory_bytes):
                memory_data[base_name][size] = memory_bytes
            if pd.notna(bits_per_item):
                bits_per_item_data[base_name][size] = bits_per_item

        except (ValueError, KeyError):
            continue

    if not memory_data and not bits_per_item_data:
        typer.secho("No memory data found in CSV", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

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

            ax1.plot(sizes, memory, "o-", label=bench_name, linewidth=2.5, markersize=8)

        pu.format_axis(
            ax1,
            xlabel="Input Size",
            ylabel="Memory Usage (MiB)",
            title="Total Memory Usage",
            xscale="log",
            yscale="log",
        )

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

            ax2.plot(sizes, bpi, "o-", label=bench_name, linewidth=2.5, markersize=8)

        pu.format_axis(
            ax2,
            xlabel="Input Size",
            ylabel="Bits Per Item",
            title="Memory Efficiency (Bits Per Item)",
            xscale="log",
        )

    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))
    output_file = output_dir / "benchmark_memory.pdf"
    pu.save_figure(None, output_file, f"Memory plot saved to {output_file}")


if __name__ == "__main__":
    app()
