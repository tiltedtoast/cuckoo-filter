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

app = typer.Typer(help="Plot runtime benchmark results")


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
    Generate runtime comparison plots from benchmark CSV results.

    Shows execution time vs input size for various benchmarks.

    Examples:
        cat results.csv | plot_runtime.py
        plot_runtime.py < results.csv
        plot_runtime.py results.csv
        plot_runtime.py results.csv -o custom/dir
    """
    df = pu.load_csv(csv_file)

    # Filter for median records only
    df = df[df["name"].str.endswith("_median")]

    benchmark_data = defaultdict(dict)

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

        if "FPR" in base_name or "InsertQueryDelete" in base_name:
            continue

        try:
            size = int(size_str)
            real_time = row.get("real_time", 0)
            if pd.notna(real_time):
                benchmark_data[base_name][size] = real_time
        except (ValueError, KeyError):
            continue

    if not benchmark_data:
        typer.secho("No benchmark data found in CSV", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    def get_last_value(bench_name):
        sizes = sorted(benchmark_data[bench_name].keys())
        if sizes:
            return benchmark_data[bench_name][sizes[-1]]
        return 0

    benchmark_names = sorted(benchmark_data.keys(), key=get_last_value, reverse=True)

    fig, ax = pu.setup_figure(
        figsize=(12, 8),
        title="Runtime Comparison",
    )

    for bench_name in benchmark_names:
        sizes = sorted(benchmark_data[bench_name].keys())
        times = [benchmark_data[bench_name][size] for size in sizes]

        ax.plot(sizes, times, "o-", label=bench_name, linewidth=2.5, markersize=8)

    pu.format_axis(
        ax,
        xlabel="Input Size",
        ylabel="Runtime (ms)",
        xscale="log",
        yscale="log",
    )

    plt.tight_layout()

    pu.format_axis(
        ax, xlabel="Input Size", ylabel="Runtime (ms)", xscale="log", yscale="log"
    )

    # Determine output directory
    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))

    output_file = output_dir / "benchmark_runtime.pdf"
    pu.save_figure(None, output_file, f"Plot saved to {output_file}")


if __name__ == "__main__":
    app()
