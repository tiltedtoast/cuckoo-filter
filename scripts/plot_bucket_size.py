#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "seaborn",
#   "typer",
# ]
# ///
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_utils as pu
import seaborn as sns
import typer

app = typer.Typer(help="Plot bucket size benchmark results")


def parse_benchmark_name(name: str) -> pd.Series:
    # Pattern: BSFixture<BucketSize>/<Operation>/<InputSize>/min_time:<MinTime>/repeats:<Repetitions>_<stat>
    # Example: BSFixture4/Insert/65536/min_time:0.500/repeats:5_median
    # Note: bucket_size is already in the CSV, so we don't need to extract it
    match = re.match(r"BSFixture\d+/(\w+)/(\d+)", name)
    if match:
        operation = match.group(1)
        input_size = int(match.group(2))
        return pd.Series(
            {
                "operation": operation,
                "input_size": input_size,
                "exponent": int(np.log2(input_size)),
            }
        )
    return pd.Series(
        {
            "operation": None,
            "input_size": None,
            "exponent": None,
        }
    )


def create_performance_heatmap(df: pd.DataFrame, operation: str, ax):
    subset = df[df["operation"] == operation].copy()

    # First create the pivot table with raw time_ms values
    pivot_table = subset.pivot(
        index="exponent",
        columns="bucket_size",
        values="time_ms",
    )

    # Normalize each row by dividing by minimum value in that row
    normalized_table = pivot_table.div(pivot_table.min(axis=1), axis=0)

    sns.heatmap(
        normalized_table,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="rocket_r",
        cbar_kws={"label": "Performance Ratio (1.0 = Optimal)"},
        vmin=1.0,
        vmax=normalized_table.max().max()
        if normalized_table.max().max() > 1.0
        else 2.0,
    )

    ax.set_title(
        f"{operation} Performance vs. Bucket Size", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Bucket Size", fontsize=12)
    ax.set_ylabel("Input Size", fontsize=12)
    ax.set_yticklabels(
        [f"$2^{{{int(exp)}}}$" for exp in normalized_table.index], rotation=0
    )


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
    Generate bucket size performance heatmap plots from benchmark CSV results.

    Shows normalized performance for Insert and Query operations across different
    bucket sizes and input sizes.

    Examples:
        cat results.csv | plot_bucket_size.py
        plot_bucket_size.py < results.csv
        plot_bucket_size.py results.csv
        plot_bucket_size.py results.csv -o custom/dir
    """
    df = pu.load_csv(csv_file)

    # Filter for median records only
    df = df[df["name"].str.endswith("_median")]

    parsed = df["name"].apply(parse_benchmark_name)
    df = pd.concat([df, parsed], axis=1)

    # bucket_size is now extracted from the benchmark name (BSFixture<N>)

    df_filtered = df[df["operation"].isin(["Insert", "Query"])].copy()

    df_filtered["time_ms"] = df_filtered["real_time"]
    df_filtered["throughput_bops"] = df_filtered["items_per_second"] / 1_000_000_000

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    create_performance_heatmap(df_filtered, "Insert", ax1)
    create_performance_heatmap(df_filtered, "Query", ax2)

    # Determine output directory
    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))

    output_file = output_dir / "benchmark_bucket_size.pdf"

    pu.save_figure(
        None, output_file, f"Bucket size performance plot saved to {output_file}"
    )


if __name__ == "__main__":
    app()
    app()
