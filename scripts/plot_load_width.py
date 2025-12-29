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

app = typer.Typer(help="Plot 128-bit vs 256-bit load width benchmark results")


@app.command()
def main(
    csv_file: Path = typer.Argument(
        ...,
        help="Path to CSV file",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots",
    ),
):
    """Generate comparison plot for 128-bit vs 256-bit load widths."""
    df = pu.load_csv(csv_file)
    # Filter for median records only (format: "128bit/Query/65536/.../manual_time_median")
    df = df[df["name"].str.contains("_median")]

    benchmark_data = defaultdict(dict)
    for _, row in df.iterrows():
        name = row["name"]
        parts = name.split("/")
        if len(parts) < 3:
            continue

        # Extract variant (128bit or 256bit) from first part
        variant_raw = parts[0].strip('"')
        if "128bit" in variant_raw:
            variant = "128-bit"
        elif "256bit" in variant_raw:
            variant = "256-bit"
        else:
            continue

        try:
            # Size is the third part (after variant and "Query")
            size = int(parts[2])
            items_per_second = row.get("items_per_second")
            if pd.notna(items_per_second):
                throughput_bops = items_per_second / 1_000_000_000
                benchmark_data[variant][size] = throughput_bops
        except (ValueError, KeyError, IndexError):
            continue

    if not benchmark_data:
        typer.secho("No data found in CSV", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # Calculate speedup of 256-bit over 128-bit (baseline)
    if "128-bit" not in benchmark_data or "256-bit" not in benchmark_data:
        typer.secho(
            "Need both 128-bit and 256-bit data to compute speedup",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    # Find common sizes between both variants
    common_sizes = sorted(
        set(benchmark_data["128-bit"].keys()) & set(benchmark_data["256-bit"].keys())
    )
    if not common_sizes:
        typer.secho(
            "No common sizes found between variants", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(1)

    # Calculate percentage improvement over 128-bit baseline
    percent_improvement = [
        (benchmark_data["256-bit"][s] / benchmark_data["128-bit"][s] - 1) * 100
        for s in common_sizes
    ]

    fig, ax = pu.setup_figure(
        figsize=(10, 6),
        title="Query Speedup: 256-bit vs 128-bit Loads",
    )

    ax.plot(  # ty:ignore[possibly-missing-attribute]
        common_sizes,
        percent_improvement,
        "o-",
        label="256-bit speedup",
        color="#2A9D8F",
        linewidth=2.5,
        markersize=8,
    )

    # Add a reference line at 0% (no improvement)
    ax.axhline(y=0, color="#888888", linestyle="--", linewidth=1.5, alpha=0.7)  # ty:ignore[possibly-missing-attribute]

    # Format y-axis as percentages
    from matplotlib.ticker import FuncFormatter

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))  # ty:ignore[possibly-missing-attribute]

    pu.format_axis(
        ax,  # ty:ignore[invalid-argument-type]
        xlabel="Input Size",
        ylabel="",
        xscale="log",
        yscale=None,
    )
    plt.tight_layout()

    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))
    output_file = output_dir / "load_width_comparison.pdf"
    pu.save_figure(None, output_file, f"Plot saved to {output_file}")


if __name__ == "__main__":
    app()
