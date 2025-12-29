#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "typer",
# ]
# ///
"""Plot k-mer benchmark results comparing GPU filter Insert and Query performance."""

import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer

app = typer.Typer(help="Plot k-mer benchmark results")


def extract_filter_and_operation(name: str) -> tuple[Optional[str], Optional[str]]:
    """Extract filter type and operation from benchmark name like 'GPUCF_Insert'."""
    match = re.match(r"(\w+)_(Insert|Query)", name)
    if match:
        return match.group(1), match.group(2)
    return None, None


def load_csv_data(csv_path: Path) -> pd.DataFrame:
    """Load and parse benchmark data from CSV."""
    df = pu.load_csv(csv_path)

    # Filter for median records only
    df = df[df["name"].str.endswith("_median")]

    results = []
    for _, row in df.iterrows():
        name = row["name"].replace("_median", "")
        filter_type, operation = extract_filter_and_operation(name)

        if filter_type is None or operation is None:
            continue

        items_per_second = row.get("items_per_second")
        if pd.notna(items_per_second):
            throughput_mops = items_per_second / 1_000_000
            results.append(
                {
                    "filter": filter_type,
                    "operation": operation,
                    "throughput": throughput_mops,
                }
            )

    return pd.DataFrame(results)


@app.command()
def main(
    csv_file: Path = typer.Argument(..., help="Path to k-mer benchmark CSV file"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Output directory for plots"
    ),
    dataset_name: Optional[str] = typer.Option(
        None, "--dataset", "-d", help="Dataset name for title (e.g., 'E. coli')"
    ),
    k: Optional[int] = typer.Option(
        None, "--k", "-k", help="K-mer size for title (e.g., 21)"
    ),
):
    """
    Plot k-mer benchmark results as clustered bar charts.

    Examples:
        ./plot_kmer.py results.csv
        ./plot_kmer.py results.csv -d "E. coli" -k 21
        ./plot_kmer.py results.csv -o custom/dir -d "Human chr14" -k 31
    """
    df = load_csv_data(csv_file)

    if df.empty:
        typer.secho("No valid data found in CSV", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))

    # Get unique filters and operations from data
    filters = df["filter"].unique().tolist()
    operations = ["Insert", "Query"]

    fig, ax = pu.setup_figure(figsize=(10, 6))

    bar_width = 0.35
    x_positions = range(len(filters))

    for i, operation in enumerate(operations):
        throughputs = []
        for filter_name in filters:
            subset = df[(df["filter"] == filter_name) & (df["operation"] == operation)]
            if not subset.empty:
                throughputs.append(subset["throughput"].values[0])
            else:
                throughputs.append(0)

        offset = (i - 0.5) * bar_width
        bars = ax.bar(
            [x + offset for x in x_positions],
            throughputs,
            bar_width,
            label=operation,
            color=pu.OPERATION_COLORS.get(operation, "#333333"),
            edgecolor="black",
            linewidth=0.5,
        )

        # Add value labels on bars
        for bar, val in zip(bars, throughputs):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=pu.LEGEND_FONT_SIZE,
                    fontweight="bold",
                )

    title = "K-mer Benchmark"
    if dataset_name and k:
        title += f"\n{dataset_name} (k={k})"
    elif dataset_name:
        title += f"\n{dataset_name}"
    elif k:
        title += f"\n(k={k})"

    # Get display names for filters
    filter_labels = [pu.get_filter_display_name(f) for f in filters]

    pu.format_axis(
        ax,  # ty:ignore[invalid-argument-type]
        "Filter",
        "Throughput [M ops/s]",
        title=title,
        xscale=None,
        grid=True,
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(filter_labels, fontsize=pu.DEFAULT_FONT_SIZE)
    pu.create_legend(
        ax,  # ty:ignore[invalid-argument-type]
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=pu.DEFAULT_FONT_SIZE,
    )

    plt.tight_layout()

    output_file = output_dir / "kmer_benchmark.pdf"
    pu.save_figure(fig, output_file, f"K-mer benchmark plot saved to {output_file}")


if __name__ == "__main__":
    app()
