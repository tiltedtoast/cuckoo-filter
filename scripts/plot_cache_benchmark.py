#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "typer",
#   "numpy",
# ]
# ///


from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import typer

app = typer.Typer(help="Plot cache benchmark results")


@app.command()
def main(
    csv_file: Path = typer.Argument(
        ...,
        help="Path to CSV file with cache benchmark results",
        exists=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots (default: build/)",
    ),
):
    """
    Generate cache hit rate vs. capacity plots from benchmark CSV results.

    Creates line plots showing how L1 and L2 cache hit rates change as the
    data size increases, helping identify where cache efficiency degrades.

    Examples:
        plot_cache_benchmark.py cache_results.csv
        plot_cache_benchmark.py cache_results.csv -o custom/dir
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        typer.secho(f"Error reading CSV: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # Validate required columns
    required_cols = ["filter", "operation", "capacity", "l1_hit_rate", "l2_hit_rate"]
    if not all(col in df.columns for col in required_cols):
        typer.secho(
            f"CSV missing required columns. Expected: {required_cols}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / "build"

    output_dir.mkdir(parents=True, exist_ok=True)

    filter_styles = {
        "cuckoo": {"color": "#2E86AB", "marker": "o"},
        "bloom": {"color": "#A23B72", "marker": "s"},
        "tcf": {"color": "#C73E1D", "marker": "^"},
    }

    operation_linestyles = {
        "insert": "-",
        "query": "--",
        "delete": ":",
    }

    # Create individual plots for each cache level and operation
    for cache_level, metric_col in [("L1", "l1_hit_rate"), ("L2", "l2_hit_rate")]:
        for operation in df["operation"].unique():
            fig, ax = plt.subplots(figsize=(12, 7))

            operation_df = df[df["operation"] == operation]

            for filter_type in sorted(operation_df["filter"].unique()):
                filter_df = operation_df[operation_df["filter"] == filter_type]

                filter_df = filter_df.sort_values("capacity")

                capacities = filter_df["capacity"].values
                hit_rates = filter_df[metric_col].values

                style = filter_styles.get(filter_type, {})

                ax.plot(
                    capacities,
                    hit_rates,
                    label=filter_type.capitalize(),
                    linewidth=2.5,
                    markersize=8,
                    color=style.get("color"),
                    marker=style.get("marker", "o"),
                    linestyle=operation_linestyles.get(operation, "-"),
                )

            ax.set_xlabel("Filter Capacity (elements)", fontsize=14, fontweight="bold")
            ax.set_ylabel(
                f"{cache_level} Cache Hit Rate (%)", fontsize=14, fontweight="bold"
            )
            ax.set_title(
                f"{cache_level} Cache Hit Rate vs. Capacity ({operation.capitalize()})",
                fontsize=16,
                fontweight="bold",
                pad=20,
            )
            ax.set_xscale("log", base=2)
            ax.grid(True, which="both", ls="--", alpha=0.3)
            ax.legend(fontsize=12, loc="best", framealpha=0.9)
            ax.set_ylim(0, 105)

            plt.tight_layout()

            output_file = output_dir / f"cache_{cache_level.lower()}_{operation}.png"
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            typer.secho(
                f"{cache_level} {operation} plot saved to {output_file}",
                fg=typer.colors.GREEN,
            )
            plt.close()

    # Create combined plots showing all operations for each filter
    for filter_type in df["filter"].unique():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        filter_df = df[df["filter"] == filter_type]

        for ax, cache_level, metric_col in [
            (ax1, "L1", "l1_hit_rate"),
            (ax2, "L2", "l2_hit_rate"),
        ]:
            for operation in sorted(filter_df["operation"].unique()):
                operation_df = filter_df[filter_df["operation"] == operation]
                operation_df = operation_df.sort_values("capacity")

                capacities = operation_df["capacity"].values
                hit_rates = operation_df[metric_col].values

                ax.plot(
                    capacities,
                    hit_rates,
                    label=operation.capitalize(),
                    linewidth=2.5,
                    markersize=8,
                    marker="o",
                    linestyle=operation_linestyles.get(operation, "-"),
                )

            ax.set_xlabel("Capacity (elements)", fontsize=12, fontweight="bold")
            ax.set_ylabel(f"{cache_level} Hit Rate (%)", fontsize=12, fontweight="bold")
            ax.set_title(
                f"{cache_level} Cache ({filter_type.capitalize()})",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xscale("log", base=2)
            ax.grid(True, which="both", ls="--", alpha=0.3)
            ax.legend(fontsize=10, loc="best", framealpha=0.9)
            ax.set_ylim(0, 105)

        plt.suptitle(
            f"Cache Hit Rate vs. Capacity - {filter_type.capitalize()} Filter",
            fontsize=16,
            fontweight="bold",
            y=1.00,
        )
        plt.tight_layout()

        output_file = output_dir / f"cache_{filter_type}_combined.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        typer.secho(
            f"{filter_type.capitalize()} combined plot saved to {output_file}",
            fg=typer.colors.GREEN,
        )
        plt.close()

    # Create an overview plot showing all filters on one chart for insert operation
    for operation in df["operation"].unique():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        operation_df = df[df["operation"] == operation]

        for ax, cache_level, metric_col in [
            (ax1, "L1", "l1_hit_rate"),
            (ax2, "L2", "l2_hit_rate"),
        ]:
            for filter_type in sorted(operation_df["filter"].unique()):
                filter_df = operation_df[operation_df["filter"] == filter_type]
                filter_df = filter_df.sort_values("capacity")

                capacities = filter_df["capacity"].values
                hit_rates = filter_df[metric_col].values

                style = filter_styles.get(filter_type, {})

                ax.plot(
                    capacities,
                    hit_rates,
                    label=filter_type.capitalize(),
                    linewidth=2.5,
                    markersize=8,
                    color=style.get("color"),
                    marker=style.get("marker", "o"),
                )

            ax.set_xlabel("Capacity (elements)", fontsize=12, fontweight="bold")
            ax.set_ylabel(f"{cache_level} Hit Rate (%)", fontsize=12, fontweight="bold")
            ax.set_title(
                f"{cache_level} Cache Hit Rate",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xscale("log", base=2)
            ax.grid(True, which="both", ls="--", alpha=0.3)
            ax.legend(fontsize=11, loc="best", framealpha=0.9)
            ax.set_ylim(0, 105)

        plt.suptitle(
            f"Cache Hit Rate Comparison - {operation.capitalize()} Operation",
            fontsize=16,
            fontweight="bold",
            y=1.00,
        )
        plt.tight_layout()

        output_file = output_dir / f"cache_overview_{operation}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        typer.secho(
            f"Overview {operation} plot saved to {output_file}",
            fg=typer.colors.GREEN,
        )
        plt.close()


if __name__ == "__main__":
    app()
