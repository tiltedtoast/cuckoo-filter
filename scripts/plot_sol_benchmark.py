#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "typer",
# ]
# ///

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import plot_utils as pu
import typer

app = typer.Typer(help="Plot Speed of Light (SOL) benchmark results")

METRICS = [
    ("sm_throughput", "Compute (SM)"),
    ("memory_throughput", "Memory"),
    ("l1_throughput", "L1 Cache"),
    ("l2_throughput", "L2 Cache"),
    ("dram_throughput", "DRAM"),
]


@app.command()
def main(
    csv_file: Path = typer.Argument(
        ...,
        help="Path to CSV file with SOL benchmark results",
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
    Generate Speed of Light (SOL) throughput plots from benchmark CSV results.

    Creates plots showing Compute, Memory, L1, L2, and DRAM throughputs as
    percentage of peak sustained performance.
    """
    df = pu.load_csv(csv_file)

    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))

    operation_markers = {
        "insert": "o",
        "query": "s",
        "delete": "^",
    }

    metric_styles = {
        "sm_throughput": {"color": "#E63946", "marker": "o", "linestyle": "-"},
        "memory_throughput": {"color": "#1D3557", "marker": "s", "linestyle": "-"},
        "l1_throughput": {"color": "#457B9D", "marker": "^", "linestyle": "--"},
        "l2_throughput": {"color": "#A8DADC", "marker": "v", "linestyle": "--"},
        "dram_throughput": {
            "color": "#2A9D8F",
            "marker": "D",
            "linestyle": ":",
        },
    }

    # 1. Per-Filter/Operation Breakdown (All metrics on one plot)
    for filter_type in df["filter"].unique():
        for operation in df["operation"].unique():
            subset = df[(df["filter"] == filter_type) & (df["operation"] == operation)]
            if subset.empty:
                continue

            subset = subset.sort_values("capacity")
            capacities = subset["capacity"].values

            fig, ax = plt.subplots(figsize=(12, 7))

            for metric_col, metric_name in METRICS:
                if metric_col not in subset.columns:
                    continue

                values = subset[metric_col].values
                style = metric_styles.get(metric_col, {})

                ax.plot(
                    capacities,
                    values,
                    label=metric_name,
                    linewidth=2.5,
                    markersize=8,
                    **style,  # ty:ignore[invalid-argument-type]
                )

            ax.set_xlabel("Filter Capacity (elements)", fontsize=14, fontweight="bold")
            ax.set_ylabel("Throughput (% of Peak)", fontsize=14, fontweight="bold")
            ax.set_title(
                f"SOL Throughput Analysis - {pu.get_filter_display_name(filter_type)} / {operation.capitalize()}",
                fontsize=16,
                fontweight="bold",
                pad=20,
            )
            ax.set_xscale("log", base=2)
            ax.set_ylim(0, 105)
            ax.grid(True, which="both", ls="--", alpha=0.3)
            ax.legend(fontsize=12, loc="best", framealpha=0.9)

            plt.tight_layout()

            output_file = output_dir / f"sol_{filter_type}_{operation}.pdf"
            plt.savefig(output_file, bbox_inches="tight")
            typer.secho(f"Saved {output_file}", fg=typer.colors.GREEN)
            plt.close()

    # 2. Per-Metric Comparison (Comparing filters for a specific metric)
    for metric_col, metric_name in METRICS:
        if metric_col not in df.columns:
            continue

        # Separate by operation to keep plots readable
        for operation in df["operation"].unique():
            op_subset = df[df["operation"] == operation]
            if op_subset.empty:
                continue

            fig, ax = plt.subplots(figsize=(12, 7))

            for filter_type in sorted(op_subset["filter"].unique()):
                filter_subset = op_subset[
                    op_subset["filter"] == filter_type
                ].sort_values("capacity")
                if filter_subset.empty:
                    continue

                style = pu.FILTER_STYLES.get(filter_type, {})

                ax.plot(
                    filter_subset["capacity"].values,
                    filter_subset[metric_col].values,
                    label=f"{pu.get_filter_display_name(filter_type)} ({metric_name.replace('_', ' ').title()})",
                    linewidth=2.5,
                    markersize=8,
                    **style,  # ty:ignore[invalid-argument-type]
                )

            ax.set_xlabel("Filter Capacity (elements)", fontsize=14, fontweight="bold")
            ax.set_ylabel(
                f"{metric_name} Throughput (% of Peak)", fontsize=14, fontweight="bold"
            )
            ax.set_title(
                f"{metric_name} Throughput Comparison - {operation.capitalize()}",
                fontsize=16,
                fontweight="bold",
                pad=20,
            )
            ax.set_xscale("log", base=2)
            ax.set_ylim(0, 105)
            ax.grid(True, which="both", ls="--", alpha=0.3)
            ax.legend(fontsize=12, loc="best", framealpha=0.9)

            plt.tight_layout()

            output_file = output_dir / f"sol_compare_{metric_col}_{operation}.pdf"
            plt.savefig(
                output_file,
                bbox_inches="tight",
                transparent=True,
                format="pdf",
                dpi=600,
            )
            typer.secho(f"Saved {output_file}", fg=typer.colors.GREEN)
            plt.close()

    # 3. Small Multiples: 2x2 grid with one subplot per filter
    # Each subplot shows all metrics, with operations as line styles
    for metric_col, metric_name in METRICS:
        if metric_col not in df.columns:
            continue

        filters = ["cuckoo", "bloom", "tcf", "gqf"]
        available_filters = [f for f in filters if f in df["filter"].unique()]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False, sharey=False)
        axes = axes.flatten()

        for idx, filter_type in enumerate(available_filters):
            ax = axes[idx]
            filter_df = df[df["filter"] == filter_type]

            for operation in sorted(filter_df["operation"].unique()):
                subset = filter_df[filter_df["operation"] == operation].sort_values(
                    "capacity"
                )
                if subset.empty:
                    continue

                marker = operation_markers.get(operation, "o")
                style = pu.FILTER_STYLES.get(filter_type, {})

                ax.plot(
                    subset["capacity"].values,
                    subset[metric_col].values,
                    label=operation.capitalize(),
                    linewidth=2.5,
                    markersize=8,
                    color=style.get("color"),
                    marker=marker,
                    linestyle="-",
                )

            ax.set_title(
                pu.get_filter_display_name(filter_type),
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xscale("log", base=2)
            ax.set_ylim(0, 105)
            ax.grid(True, which="both", ls="--", alpha=0.3)
            ax.legend(fontsize=10, loc="best", framealpha=0)

        # Hide unused subplots if fewer than 4 filters
        for idx in range(len(available_filters), 4):
            axes[idx].set_visible(False)

        # Common axis labels
        fig.supxlabel("Filter Capacity (elements)", fontsize=14, fontweight="bold")
        fig.supylabel(
            f"{metric_name} Throughput (% of Peak)", fontsize=14, fontweight="bold"
        )
        fig.suptitle(
            f"{metric_name} Throughput by Filter",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )

        plt.tight_layout()

        output_file = output_dir / f"sol_grid_{metric_col}.pdf"
        plt.savefig(
            output_file,
            bbox_inches="tight",
            transparent=True,
            format="pdf",
            dpi=600,
        )
        typer.secho(f"Saved {output_file}", fg=typer.colors.GREEN)
        plt.close()


if __name__ == "__main__":
    app()
