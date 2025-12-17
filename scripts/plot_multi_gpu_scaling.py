#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "typer",
# ]
# ///

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_utils as pu
import typer
from matplotlib.patches import Patch

app = typer.Typer(help="Plot multi-GPU scaling benchmark results")


def parse_benchmark_name(name: str) -> dict:
    """Extract fixture type and operation from benchmark name."""
    result = {
        "operation": None,
        "fixture": None,
    }

    # Parse fixture type
    if "WeakFixture" in name:
        result["fixture"] = "weak"
    elif "StrongFixture" in name:
        result["fixture"] = "strong"
    elif "ScalingFixture" in name:
        # Legacy single fixture
        result["fixture"] = "weak"

    # Parse operation
    if "/Insert" in name:
        result["operation"] = "Insert"
    elif "/Query" in name:
        result["operation"] = "Query"
    elif "/Delete" in name:
        result["operation"] = "Delete"

    return result


def load_and_parse_csv(csv_path: Path) -> pd.DataFrame:
    """Load and parse benchmark data from a CSV file."""
    df = pu.load_csv(csv_path)

    # Filter to median results only
    df = df[df["name"].str.contains("_median", na=False)]

    parsed = df["name"].apply(parse_benchmark_name)
    df["operation"] = parsed.apply(lambda x: x["operation"])
    df["fixture"] = parsed.apply(lambda x: x["fixture"])
    df["throughput_mops"] = df["items_per_second"] / 1e6
    df["gpus"] = df["gpus"].astype(int)

    return df


def plot_scaling_on_axis(
    ax: plt.Axes,
    df: pd.DataFrame,
    scaling_mode: str,
    show_xlabel: bool = True,
) -> list:
    """Plot scaling results on a single axis. Returns legend elements."""
    # Filter to this scaling mode
    mode_df = df[df["fixture"] == scaling_mode]

    if len(mode_df) == 0:
        ax.text(
            0.5,
            0.5,
            f"No {scaling_mode} scaling data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return []

    gpu_counts = sorted(mode_df["gpus"].unique())
    operations = ["Insert", "Query", "Delete"]

    n_gpu_counts = len(gpu_counts)
    n_operations = len(operations)
    bar_width = 0.25
    group_width = bar_width * n_operations + 0.15

    for gpu_idx, gpu_count in enumerate(gpu_counts):
        gpu_data = mode_df[mode_df["gpus"] == gpu_count]
        group_center = gpu_idx * group_width

        for op_idx, operation in enumerate(operations):
            op_data = gpu_data[gpu_data["operation"] == operation]
            throughput = op_data["throughput_mops"].max() if len(op_data) > 0 else 0

            x_pos = group_center + (op_idx - 1) * bar_width

            # Apply hatching for Insert and Delete
            hatch = None
            alpha = 1.0
            if operation == "Insert":
                hatch = "//"
                alpha = 0.8
            elif operation == "Delete":
                hatch = "--"
                alpha = 0.8

            ax.bar(
                x_pos,
                throughput,
                bar_width,
                color=pu.OPERATION_COLORS.get(operation, "#999999"),
                edgecolor="white",
                linewidth=0.5,
                hatch=hatch,
                alpha=alpha,
            )

    # Set x-axis labels
    group_centers = [i * group_width for i in range(n_gpu_counts)]
    ax.set_xticks(group_centers)
    ax.set_xticklabels([str(g) for g in gpu_counts], fontsize=12)

    if show_xlabel:
        ax.set_xlabel("Number of GPUs", fontsize=14, fontweight="bold")
    ax.set_ylabel("Throughput [M ops/s]", fontsize=14, fontweight="bold")

    # Build title with capacity info
    if (
        "capacity_per_gpu" in mode_df.columns
        and not mode_df["capacity_per_gpu"].isna().all()
    ):
        if scaling_mode == "weak":
            cap = mode_df["capacity_per_gpu"].iloc[0]
            cap_exp = int(np.log2(cap)) if cap > 0 else 0
            title = rf"Weak Scaling (${2}^{{{cap_exp}}}$ slots/GPU)"
        else:
            total_cap = (
                mode_df["total_capacity"].iloc[0]
                if "total_capacity" in mode_df.columns
                else 0
            )
            total_exp = int(np.log2(total_cap)) if total_cap > 0 else 0
            title = rf"Strong Scaling (${2}^{{{total_exp}}}$ total slots)"
    else:
        title = f"{scaling_mode.capitalize()} Scaling"

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.3, axis="y")

    # Build legend elements
    legend_elements = [
        Patch(facecolor=pu.OPERATION_COLORS["Query"], label="Query"),
        Patch(
            facecolor=pu.OPERATION_COLORS["Insert"],
            hatch="//",
            alpha=0.8,
            label="Insert",
        ),
        Patch(
            facecolor=pu.OPERATION_COLORS["Delete"],
            hatch="--",
            alpha=0.8,
            label="Delete",
        ),
    ]

    return legend_elements


@app.command()
def main(
    csv_file: Path = typer.Argument(
        ...,
        help="Path to CSV file with benchmark results",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots (default: build/)",
    ),
):
    """
    Generate multi-GPU scaling throughput plots from benchmark CSV.

    Creates a figure with two vertically stacked plots:
    - Top: Weak scaling (total capacity grows with GPU count)
    - Bottom: Strong scaling (fixed total capacity)
    """
    if not csv_file.exists():
        typer.secho(f"Error: File not found: {csv_file}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    df = load_and_parse_csv(csv_file)

    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))

    # Check which scaling modes are present
    scaling_modes = df["fixture"].dropna().unique().tolist()
    n_modes = len(scaling_modes)

    if n_modes == 0:
        typer.secho("No valid scaling data found in CSV", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # Create figure with appropriate number of subplots
    if n_modes == 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        axes = [ax]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 14))

    all_legend_elements = []
    seen_labels = set()

    # Plot each scaling mode
    for idx, (ax, mode) in enumerate(zip(axes, ["weak", "strong"][:n_modes])):
        if mode not in scaling_modes:
            continue

        show_xlabel = idx == n_modes - 1
        legend_elements = plot_scaling_on_axis(ax, df, mode, show_xlabel=show_xlabel)

        for elem in legend_elements:
            if elem.get_label() not in seen_labels:
                all_legend_elements.append(elem)
                seen_labels.add(elem.get_label())

    plt.tight_layout()

    if all_legend_elements:
        fig.legend(
            handles=all_legend_elements,
            fontsize=10,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            framealpha=0.9,
        )

    output_path = output_dir / "multi_gpu_scaling.pdf"
    pu.save_figure(fig, output_path, f"Saved plot to {output_path}")


if __name__ == "__main__":
    app()
