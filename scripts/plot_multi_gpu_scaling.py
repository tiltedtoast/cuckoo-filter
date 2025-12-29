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
    df["gpus"] = df["gpus"].astype(int)
    df["throughput_mops"] = df["items_per_second"] / 1e6

    return df


def normalize_by_baseline(df: pd.DataFrame, baseline_gpus: int = 2) -> pd.DataFrame:
    """Normalize time and throughput values so that baseline_gpus has value 1.0."""
    df = df.copy()

    for fixture in df["fixture"].unique():
        for operation in df["operation"].unique():
            mask = (df["fixture"] == fixture) & (df["operation"] == operation)
            baseline_mask = mask & (df["gpus"] == baseline_gpus)

            if baseline_mask.sum() > 0:
                # Normalize time
                baseline_time = df.loc[baseline_mask, "real_time"].values[0]
                df.loc[mask, "normalized_time"] = (
                    df.loc[mask, "real_time"] / baseline_time
                )
                # Normalize throughput
                baseline_throughput = df.loc[baseline_mask, "throughput_mops"].values[0]
                df.loc[mask, "normalized_throughput"] = (
                    df.loc[mask, "throughput_mops"] / baseline_throughput
                )
            else:
                # No baseline found, use raw values
                df.loc[mask, "normalized_time"] = df.loc[mask, "real_time"]
                df.loc[mask, "normalized_throughput"] = df.loc[mask, "throughput_mops"]

    return df


def plot_scaling_on_axis(
    ax: plt.Axes,
    df: pd.DataFrame,
    scaling_mode: str,
    show_xlabel: bool = True,
    baseline_gpus: int = 2,
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
    operations = ["Query", "Insert", "Delete"]

    n_gpu_counts = len(gpu_counts)
    n_operations = len(operations)
    bar_width = 0.25
    group_width = bar_width * n_operations + 0.15

    # Choose metric based on scaling mode
    # Weak scaling: normalized throughput (higher = better, ideal = n GPUs / baseline GPUs)
    # Strong scaling: normalized time (lower = better, ideal = baseline GPUs / n GPUs)
    use_throughput = scaling_mode == "weak"

    for gpu_idx, gpu_count in enumerate(gpu_counts):
        gpu_data = mode_df[mode_df["gpus"] == gpu_count]
        group_center = gpu_idx * group_width

        for op_idx, operation in enumerate(operations):
            op_data = gpu_data[gpu_data["operation"] == operation]

            if use_throughput:
                value = (
                    op_data["normalized_throughput"].max() if len(op_data) > 0 else 0
                )
            else:
                value = op_data["normalized_time"].max() if len(op_data) > 0 else 0

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
                value,
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

    if use_throughput:
        ax.set_ylabel("Normalized Throughput", fontsize=14, fontweight="bold")
    else:
        ax.set_ylabel("Normalized Time", fontsize=14, fontweight="bold")

    # Build title with capacity info
    if (
        "capacity_per_gpu" in mode_df.columns
        and not mode_df["capacity_per_gpu"].isna().all()
    ):
        if scaling_mode == "weak":
            cap = mode_df["capacity_per_gpu"].iloc[0]
            cap_exp = round(np.log2(cap)) if cap > 0 else 0
            title = rf"Weak Scaling (${2}^{{{cap_exp}}}$ slots/GPU)"
        else:
            total_cap = (
                mode_df["total_capacity"].iloc[0]
                if "total_capacity" in mode_df.columns
                else 0
            )
            total_exp = round(np.log2(total_cap)) if total_cap > 0 else 0
            title = rf"Strong Scaling (${2}^{{{total_exp}}}$ total slots)"
    else:
        title = f"{scaling_mode.capitalize()} Scaling"

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.3, axis="y")

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

    Creates separate figures for each scaling mode:
    - Weak scaling (total capacity grows with GPU count)
    - Strong scaling (fixed total capacity)
    """
    if not csv_file.exists():
        typer.secho(f"Error: File not found: {csv_file}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    df = load_and_parse_csv(csv_file)
    df = normalize_by_baseline(df, baseline_gpus=2)

    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))

    # Check which scaling modes are present
    scaling_modes = df["fixture"].dropna().unique().tolist()

    if len(scaling_modes) == 0:
        typer.secho("No valid scaling data found in CSV", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # Create separate figure for each scaling mode
    for mode in ["weak", "strong"]:
        if mode not in scaling_modes:
            continue

        fig, ax = plt.subplots(figsize=(12, 8))

        legend_elements = plot_scaling_on_axis(ax, df, mode, show_xlabel=True)

        if legend_elements:
            fig.legend(
                handles=legend_elements,
                fontsize=10,
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
                framealpha=0.9,
            )

        plt.tight_layout()

        output_path = output_dir / f"{mode}_scaling.pdf"
        pu.save_figure(fig, output_path, f"Saved plot to {output_path}")


if __name__ == "__main__":
    app()
