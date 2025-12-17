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

app = typer.Typer(help="Plot FPR sweep benchmark results")


FPR_TARGETS = [
    (0.10, r"$\leq 10\%$"),
    (0.01, r"$\leq 1\%$"),
    (0.001, r"$\leq 0.1\%$"),
    (0.0001, r"$\leq 0.01\%$"),
]


def parse_benchmark_name(name: str) -> dict:
    """Extract filter type, fingerprint bits, load factor, and operation from benchmark name."""
    result = {
        "filter": None,
        "fingerprint_bits": None,
        "load_factor": None,
        "operation": "negative_query",  # FPR_Sweep measures negative lookups
    }

    if "_Insert_Sweep" in name:
        result["operation"] = "insert"
    elif "_PositiveQuery_Sweep" in name:
        result["operation"] = "positive_query"
    elif "_Delete_Sweep" in name:
        result["operation"] = "delete"

    # Map benchmark name prefix to standardized filter name
    # GPUCF_FPR_Sweep → "gpucf" → "GPU Cuckoo"
    for prefix in ["GPUCF", "Bloom", "TCF", "GQF"]:
        if name.startswith(prefix):
            result["filter"] = pu.get_filter_display_name(prefix.lower())
            break

    type_to_bits = {
        "uint8_t": 8,
        "uint16_t": 16,
        "uint32_t": 32,
        "uint64_t": 64,
    }

    if "<" in name and ">" in name:
        params = name[name.index("<") + 1 : name.index(">")].split(",")
        params = [p.strip() for p in params]

        if len(params) == 2:
            first_param = params[0]
            if first_param in type_to_bits:
                result["fingerprint_bits"] = type_to_bits[first_param]
            else:
                result["fingerprint_bits"] = int(first_param)
            result["load_factor"] = int(params[1]) / 100.0
        elif len(params) == 1:
            result["load_factor"] = int(params[0]) / 100.0

    return result


def load_and_parse_csv(csv_path: Path) -> pd.DataFrame:
    """Load and parse benchmark data from a CSV file."""
    df = pu.load_csv(csv_path)

    df = df[df["name"].str.contains("_median", na=False)]

    parsed = df["name"].apply(parse_benchmark_name)
    df["filter"] = parsed.apply(lambda x: x["filter"])
    df["fingerprint_bits"] = parsed.apply(lambda x: x["fingerprint_bits"])
    df["load_factor"] = parsed.apply(lambda x: x["load_factor"])
    df["operation"] = parsed.apply(lambda x: x["operation"])
    df["throughput_mops"] = df["items_per_second"] / 1e6

    return df


def plot_fpr_comparison_on_axis(
    ax: plt.Axes,
    df: pd.DataFrame,
    hit_rate: float,
    exponent: Optional[int] = None,
    show_xlabel: bool = True,
) -> list:
    """Plot FPR comparison on a single axis. Returns legend elements."""

    # Separate by operation type
    neg_query_df = df[df["operation"] == "negative_query"].copy()
    pos_query_df = df[df["operation"] == "positive_query"].copy()
    insert_df = df[df["operation"] == "insert"].copy()

    filter_names = [
        f for f in pu.FILTER_COLORS.keys() if f in neg_query_df["filter"].values
    ]

    # Create config key to match data across operations
    def make_config_key(row):
        return f"{row['filter']}_{row.get('fingerprint_bits', '')}_{row.get('load_factor', '')}"

    neg_query_df["config_key"] = neg_query_df.apply(make_config_key, axis=1)

    # Map positive query throughput
    if len(pos_query_df) > 0:
        pos_query_df["config_key"] = pos_query_df.apply(make_config_key, axis=1)
        pos_lookup = pos_query_df.set_index("config_key")["throughput_mops"].to_dict()
        neg_query_df["positive_query_throughput"] = neg_query_df["config_key"].map(
            pos_lookup
        )
    else:
        neg_query_df["positive_query_throughput"] = np.nan

    # Calculate weighted average query throughput based on hit rate
    neg_query_df["avg_query_throughput"] = neg_query_df.apply(
        lambda r: (
            hit_rate * r["positive_query_throughput"]
            + (1 - hit_rate) * r["throughput_mops"]
        )
        if pd.notna(r["positive_query_throughput"])
        else r["throughput_mops"],
        axis=1,
    )

    # Map insert throughput
    if len(insert_df) > 0:
        insert_df["config_key"] = insert_df.apply(make_config_key, axis=1)
        insert_lookup = insert_df.set_index("config_key")["throughput_mops"].to_dict()
        neg_query_df["insert_throughput"] = neg_query_df["config_key"].map(
            insert_lookup
        )
    else:
        neg_query_df["insert_throughput"] = np.nan

    # Map delete throughput
    delete_df = df[df["operation"] == "delete"].copy()
    if len(delete_df) > 0:
        delete_df["config_key"] = delete_df.apply(make_config_key, axis=1)
        delete_lookup = delete_df.set_index("config_key")["throughput_mops"].to_dict()
        neg_query_df["delete_throughput"] = neg_query_df["config_key"].map(
            delete_lookup
        )
    else:
        neg_query_df["delete_throughput"] = np.nan

    has_insert_data = neg_query_df["insert_throughput"].notna().any()
    has_positive_data = neg_query_df["positive_query_throughput"].notna().any()
    has_delete_data = neg_query_df["delete_throughput"].notna().any()

    n_fpr_targets = len(FPR_TARGETS)
    n_filters = len(filter_names)

    # Calculate bar widths based on available data (query + insert + delete)
    n_bars = 1 + (1 if has_insert_data else 0) + (1 if has_delete_data else 0)
    bar_width = 0.25
    group_width = bar_width * n_bars + 0.1
    target_width = n_filters * group_width + 0.5

    for target_idx, (fpr_target, fpr_label) in enumerate(FPR_TARGETS):
        target_offset = target_idx * target_width

        for filter_idx, filter_name in enumerate(filter_names):
            filter_data = neg_query_df[neg_query_df["filter"] == filter_name]
            qualifying = filter_data[filter_data["fpr_percentage"] / 100 <= fpr_target]

            # Use weighted average query throughput if positive data available
            if has_positive_data:
                query_tp = (
                    qualifying["avg_query_throughput"].max()
                    if len(qualifying) > 0
                    else 0
                )
            else:
                query_tp = (
                    qualifying["throughput_mops"].max() if len(qualifying) > 0 else 0
                )

            insert_qualifying = qualifying[qualifying["insert_throughput"].notna()]
            insert_tp = (
                insert_qualifying["insert_throughput"].max()
                if len(insert_qualifying) > 0
                else 0
            )

            delete_qualifying = qualifying[qualifying["delete_throughput"].notna()]
            delete_tp = (
                delete_qualifying["delete_throughput"].max()
                if len(delete_qualifying) > 0
                else 0
            )

            x_base = target_offset + filter_idx * group_width
            bar_offset = 0

            # Query bar (solid)
            ax.bar(
                x_base,
                query_tp,
                bar_width,
                color=pu.FILTER_COLORS[filter_name],
                edgecolor="white",
                linewidth=0.5,
            )
            bar_offset += bar_width

            # Insert bar (diagonal hatch)
            if has_insert_data:
                ax.bar(
                    x_base + bar_offset,
                    insert_tp,
                    bar_width,
                    color=pu.FILTER_COLORS[filter_name],
                    edgecolor="white",
                    linewidth=0.5,
                    hatch="//",
                    alpha=0.7,
                )
                bar_offset += bar_width

            # Delete bar (horizontal hatch)
            if has_delete_data:
                ax.bar(
                    x_base + bar_offset,
                    delete_tp,
                    bar_width,
                    color=pu.FILTER_COLORS[filter_name],
                    edgecolor="white",
                    linewidth=0.5,
                    hatch="--",
                    alpha=0.7,
                )

    # Set x-axis labels at center of each FPR target group
    target_centers = [
        i * target_width + (n_filters * group_width) / 2 - group_width / 2
        for i in range(n_fpr_targets)
    ]
    ax.set_xticks(target_centers)
    ax.set_xticklabels([label for _, label in FPR_TARGETS], fontsize=12)

    if show_xlabel:
        ax.set_xlabel("Target FPR", fontsize=14, fontweight="bold")
    ax.set_ylabel("Throughput [M ops/s]", fontsize=14, fontweight="bold")

    hit_pct = int(hit_rate * 100)
    title = rf"Best Throughput by Target FPR ({hit_pct}% hit rate)"
    if exponent is not None:
        title += f" $\\left(n=2^{{{exponent}}}\\right)$"
    ax.set_title(title, fontsize=16, fontweight="bold")

    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.3)

    # Build legend elements
    legend_elements = [
        Patch(facecolor=pu.FILTER_COLORS[name], label=name) for name in filter_names
    ]
    legend_elements.append(Patch(facecolor="gray", label="Query"))
    if has_insert_data:
        legend_elements.append(
            Patch(facecolor="gray", hatch="//", alpha=0.7, label="Insert")
        )
    if has_delete_data:
        legend_elements.append(
            Patch(facecolor="gray", hatch="--", alpha=0.7, label="Delete")
        )

    return legend_elements, filter_names  # ty:ignore[invalid-return-type]


@app.command()
def main(
    csv_file_top: Path = typer.Argument(
        ...,
        help="Path to CSV file for top plot",
    ),
    csv_file_bottom: Path = typer.Argument(
        ...,
        help="Path to CSV file for bottom plot",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots (default: build/)",
    ),
    exponent_top: Optional[int] = typer.Option(
        None,
        "--exponent-top",
        "-et",
        help="Exponent for n (as power of 2) to display in top plot title",
    ),
    exponent_bottom: Optional[int] = typer.Option(
        None,
        "--exponent-bottom",
        "-eb",
        help="Exponent for n (as power of 2) to display in bottom plot title",
    ),
    hit_rate: float = typer.Option(
        50.0,
        "--hit-rate",
        "-h",
        help="Expected percentage of positive queries (0-100, default: 50)",
    ),
):
    """
    Generate FPR sweep throughput plots from two benchmark CSV files.

    Creates a single figure with two vertically stacked plots.

    Examples:
        plot_fpr_sweep.py small.csv large.csv
        plot_fpr_sweep.py small.csv large.csv --exponent-top 22 --exponent-bottom 26
    """
    # Load data from both CSV files
    csv_files = [
        (csv_file_top, exponent_top),
        (csv_file_bottom, exponent_bottom),
    ]

    data_list = []
    for csv_file, exponent in csv_files:
        if not csv_file.exists():
            typer.secho(
                f"Error: File not found: {csv_file}", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(1)
        df = load_and_parse_csv(csv_file)
        data_list.append((df, exponent))

    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))

    # Create figure with 2 vertically stacked subplots
    fig, axes = plt.subplots(2, 1, figsize=(16, 14))

    all_legend_elements = []
    seen_labels = set()

    for idx, (ax, (df, exponent)) in enumerate(zip(axes, data_list)):
        show_xlabel = idx == 1  # Only show x-label on bottom plot

        legend_elements, filter_names = plot_fpr_comparison_on_axis(
            ax,
            df,
            hit_rate / 100.0,
            exponent=exponent,
            show_xlabel=show_xlabel,
        )

        # Collect unique legend elements
        for elem in legend_elements:
            if elem.get_label() not in seen_labels:
                all_legend_elements.append(elem)
                seen_labels.add(elem.get_label())

    plt.tight_layout()

    # Create combined legend outside on the right
    if all_legend_elements:
        fig.legend(
            handles=all_legend_elements,
            fontsize=10,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            framealpha=0,
        )

    output_path = output_dir / "fpr_sweep_throughput.pdf"
    pu.save_figure(fig, output_path, f"Saved throughput comparison to {output_path}")


if __name__ == "__main__":
    app()
