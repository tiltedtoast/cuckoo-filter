#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "typer",
# ]
# ///
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer

app = typer.Typer(help="Plot load factor benchmark results")


def extract_num_elements(name: str) -> Optional[int]:
    """Extract number of elements from benchmark name like CF_5/Insert/4194304/..."""
    # Match pattern like /Insert/4194304/ or /Query/4194304/
    match = re.search(r"/(?:Insert|Query|QueryNegative|Delete)/(\d+)/", name)
    if match:
        return int(match.group(1))
    return None


def extract_load_factor(name: str) -> Optional[float]:
    """Extract load factor from benchmark name like CF_5/Insert, BBF_95/Query, or CF_99_5/Insert"""
    # Match patterns like _95/ or _99_5/ (where underscore represents decimal point)
    match = re.search(r"_([\d_]+)/", name)
    if match:
        # Replace underscore with decimal point (e.g., "99_5" -> "99.5")
        value_str = match.group(1).replace("_", ".")
        return float(value_str) / 100.0
    return None


def extract_operation_type(name: str) -> Optional[str]:
    """Extract operation type from benchmark name"""
    if "/Insert/" in name:
        return "Insert"
    elif "/QueryNegative/" in name:
        return "Query"
    elif "/Query/" in name:
        return "Query"
    elif "/Delete/" in name:
        return "Delete"
    return None


def extract_lookup_type(name: str) -> Optional[str]:
    """Extract lookup type (Positive or Negative) for Query operations"""
    if "/QueryNegative/" in name:
        return "Negative"
    elif "/Query/" in name:
        return "Positive"
    return None


def load_csv_data(csv_path: Path) -> dict:
    """Load and parse benchmark data from a CSV file.

    Returns a tuple of (benchmark_data, num_elements_per_operation)
    """
    df = pu.load_csv(csv_path)

    # Filter for median records only
    df = df[df["name"].str.endswith("_median")]

    # Dictionary structure: operation -> filter_type -> {load_factor: throughput}
    benchmark_data = defaultdict(lambda: defaultdict(dict))
    # Track number of elements per operation
    num_elements_per_operation = {}

    for _, row in df.iterrows():
        name = row["name"]

        # Extract filter type using standardized approach
        filter_key = pu.normalize_benchmark_name(name)
        filter_type = pu.get_filter_display_name(filter_key)

        if not filter_type:
            continue

        load_factor = extract_load_factor(name)
        operation_type = extract_operation_type(name)
        lookup_type = extract_lookup_type(name)
        num_elements = extract_num_elements(name)

        if filter_type is None or load_factor is None or operation_type is None:
            continue

        # Store the number of elements for this operation (assumes all rows have same count)
        if (
            operation_type
            and num_elements
            and operation_type not in num_elements_per_operation
        ):
            num_elements_per_operation[operation_type] = num_elements

        # For Query operations, append lookup type to filter name
        if operation_type == "Query" and lookup_type:
            filter_key = f"{filter_type} ({lookup_type})"
        else:
            filter_key = filter_type

        items_per_second = row.get("items_per_second")
        if pd.notna(items_per_second):
            throughput_bops = items_per_second / 1_000_000_000
            benchmark_data[operation_type][filter_key][load_factor] = throughput_bops

    return benchmark_data, num_elements_per_operation  # ty:ignore[invalid-return-type]


def get_filter_styles() -> dict:
    """Define colors and markers for each filter type, with positive/negative variants."""
    # Use the standardized filter styles from plot_utils as base
    base_styles = {
        "GPU Cuckoo": pu.FILTER_STYLES.get(
            "gpucuckoo", {"color": "#2E86AB", "marker": "o"}
        ),
        "CPU Cuckoo": pu.FILTER_STYLES.get(
            "cpucuckoo", {"color": "#00B4D8", "marker": "o"}
        ),
        "Blocked Bloom": pu.FILTER_STYLES.get(
            "blockedbloom", {"color": "#A23B72", "marker": "s"}
        ),
        "TCF": pu.FILTER_STYLES.get("tcf", {"color": "#C73E1D", "marker": "v"}),
        "GQF": pu.FILTER_STYLES.get("gqf", {"color": "#F18F01", "marker": "^"}),
        "Partitioned Cuckoo": pu.FILTER_STYLES.get(
            "partitionedcuckoo", {"color": "#6A994E", "marker": "D"}
        ),
    }

    # Generate styles for both positive and negative variants
    filter_styles = {}
    for filter_name, base_style in base_styles.items():
        # Base style (for non-query operations)
        filter_styles[filter_name] = {
            "color": base_style["color"],
            "marker": base_style["marker"],
            "linestyle": "-",
        }
        # Positive lookups: solid line
        filter_styles[f"{filter_name} (Positive)"] = {
            "color": base_style["color"],
            "marker": base_style["marker"],
            "linestyle": "-",
        }
        # Negative lookups: dashed line
        filter_styles[f"{filter_name} (Negative)"] = {
            "color": base_style["color"],
            "marker": base_style["marker"],
            "linestyle": "--",
        }

    return filter_styles


def plot_operation_on_axis(
    ax: plt.Axes,
    operation_type: str,
    benchmark_data: dict,
    num_elements_per_operation: dict,
    filter_styles: dict,
    show_ylabel: bool = True,
    title_suffix: Optional[str] = None,
) -> tuple[list, list]:
    """Plot a single operation's data on the given axis.

    Returns:
        A tuple of (handles, labels) for use in creating a combined legend.
    """
    handles = []
    labels = []

    for filter_type in sorted(benchmark_data[operation_type].keys()):
        load_factors = sorted(benchmark_data[operation_type][filter_type].keys())
        throughputs = [
            benchmark_data[operation_type][filter_type][lf] for lf in load_factors
        ]

        style = filter_styles.get(filter_type, {"marker": "o", "linestyle": "-"})
        (line,) = ax.plot(
            load_factors,
            throughputs,
            label=filter_type,
            linewidth=2.5,
            markersize=8,
            color=style.get("color"),
            marker=style.get("marker", "o"),
            linestyle=style.get("linestyle", "-"),
        )
        handles.append(line)
        labels.append(filter_type)

    ax.set_xlabel("Load Factor", fontsize=14, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("Throughput [B ops/s]", fontsize=14, fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    # Build title with element count and optional suffix
    title = f"{operation_type} Performance"
    if operation_type in num_elements_per_operation:
        n = num_elements_per_operation[operation_type]
        # Calculate power of 2
        power = int(math.log2(n))
        title += f" $\\left(n=2^{{{power}}}\\right)$"
    if title_suffix:
        title += f" ({title_suffix})"

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
    )

    ax.set_yscale("log")

    return handles, labels


@app.command()
def main(
    csv_top_left: Path = typer.Argument(
        ...,
        help="Path to CSV file for top-left plot",
    ),
    csv_top_right: Path = typer.Argument(
        ...,
        help="Path to CSV file for top-right plot",
    ),
    csv_bottom_left: Path = typer.Argument(
        ...,
        help="Path to CSV file for bottom-left plot",
    ),
    csv_bottom_right: Path = typer.Argument(
        ...,
        help="Path to CSV file for bottom-right plot",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots (default: build/)",
    ),
    label_top_left: Optional[str] = typer.Option(
        None,
        "--label-tl",
        help="Label to append to top-left plot title",
    ),
    label_top_right: Optional[str] = typer.Option(
        None,
        "--label-tr",
        help="Label to append to top-right plot title",
    ),
    label_bottom_left: Optional[str] = typer.Option(
        None,
        "--label-bl",
        help="Label to append to bottom-left plot title",
    ),
    label_bottom_right: Optional[str] = typer.Option(
        None,
        "--label-br",
        help="Label to append to bottom-right plot title",
    ),
):
    """
    Generate throughput vs load factor comparison plots from four benchmark CSV files.

    Creates a 2x2 grid of plots for each operation (insert, query, delete) with data
    from each CSV file in the corresponding position.

    Examples:
        plot_load_factor.py tl.csv tr.csv bl.csv br.csv
        plot_load_factor.py tl.csv tr.csv bl.csv br.csv -o custom/dir
    """
    # Load data from all four CSV files
    csv_files = [
        (csv_top_left, "top-left"),
        (csv_top_right, "top-right"),
        (csv_bottom_left, "bottom-left"),
        (csv_bottom_right, "bottom-right"),
    ]
    labels = [label_top_left, label_top_right, label_bottom_left, label_bottom_right]

    benchmark_data_list = []
    num_elements_list = []

    for csv_file, position in csv_files:
        data, num_elements = load_csv_data(csv_file)
        if not data:
            typer.secho(
                f"No throughput data found in {csv_file} ({position})",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)
        benchmark_data_list.append(data)
        num_elements_list.append(num_elements)

    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))

    filter_styles = get_filter_styles()

    # Get all operations from all datasets
    all_operations = set[str]()
    for data in benchmark_data_list:
        all_operations.update(data.keys())

    # Grid positions: (row, col)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for operation in sorted(all_operations):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)

        # Collect handles and labels for combined legend
        all_handles = []
        all_labels = []

        for idx, (data, num_elements, label, (row, col)) in enumerate(
            zip(benchmark_data_list, num_elements_list, labels, positions)
        ):
            ax = axes[row, col]
            has_data = operation in data and data[operation]

            # Show ylabel only on left column
            show_ylabel = col == 0

            if has_data:
                handles, plot_labels = plot_operation_on_axis(
                    ax,
                    operation,
                    data,
                    num_elements,
                    filter_styles,
                    show_ylabel=show_ylabel,
                    title_suffix=label,
                )
                # Only add handles/labels that aren't already in the combined list
                for handle, lbl in zip(handles, plot_labels):
                    if lbl not in all_labels:
                        all_handles.append(handle)
                        all_labels.append(lbl)
            else:
                ax.set_title(f"{operation} (No data)", fontsize=16, fontweight="bold")
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

        plt.tight_layout()

        # Create combined legend outside on the right
        if all_handles:
            fig.legend(
                all_handles,
                all_labels,
                fontsize=10,
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
                framealpha=0,
            )

        output_file = (
            output_dir / f"load_factor_{operation.lower().replace(' ', '_')}.pdf"
        )
        pu.save_figure(
            fig,
            output_file,
            f"{operation} throughput comparison plot saved to {output_file}",
            close=True,
        )


if __name__ == "__main__":
    app()
