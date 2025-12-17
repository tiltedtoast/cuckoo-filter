"""Shared plotting utilities for benchmark visualization scripts.

This module provides common constants, styles, and helper functions used across
multiple plotting scripts to reduce code duplication and ensure visual consistency.
"""

import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

FILTER_STYLES = {
    "cuckoo": {"color": "#2E86AB", "marker": "o"},
    "bloom": {"color": "#A23B72", "marker": "s"},
    "tcf": {"color": "#C73E1D", "marker": "^"},
    "gqf": {"color": "#F18F01", "marker": "D"},
    "partitioned": {"color": "#6A994E", "marker": "D"},
    "cpu_cuckoo": {"color": "#00B4D8", "marker": "o"},
}

FILTER_COLORS = {
    "GPU Cuckoo": "#2E86AB",
    "Cuckoo": "#2E86AB",
    "Cuckoo Filter": "#2E86AB",
    "CPU Cuckoo": "#00B4D8",
    "Blocked Bloom": "#A23B72",
    "TCF": "#C73E1D",
    "GQF": "#F18F01",
    "Partitioned Cuckoo": "#6A994E",
}

FILTER_DISPLAY_NAMES = {
    # Standard names (lowercase of C++ fixture names)
    "gpucuckoo": "GPU Cuckoo",
    "blockedbloom": "Blocked Bloom",
    "tcf": "TCF",
    "gqf": "GQF",
    "partitionedcuckoo": "Partitioned Cuckoo",
    "cpucuckoo": "CPU Cuckoo",
    # Legacy compatibility (for existing CSVs)
    "cuckoo": "GPU Cuckoo",
    "bloom": "Blocked Bloom",
    "bbf": "Blocked Bloom",
    "cf": "GPU Cuckoo",
    "cpucf": "CPU Cuckoo",
    "pcf": "Partitioned Cuckoo",
    "cpu": "CPU Cuckoo",
    "partitioned": "Partitioned Cuckoo",
    # FPR sweep variants
    "gpucf": "GPU Cuckoo",
    "Bloom": "Blocked Bloom",
    "TCF": "TCF",
    "GQF": "GQF",
    "GPUCF": "GPU Cuckoo",
}

OPERATION_COLORS = {
    "Insert": "#2E86AB",
    "Query": "#A23B72",
    "Delete": "#F18F01",
}


DEFAULT_FONT_SIZE = 12
AXIS_LABEL_FONT_SIZE = 14
TITLE_FONT_SIZE = 16
LEGEND_FONT_SIZE = 10
LINE_WIDTH = 2.5
MARKER_SIZE = 8
GRID_ALPHA = 0.3


def get_filter_display_name(filter_type: str) -> str:
    """Get human-readable display name for a filter type.

    Args:
        filter_type: Internal filter identifier (e.g., 'cuckoo', 'bloom')

    Returns:
        Display name (e.g., 'Cuckoo', 'Blocked Bloom')
    """
    return FILTER_DISPLAY_NAMES.get(filter_type, filter_type.capitalize())


def format_power_of_two(n: int) -> str:
    """Format a number as a LaTeX power of 2 for use in plot titles.

    Args:
        n: Number to format (should be a power of 2)

    Returns:
        LaTeX string like '$\\left(n=2^{20}\\right)$'
    """
    if n <= 0:
        return ""
    power = int(math.log2(n))
    return rf"$\left(n=2^{{{power}}}\right)$"


def format_capacity_title(base_title: str, capacity: Optional[int]) -> str:
    """Format a title with capacity as power of 2.

    Args:
        base_title: Base title string
        capacity: Capacity value (power of 2)

    Returns:
        Title with capacity appended, e.g., 'Insert Throughput (n=2^20)'
    """
    if capacity is not None and capacity > 0:
        return f"{base_title} {format_power_of_two(capacity)}"
    return base_title


def load_csv(csv_path: Path) -> pd.DataFrame:
    """Load a CSV file with consistent error handling.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Loaded DataFrame

    Raises:
        typer.Exit: If CSV cannot be read
    """
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        typer.secho(f"Error reading CSV {csv_path}: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


def resolve_output_dir(output_dir: Optional[Path], script_path: Path) -> Path:
    """Resolve and create the output directory.

    Args:
        output_dir: User-specified output directory, or None for default
        script_path: Path to the calling script (typically __file__)

    Returns:
        Resolved output directory path (created if it doesn't exist)
    """
    if output_dir is None:
        script_dir = Path(script_path).parent
        output_dir = script_dir.parent / "build"

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def format_axis(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    title: Optional[str] = None,
    xscale: Optional[str] = "log",
    yscale: Optional[str] = None,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    grid: bool = True,
) -> None:
    """Apply consistent formatting to a matplotlib axis.

    Args:
        ax: Matplotlib axis to format
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Optional title for the axis
        xscale: X-axis scale ('log', 'linear', or None to skip)
        yscale: Y-axis scale ('log', 'linear', or None to skip)
        xlim: Optional (min, max) tuple for x-axis limits
        ylim: Optional (min, max) tuple for y-axis limits
        grid: Whether to show grid lines
    """
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONT_SIZE, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE, fontweight="bold")

    if title:
        ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight="bold")

    if xscale == "log":
        ax.set_xscale("log", base=2)
    elif xscale:
        ax.set_xscale(xscale)

    if yscale == "log":
        ax.set_yscale("log")
    elif yscale:
        ax.set_yscale(yscale)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    if grid:
        ax.grid(True, which="both", ls="--", alpha=GRID_ALPHA)


def save_figure(
    fig_or_path,
    output_path: Path,
    message: Optional[str] = None,
    close: bool = True,
) -> None:
    """Save a figure with consistent options and print success message.

    Args:
        fig_or_path: Figure object or None to use plt.savefig
        output_path: Path to save the figure
        message: Optional custom success message (default: 'Saved {path}')
        close: Whether to close the figure after saving
    """
    save_kwargs = {
        "bbox_inches": "tight",
        "transparent": True,
        "format": "pdf",
        "dpi": 600,
    }

    if fig_or_path is None:
        plt.savefig(output_path, **save_kwargs)
    else:
        fig_or_path.savefig(output_path, **save_kwargs)

    if message is None:
        message = f"Saved {output_path}"

    typer.secho(message, fg=typer.colors.GREEN)

    if close:
        if fig_or_path is None:
            plt.close()
        else:
            plt.close(fig_or_path)


def get_filter_style(filter_type: str, positive_negative: Optional[str] = None) -> dict:
    """Get the style dictionary for a filter type.

    Args:
        filter_type: Filter identifier (e.g., 'cuckoo', 'bloom')
        positive_negative: Optional 'Positive' or 'Negative' for query styling

    Returns:
        Dictionary with color, marker, and optionally linestyle
    """
    base_style = FILTER_STYLES.get(
        filter_type.lower(), {"color": "#333333", "marker": "o"}
    )

    style = dict(base_style)

    if positive_negative == "Positive":
        style["linestyle"] = "-"
    elif positive_negative == "Negative":
        style["linestyle"] = "--"

    return style


def setup_figure(
    figsize: tuple[int, int] = (12, 8),
    title: Optional[str] = None,
    nrows: int = 1,
    ncols: int = 1,
    sharex: bool = False,
    sharey: bool = False,
) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
    """Create a figure with consistent styling.

    Args:
        figsize: Figure size tuple
        title: Optional figure super title
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        sharex: Share X axis
        sharey: Share Y axis

    Returns:
        Tuple of (Figure, Axes/Array of Axes)
    """
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey
    )
    if title:
        fig.suptitle(title, fontsize=TITLE_FONT_SIZE, fontweight="bold")
    return fig, axes


def create_legend(ax: plt.Axes, **kwargs):
    """Create a legend with consistent default styling.

    Args:
        ax: Matplotlib axis to add legend to
        **kwargs: Override default legend parameters

    Returns:
        The created Legend object

    Example:
        create_legend(ax, loc="upper right", ncol=2)
    """
    defaults = {
        "fontsize": LEGEND_FONT_SIZE,
        "loc": "best",
        "framealpha": 0,
    }
    defaults.update(kwargs)
    return ax.legend(**defaults)


def normalize_benchmark_name(name: str) -> str:
    """Convert benchmark name to standardized format.

    Handles both new format (GPUCuckoo_5/Insert/...) and old format (CF_5/Insert/...).

    Args:
        name: Benchmark name like "GPUCuckoo_5/Insert/268435456/min_time:0.500/..."

    Returns:
        Standardized name like "gpucuckoo" or "blockedbloom"

    Examples:
        GPUCuckoo_5/Insert/... → "gpucuckoo"
        BlockedBloom_10/Query/... → "blockedbloom"
        CF_5/Insert/... → "cf" (legacy)
    """
    # Extract the filter prefix before the first underscore
    parts = name.split("/")
    if parts:
        # Split on underscore to get filter name: "GPUCuckoo_5" → "GPUCuckoo"
        base = parts[0].split("_")[0]
        return base.lower()

    return name.lower()
