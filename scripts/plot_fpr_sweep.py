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

import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

app = typer.Typer(help="Plot FPR sweep benchmark results")

FILTER_TYPES = {
    "GPUCF": "GPU Cuckoo",
    "CPUCF": "CPU Cuckoo",
    "Bloom": "Blocked Bloom",
    "TCF": "TCF",
    "GQF": "GQF",
    "PartitionedCF": "Partitioned CF",
}

FILTER_COLORS = {
    "GPU Cuckoo": "#2E86AB",
    "CPU Cuckoo": "#00B4D8",
    "Blocked Bloom": "#A23B72",
    "TCF": "#C73E1D",
    "GQF": "#F18F01",
    "Partitioned CF": "#6A994E",
}


def parse_benchmark_name(name: str) -> dict:
    """Extract filter type, fingerprint bits, and load factor from benchmark name."""
    # Examples:
    # GPUCF_FPR_Sweep<8, 50>
    # Bloom_FPR_Sweep<4, 50>
    # TCF_FPR_Sweep<uint16_t, 50>
    # GQF_FPR_Sweep<50>

    result = {"filter": None, "fingerprint_bits": None, "load_factor": None}

    for prefix, filter_name in FILTER_TYPES.items():
        if name.startswith(prefix):
            result["filter"] = filter_name
            break

    type_to_bits = {
        "uint8_t": 8,
        "uint16_t": 16,
        "uint32_t": 32,
        "uint64_t": 64,
    }

    # Extract template parameters
    if "<" in name and ">" in name:
        params = name[name.index("<") + 1 : name.index(">")].split(",")
        params = [p.strip() for p in params]

        if len(params) == 2:
            # Could be int or type name
            first_param = params[0]
            if first_param in type_to_bits:
                result["fingerprint_bits"] = type_to_bits[first_param]
            else:
                result["fingerprint_bits"] = int(first_param)
            result["load_factor"] = int(params[1]) / 100.0
        elif len(params) == 1:
            result["load_factor"] = int(params[0]) / 100.0

    return result


def create_fastest_filter_heatmap(df: pd.DataFrame, output_path: Path):
    """Create a heatmap showing the fastest filter for each (FPR, space) combination."""

    # Define bins for FPR and bits_per_item
    fpr_bins = np.array([2**i for i in range(-16, 0)])  # 2^-16 to 2^-1
    space_bins = np.array(
        [4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 50, 64, 100]
    )  # bits per item

    # Create grid
    n_fpr = len(fpr_bins) - 1
    n_space = len(space_bins) - 1

    # Map filter names to indices
    filter_names = list(FILTER_COLORS.keys())
    filter_to_idx = {name: i for i, name in enumerate(filter_names)}

    # Initialize grids
    fastest_filter = np.full((n_fpr, n_space), -1, dtype=int)
    best_throughput = np.zeros((n_fpr, n_space))

    # Assign each data point to a bin and track fastest
    for _, row in df.iterrows():
        if row["filter"] is None:
            continue

        fpr = row.get("fpr_percentage", 0) / 100  # Convert to fraction
        bits = row.get("bits_per_item", 0)
        throughput = row.get("throughput_mops", 0)

        if fpr <= 0 or bits <= 0:
            continue

        # Find bin indices
        fpr_idx = np.searchsorted(fpr_bins, fpr) - 1
        space_idx = np.searchsorted(space_bins, bits) - 1

        if 0 <= fpr_idx < n_fpr and 0 <= space_idx < n_space:
            if throughput > best_throughput[fpr_idx, space_idx]:
                best_throughput[fpr_idx, space_idx] = throughput
                fastest_filter[fpr_idx, space_idx] = filter_to_idx[row["filter"]]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Fastest filter heatmap
    ax1 = axes[0, 0]

    # Create colormap
    colors = ["white"] + [FILTER_COLORS[name] for name in filter_names]
    cmap = ListedColormap(colors)

    # Plot with -1 for empty cells mapped to white
    ax1.imshow(
        fastest_filter + 1, cmap=cmap, aspect="auto", vmin=0, vmax=len(filter_names)
    )

    # Set axis labels
    ax1.set_xticks(range(n_space))
    ax1.set_xticklabels([f"{int(space_bins[i])}" for i in range(n_space)])
    ax1.set_yticks(range(n_fpr))
    ax1.set_yticklabels([f"$2^{{{int(np.log2(fpr_bins[i]))}}}$" for i in range(n_fpr)])

    ax1.set_xlabel("Bits per item")
    ax1.set_ylabel("FPR")
    ax1.set_title("Fastest Filter")

    # Legend
    legend_elements = [
        Patch(facecolor=FILTER_COLORS[name], label=name) for name in filter_names
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=8)

    # Plot 2: Throughput heatmap
    ax2 = axes[0, 1]

    throughput_masked = np.ma.masked_where(best_throughput == 0, best_throughput)
    im2 = ax2.imshow(
        throughput_masked,
        cmap="viridis",
        aspect="auto",
        norm=plt.matplotlib.colors.LogNorm(vmin=1, vmax=throughput_masked.max()),
    )

    ax2.set_xticks(range(n_space))
    ax2.set_xticklabels([f"{int(space_bins[i])}" for i in range(n_space)])
    ax2.set_yticks(range(n_fpr))
    ax2.set_yticklabels([f"$2^{{{int(np.log2(fpr_bins[i]))}}}$" for i in range(n_fpr)])

    ax2.set_xlabel("Bits per item")
    ax2.set_ylabel("False positive rate (f)")
    ax2.set_title("Query Throughput (Mops/s)")

    plt.colorbar(im2, ax=ax2, label="Mops/s")

    # Plot 3: FPR vs bits_per_item scatter
    ax3 = axes[1, 0]

    for filter_name in filter_names:
        filter_df = df[df["filter"] == filter_name]
        if len(filter_df) > 0:
            fpr = filter_df["fpr_percentage"] / 100.0
            bits = filter_df["bits_per_item"]
            ax3.scatter(
                bits,
                fpr,
                c=FILTER_COLORS[filter_name],
                label=filter_name,
                alpha=0.7,
                s=50,
            )

    ax3.set_xlabel("Bits per item")
    ax3.set_ylabel("FPR")
    ax3.set_yscale("log", base=2)
    ax3.invert_yaxis()
    ax3.set_title("FPR vs Space Trade-off")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Throughput vs bits_per_item scatter
    ax4 = axes[1, 1]

    for filter_name in filter_names:
        filter_df = df[df["filter"] == filter_name]
        if len(filter_df) > 0:
            bits = filter_df["bits_per_item"]
            throughput = filter_df["throughput_mops"]
            ax4.scatter(
                bits,
                throughput,
                c=FILTER_COLORS[filter_name],
                label=filter_name,
                alpha=0.7,
                s=50,
            )

    ax4.set_xlabel("Bits per item")
    ax4.set_ylabel("Throughput (Mops/s)")
    ax4.set_title("Throughput vs Space Trade-off")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    typer.secho(f"Saved plot to {output_path}", fg=typer.colors.GREEN)


@app.command()
def main(
    csv_file: Path = typer.Argument(
        "-",
        help="Path to benchmark CSV file, or '-' to read from stdin (default: stdin)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots (default: build/)",
    ),
):
    """
    Plot FPR sweep benchmark results to compare filters at similar FPR/space tradeoffs.

    This generates a heatmap showing the fastest filter at each (FPR, space overhead)
    combination, similar to Figure 1 from filter comparison papers.

    Examples:
        cat results.csv | plot_fpr_sweep.py
        plot_fpr_sweep.py < results.csv
        plot_fpr_sweep.py results.csv
        plot_fpr_sweep.py results.csv -o custom/path.png
    """
    try:
        if str(csv_file) == "-":
            df = pd.read_csv(sys.stdin)
        elif not csv_file.exists():
            typer.secho(
                f"Error: File not found: {csv_file}", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(1)
        else:
            df = pd.read_csv(csv_file)
    except Exception as e:
        typer.secho(f"Error parsing CSV: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    df = df[df["name"].str.contains("_median", na=False)]

    # Parse benchmark names
    parsed = df["name"].apply(parse_benchmark_name)
    df["filter"] = parsed.apply(lambda x: x["filter"])
    df["fingerprint_bits"] = parsed.apply(lambda x: x["fingerprint_bits"])
    df["load_factor"] = parsed.apply(lambda x: x["load_factor"])

    df["throughput_mops"] = df["items_per_second"] / 1e6

    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / "build"

    output_dir.mkdir(parents=True, exist_ok=True)

    output = output_dir / "fpr_sweep.png"

    create_fastest_filter_heatmap(df, output)


if __name__ == "__main__":
    app()
