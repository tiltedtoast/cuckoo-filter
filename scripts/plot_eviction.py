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
import typer

app = typer.Typer(help="Plot eviction benchmark results")


def extract_eviction_policy(name: str) -> Optional[str]:
    """Extract eviction policy from benchmark name like BFSFixture/Evictions/..."""
    if name.startswith("BFSFixture"):
        return "BFS"
    elif name.startswith("DFSFixture"):
        return "DFS"
    return None


def extract_load_factor(row: pd.Series) -> Optional[float]:
    """Extract load factor from benchmark counter."""
    lf = row.get("load_factor")
    if pd.notna(lf):
        return float(lf) / 100.0
    return None


@app.command()
def main(
    csv_file: Path = typer.Argument(
        "-",
        help="Path to CSV file, or '-' to read from stdin (default: stdin)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots (default: build/)",
    ),
):
    """
    Generate eviction count plots from benchmark CSV results.

    Creates plots showing evictions vs load factor for BFS and DFS policies.

    Examples:
        cat results.csv | plot_eviction.py
        plot_eviction.py < results.csv
        plot_eviction.py results.csv
        plot_eviction.py results.csv -o custom/dir
    """
    try:
        if str(csv_file) == "-":
            import sys

            df = pd.read_csv(sys.stdin)
        else:
            df = pd.read_csv(csv_file)
    except Exception as e:
        typer.secho(f"Error parsing CSV: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # Filter for median records only
    df = df[df["name"].str.endswith("_median")]

    # Dictionary: policy -> {load_factor: value}
    eviction_data = defaultdict(dict)
    total_evictions_data = defaultdict(dict)
    throughput_data = defaultdict(dict)

    for _, row in df.iterrows():
        name = row["name"]

        policy = extract_eviction_policy(name)
        load_factor = extract_load_factor(row)

        if policy is None or load_factor is None:
            continue

        evictions_per_insert = row.get("evictions_per_insert")
        evictions = row.get("evictions")
        items_per_second = row.get("items_per_second")

        if pd.notna(evictions_per_insert):
            eviction_data[policy][load_factor] = evictions_per_insert
        if pd.notna(evictions):
            total_evictions_data[policy][load_factor] = evictions
        if pd.notna(items_per_second):
            throughput_data[policy][load_factor] = items_per_second

    if not eviction_data:
        typer.secho("No eviction data found in CSV", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    if not throughput_data:
        typer.secho("No throughput data found in CSV", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # Determine output directory
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / "build"

    output_dir.mkdir(parents=True, exist_ok=True)

    policy_styles = {
        "BFS": {"color": "#2E86AB", "marker": "o", "linestyle": "-"},
        "DFS": {"color": "#A23B72", "marker": "s", "linestyle": "--"},
    }

    # Plot 1: Evictions per insert vs load factor
    fig, ax = plt.subplots(figsize=(12, 8))

    for policy in sorted(eviction_data.keys()):
        load_factors = sorted(eviction_data[policy].keys())
        evictions = [eviction_data[policy][lf] for lf in load_factors]

        style = policy_styles.get(policy, {"marker": "o", "linestyle": "-"})
        ax.plot(
            load_factors,
            evictions,
            label=policy,
            linewidth=2.5,
            markersize=8,
            color=style.get("color"),
            marker=style.get("marker", "o"),
            linestyle=style.get("linestyle", "-"),
        )

    ax.set_xlabel("Load Factor", fontsize=14, fontweight="bold")
    ax.set_ylabel("Evictions per Insert", fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(fontsize=12, loc="upper left", framealpha=0)
    ax.set_title("Evictions per Insert vs Load Factor", fontsize=16, fontweight="bold")

    plt.tight_layout()

    output_file = output_dir / "eviction_per_insert.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight", transparent=True)
    typer.secho(f"Evictions per insert plot saved to {output_file}", fg=typer.colors.GREEN)
    plt.close()

    # Plot 2: Total evictions vs load factor
    if total_evictions_data:
        fig, ax = plt.subplots(figsize=(12, 8))

        for policy in sorted(total_evictions_data.keys()):
            load_factors = sorted(total_evictions_data[policy].keys())
            evictions = [total_evictions_data[policy][lf] for lf in load_factors]

            style = policy_styles.get(policy, {"marker": "o", "linestyle": "-"})
            ax.plot(
                load_factors,
                evictions,
                label=policy,
                linewidth=2.5,
                markersize=8,
                color=style.get("color"),
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
            )

        ax.set_xlabel("Load Factor", fontsize=14, fontweight="bold")
        ax.set_ylabel("Total Evictions", fontsize=14, fontweight="bold")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(fontsize=12, loc="upper left", framealpha=0)
        ax.set_title("Total Evictions vs Load Factor", fontsize=16, fontweight="bold")

        plt.tight_layout()

        output_file = output_dir / "eviction_total.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight", transparent=True)
        typer.secho(f"Total evictions plot saved to {output_file}", fg=typer.colors.GREEN)
        plt.close()

    # Plot 3: Throughput (items per second) vs load factor
    if throughput_data:
        fig, ax = plt.subplots(figsize=(12, 8))

        for policy in sorted(throughput_data.keys()):
            load_factors = sorted(throughput_data[policy].keys())
            throughputs = [throughput_data[policy][lf] / 1e6 for lf in load_factors]  # Convert to millions

            style = policy_styles.get(policy, {"marker": "o", "linestyle": "-"})
            ax.plot(
                load_factors,
                throughputs,
                label=policy,
                linewidth=2.5,
                markersize=8,
                color=style.get("color"),
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
            )

        ax.set_xlabel("Load Factor", fontsize=14, fontweight="bold")
        ax.set_ylabel("Throughput [M ops/s]", fontsize=14, fontweight="bold")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(fontsize=12, loc="upper right", framealpha=0)
        ax.set_title("Insert Throughput vs Load Factor (75% Pre-filled)", fontsize=16, fontweight="bold")

        plt.tight_layout()

        output_file = output_dir / "eviction_throughput.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight", transparent=True)
        typer.secho(f"Throughput plot saved to {output_file}", fg=typer.colors.GREEN)
        plt.close()


if __name__ == "__main__":
    app()
