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

app = typer.Typer(help="Plot 128-bit vs 256-bit load width benchmark results")


@app.command()
def main(
    csv_file: Path = typer.Argument(..., help="Path to CSV file"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Output directory for plots"
    ),
):
    """Generate comparison plot for 128-bit vs 256-bit load widths."""
    df = pd.read_csv(csv_file)
    # Filter for median records only (format: "128bit/Query/65536/.../manual_time_median")
    df = df[df["name"].str.contains("_median")]

    benchmark_data = defaultdict(dict)
    for _, row in df.iterrows():
        name = row["name"]
        parts = name.split("/")
        if len(parts) < 3:
            continue

        # Extract variant (128bit or 256bit) from first part
        variant_raw = parts[0].strip('"')
        if "128bit" in variant_raw:
            variant = "128-bit"
        elif "256bit" in variant_raw:
            variant = "256-bit"
        else:
            continue

        try:
            # Size is the third part (after variant and "Query")
            size = int(parts[2])
            items_per_second = row.get("items_per_second")
            if pd.notna(items_per_second):
                throughput_mops = items_per_second / 1_000_000
                benchmark_data[variant][size] = throughput_mops
        except (ValueError, KeyError, IndexError):
            continue

    if not benchmark_data:
        typer.secho("No data found in CSV", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "build"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"128-bit": "#E63946", "256-bit": "#2A9D8F"}

    for variant in ["128-bit", "256-bit"]:
        if variant not in benchmark_data:
            continue
        sizes = sorted(benchmark_data[variant].keys())
        throughput = [benchmark_data[variant][s] for s in sizes]
        ax.plot(
            sizes,
            throughput,
            "o-",
            label=f"{variant} loads",
            color=colors[variant],
            linewidth=2.5,
            markersize=8,
        )

    ax.set_xlabel("Input Size", fontsize=14, fontweight="bold")
    ax.set_ylabel("Throughput [M ops/s]", fontsize=14, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=12, loc="best", framealpha=0)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.set_title(
        "Query Throughput: 128-bit vs 256-bit Loads", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()

    output_file = output_dir / "load_width_comparison.pdf"
    plt.savefig(
        output_file,
        bbox_inches="tight",
        transparent=True,
        format="pdf",
        dpi=600,
    )
    typer.secho(f"Plot saved to {output_file}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
