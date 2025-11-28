#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas",
#   "typer",
# ]
# ///


import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from ncu_profiler import FILTERS, OPERATIONS, run_ncu_profile, to_superscript

app = typer.Typer(help="Run cache benchmarks with NCU profiling across input sizes")

DEFAULT_MIN_CAPACITY_LOG2 = 16
DEFAULT_MAX_CAPACITY_LOG2 = 28

CACHE_METRICS = [
    "l1tex__t_sector_hit_rate.pct",  # L1/tex hit rate (%)
    "lts__t_sector_hit_rate.pct",  # L2 hit rate (%)
]


def get_hit_rates(metrics: dict) -> tuple[float, float]:
    """Extract L1 and L2 cache hit rates from NCU metrics (already in %)."""
    l1_hit_rate = metrics.get("l1tex__t_sector_hit_rate.pct", 0.0)
    l2_hit_rate = metrics.get("lts__t_sector_hit_rate.pct", 0.0)

    return l1_hit_rate, l2_hit_rate


@app.command()
def main(
    executable: Path = typer.Argument(
        ...,
        help="Path to the cache_benchmark executable",
        exists=True,
    ),
    output: Path = typer.Option(
        "cache_benchmark_results.csv",
        "--output",
        "-o",
        help="Output CSV file for results",
    ),
    min_capacity_log2: int = typer.Option(
        DEFAULT_MIN_CAPACITY_LOG2,
        "--min-log2",
        help="Minimum capacity as log2(capacity)",
    ),
    max_capacity_log2: int = typer.Option(
        DEFAULT_MAX_CAPACITY_LOG2,
        "--max-log2",
        help="Maximum capacity as log2(capacity)",
    ),
    load_factor: float = typer.Option(
        0.95,
        "--load-factor",
        "-l",
        help="Load factor for all runs",
    ),
    filters: Optional[list[str]] = typer.Option(
        None,
        "--filter",
        "-f",
        help="Specific filters to test (can specify multiple)",
    ),
):
    """
    Profile cache hit rates across varying input sizes.

    This command runs NVIDIA Nsight Compute to measure L1 and L2 cache hit rates
    for various filter operations across a range of capacities.

    Example:
        run_cache_benchmark.py ./build/cache-benchmark
        run_cache_benchmark.py ./build/cache-benchmark --min-log2 18 --max-log2 24
        run_cache_benchmark.py ./build/cache-benchmark -f cuckoo -f bloom
    """
    if not executable.exists():
        typer.secho(
            f"Executable not found: {executable}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    # Check if ncu is available
    try:
        subprocess.run(["ncu", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        typer.secho(
            "NVIDIA Nsight Compute (ncu) not found. Please install it.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    test_filters = filters if filters else FILTERS

    exponents = list(range(min_capacity_log2, max_capacity_log2 + 1))
    capacities = [2**e for e in exponents]

    typer.secho(
        f"Testing {len(test_filters)} filter(s) across {len(exponents)} capacity values with load factor {load_factor}",
        fg=typer.colors.GREEN,
        err=True,
    )
    typer.secho(
        f"Capacity range: 2{to_superscript(exponents[0])} ({capacities[0]:,}) to 2{to_superscript(exponents[-1])} ({capacities[-1]:,})",
        fg=typer.colors.GREEN,
        err=True,
    )

    results = []
    total_runs = sum(len(OPERATIONS[f]) for f in test_filters) * len(exponents)
    current_run = 0

    for filter_type in test_filters:
        for operation in OPERATIONS[filter_type]:
            for exponent in exponents:
                current_run += 1
                typer.secho(
                    f"\n[{current_run}/{total_runs}] ",
                    fg=typer.colors.BRIGHT_BLUE,
                    err=True,
                    nl=False,
                )

                metrics = run_ncu_profile(
                    executable,
                    filter_type,
                    operation,
                    exponent,
                    load_factor,
                    CACHE_METRICS,
                )

                if metrics:
                    l1_hit_rate, l2_hit_rate = get_hit_rates(metrics)
                    capacity = 2**exponent  # Compute capacity for storage

                    results.append(
                        {
                            "filter": filter_type,
                            "operation": operation,
                            "capacity": capacity,
                            "load_factor": load_factor,
                            "l1_hit_rate": l1_hit_rate,
                            "l2_hit_rate": l2_hit_rate,
                        }
                    )

                    typer.secho(
                        f"  L1: {l1_hit_rate:.2f}%, L2: {l2_hit_rate:.2f}%",
                        fg=typer.colors.GREEN,
                        err=True,
                    )

    if not results:
        typer.secho(
            "\nNo results collected",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    df = pd.DataFrame(results)
    df.to_csv(output, index=False)

    typer.secho(
        f"\nResults saved to {output} ({len(results)} data points)",
        fg=typer.colors.GREEN,
        err=True,
    )

    typer.secho("\nSummary Statistics:", fg=typer.colors.BRIGHT_CYAN, err=True)
    summary = (
        df.groupby(["filter", "operation"])
        .agg(
            {
                "l1_hit_rate": ["mean", "min", "max"],
                "l2_hit_rate": ["mean", "min", "max"],
            }
        )
        .round(2)
    )
    print(summary)


if __name__ == "__main__":
    app()
    app()
