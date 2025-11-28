import io
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

FILTERS = ["cuckoo", "bloom", "tcf"]
OPERATIONS = {
    "cuckoo": ["insert", "query", "delete"],
    "bloom": ["insert", "query"],
    "tcf": ["insert", "query", "delete"],
}

KERNEL_PATTERNS = {
    "cuckoo": {
        "insert": ["insertKernel"],
        "query": ["containsKernel"],
        "delete": ["deleteKernel"],
    },
    "bloom": {
        "insert": ["add"],
        "query": ["contains_if_n"],
    },
    "tcf": {
        "insert": ["sorted_bulk_insert_kernel"],
        "query": ["bulk_sorted_query_kernel"],
        "delete": ["bulk_sorted_delete_kernel"],
    },
}


def to_superscript(n: int) -> str:
    """Convert a number to its superscript representation."""
    superscripts = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    return str(n).translate(superscripts)


def run_ncu_profile(
    executable: Path,
    filter_type: str,
    operation: str,
    capacity_exponent: int,
    load_factor: float,
    metrics: list[str],
) -> Optional[dict[str, float]]:
    """
    Run ncu profiling for a specific configuration and return requested metrics.

    Args:
        executable: Path to the benchmark executable
        filter_type: Type of filter (cuckoo, bloom, tcf)
        operation: Operation (insert, query, delete)
        capacity_exponent: Log2 of capacity (e.g. 20 for 1M)
        load_factor: Load factor (e.g. 0.95)
        metrics: List of NCU metric names to collect

    Returns:
        dictionary mapping metric names to their average values, or None if failed.
    """
    capacity = 2**capacity_exponent

    typer.secho(
        f"Profiling {filter_type}/{operation} @ capacity=2{to_superscript(capacity_exponent)} ({capacity:,})...",
        fg=typer.colors.CYAN,
        err=True,
    )

    metrics_arg = ",".join(metrics)
    cmd = [
        "ncu",
        "--metrics",
        metrics_arg,
        "--csv",
        "--page",
        "raw",
        str(executable),
        filter_type,
        operation,
        str(capacity_exponent),
        "--load-factor",
        str(load_factor),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,  # 5 minute timeout per run
        )

        lines = result.stdout.strip().split("\n")

        csv_start_idx = None
        for i, line in enumerate(lines):
            if line.startswith('"'):
                csv_start_idx = i
                break

        if csv_start_idx is None:
            typer.secho(
                "No CSV data found in NCU output",
                fg=typer.colors.RED,
                err=True,
            )
            return None

        # Get CSV lines starting from the header
        csv_lines = lines[csv_start_idx:]
        if len(csv_lines) < 3:  # Need at least header + units + 1 data row
            typer.secho(
                "Insufficient CSV data",
                fg=typer.colors.RED,
                err=True,
            )
            return None

        try:
            csv_content = "\n".join(csv_lines)
            df = pd.read_csv(
                io.StringIO(csv_content),
                on_bad_lines="skip",
                skiprows=[1],  # Skip the units row (2nd row)
            )
        except Exception as e:
            typer.secho(
                f"Failed to parse CSV: {e}",
                fg=typer.colors.RED,
                err=True,
            )
            return None

        collected_metrics = {}

        patterns = KERNEL_PATTERNS.get(filter_type, {}).get(operation, [])

        if "Kernel Name" in df.columns:
            # Create a boolean mask for matching kernels
            mask = pd.Series(False, index=df.index)
            for pattern in patterns:
                mask |= (
                    df["Kernel Name"]
                    .astype(str)
                    .str.contains(pattern, case=False, na=False)
                )

            target_df = df[mask]

            if target_df.empty:
                typer.secho(
                    f"ERROR: No kernels matched patterns {patterns}",
                    fg=typer.colors.RED,
                    err=True,
                )
                return None

        else:
            target_df = df

        if target_df.empty:
            typer.secho(
                "No relevant kernels found to profile",
                fg=typer.colors.RED,
                err=True,
            )
            return None

        for metric in metrics:
            if metric in target_df.columns:
                # Get non-null values for this metric and compute mean
                values = target_df[metric].dropna()
                if len(values) > 0:
                    collected_metrics[metric] = float(values.mean())

        if not collected_metrics:
            typer.secho(
                "No metrics found in NCU output",
                fg=typer.colors.YELLOW,
                err=True,
            )
            return None

        return collected_metrics

    except subprocess.TimeoutExpired:
        typer.secho(
            f"Timeout for {filter_type}/{operation} @ {capacity:,}",
            fg=typer.colors.YELLOW,
            err=True,
        )
        return None
    except subprocess.CalledProcessError as e:
        typer.secho(
            f"NCU profiling failed: {e.stderr[:200]}",
            fg=typer.colors.RED,
            err=True,
        )
        return None
    except Exception as e:
        typer.secho(
            f"Error: {e}",
            fg=typer.colors.RED,
            err=True,
        )
        return None
