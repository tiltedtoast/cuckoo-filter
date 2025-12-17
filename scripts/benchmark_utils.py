"""Shared utilities for running Google Benchmark executables and processing results.

This module provides common functionality for benchmark runner scripts, including:
- Validating executable existence
- Running benchmarks with CSV output
- Merging multiple CSV files with header handling
- Build directory resolution
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Callable

import typer


def get_build_dir(script_path: Optional[Path] = None) -> Path:
    """Get the build directory relative to script location.

    Args:
        script_path: Path to the calling script (typically __file__)
                    If None, assumes current directory's parent has build/

    Returns:
        Path to the build directory
    """
    if script_path is None:
        return Path.cwd().parent / "build"
    return Path(script_path).parent.parent / "build"


def validate_executable(executable: Path) -> None:
    """Validate that a benchmark executable exists.

    Args:
        executable: Path to the executable to validate

    Raises:
        typer.Exit: If the executable does not exist
    """
    if not executable.exists():
        typer.secho(
            f"Error: {executable} not found. Did you run 'meson compile -C build'?",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)


def run_benchmark_to_csv(
    executable: Path,
    csv_output: Path,
    extra_args: Optional[list[str]] = None,
    env: Optional[dict] = None,
) -> None:
    """Run a Google Benchmark executable with CSV output.

    Args:
        executable: Path to the benchmark executable
        csv_output: Path where CSV output should be written
        extra_args: Additional command-line arguments to pass
        env: Optional environment variables (merged with os.environ)

    Raises:
        typer.Exit: If the benchmark fails to execute
    """
    cmd = [
        str(executable),
        f"--benchmark_out={csv_output}",
        "--benchmark_out_format=csv",
        "--benchmark_format=csv",
    ]

    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        typer.secho(
            f"Error running {executable.name}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)


def merge_csv_files(
    csv_paths: list[Path],
    output: Path,
    line_transformers: Optional[list[Callable[[str], str]]] = None,
) -> None:
    """Merge multiple CSV files into one, handling headers correctly.

    The first file's header is kept, and subsequent files' data rows are appended.
    If subsequent files have a different header, all their lines are appended.

    Args:
        csv_paths: List of CSV file paths to merge (in order)
        output: Output path for the merged CSV
        line_transformers: Optional list of functions to transform each line.
                          If provided, must be same length as csv_paths.
                          Each transformer is applied to lines from corresponding CSV.

    Example:
        # Replace fixture names in second CSV
        merge_csv_files(
            [csv1, csv2],
            output,
            [None, lambda line: line.replace("OldName", "NewName")]
        )
    """
    if line_transformers and len(line_transformers) != len(csv_paths):
        raise ValueError("line_transformers must match csv_paths length")

    all_lines = []
    header = None

    for idx, csv_path in enumerate(csv_paths):
        with open(csv_path) as f:
            # Filter out empty lines and Google Benchmark metadata
            # CSV lines start with "name or quotes, metadata lines don't
            lines = [
                line.rstrip()
                for line in f
                if line.strip() and (line.startswith('"') or line.startswith("name"))
            ]

        if not lines:
            continue

        # Apply transformer if provided
        if line_transformers and line_transformers[idx]:
            lines = [line_transformers[idx](line) for line in lines]

        if header is None:
            # First file: keep header and all data
            header = lines[0]
            all_lines.append(header)
            all_lines.extend(lines[1:])
        else:
            # Subsequent files: skip header if it matches, otherwise keep all
            if lines[0] == header:
                all_lines.extend(lines[1:])
            else:
                all_lines.extend(lines)

    # Write merged CSV
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for line in all_lines:
            f.write(line + "\n")


def run_benchmarks_and_merge(
    benchmarks: list[tuple[Path, Optional[dict], Optional[Callable[[str], str]]]],
    output: Path,
    show_progress: bool = True,
) -> None:
    """Run multiple benchmarks and merge their CSV outputs.

    This is a convenience function that combines run_benchmark_to_csv and
    merge_csv_files for the common case of running several benchmarks.

    Args:
        benchmarks: List of (executable, env, line_transformer) tuples,
                    where env and line_transformer can be None
        output: Output path for merged CSV
        show_progress: Whether to show progress messages

    Example:
        run_benchmarks_and_merge(
            [
                (build_dir / "bench1", None, None),
                (build_dir / "bench2", {"VAR": "value"}, lambda l: l.replace("X", "Y")),
            ],
            Path("output.csv")
        )
    """
    temp_csvs = []

    try:
        for idx, (exe, env, transformer) in enumerate(benchmarks, 1):
            validate_executable(exe)

            if show_progress:
                typer.echo(f"Running benchmark {idx}/{len(benchmarks)}: {exe.name}")

            # Create temp file for this benchmark's output
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".csv", delete=False
            ) as f:
                temp_csv = Path(f.name)

            temp_csvs.append(temp_csv)
            run_benchmark_to_csv(exe, temp_csv, env=env)

        # Merge all CSVs
        transformers = [b[2] for b in benchmarks]
        merge_csv_files(temp_csvs, output, transformers)  # ty:ignore[invalid-argument-type]

    finally:
        # Clean up temp files
        for temp_csv in temp_csvs:
            if temp_csv.exists():
                temp_csv.unlink()
