#!/usr/bin/env python3
"""Run k-mer benchmarks with real genomic data.

This script runs the k-mer benchmark executable with specified k-mer data files
and outputs results to CSV format.
"""

from pathlib import Path

import typer
from benchmark_utils import (
    get_build_dir,
    run_benchmark_to_csv,
    validate_executable,
)

app = typer.Typer()


@app.command()
def main(
    kmer_file: Path = typer.Option(
        ...,
        "--kmer-file",
        "-k",
        exists=True,
        dir_okay=False,
        help="Path to binary k-mer file",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output CSV file path",
    ),
    multi_gpu: bool = typer.Option(
        False,
        "--multi-gpu",
        help="Use multi-GPU benchmark",
    ),
    build_dir: Path = typer.Option(
        None,
        "--build-dir",
        "-b",
        help="Build directory (default: ../build relative to script)",
    ),
) -> None:
    """Run k-mer benchmark with real genomic data."""
    if build_dir is None:
        build_dir = get_build_dir(Path(__file__))

    executable = build_dir / "kmer_benchmark"
    validate_executable(executable)

    extra_args = [
        "--kmer-file",
        str(kmer_file.absolute()),
    ]

    if multi_gpu:
        extra_args.append("--multi-gpu")

    typer.echo(f"K-mer file: {kmer_file}")
    typer.echo(f"Multi-GPU: {multi_gpu}")
    typer.echo(f"Output: {output}")
    typer.echo("")

    typer.echo("Running k-mer benchmark...")
    run_benchmark_to_csv(executable, output, extra_args=extra_args)

    typer.secho(
        f"\nBenchmark complete! Results saved to {output}", fg=typer.colors.GREEN
    )


if __name__ == "__main__":
    app()
