#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "typer",
# ]
# ///

import os
from pathlib import Path

import benchmark_utils as bu
import typer

app = typer.Typer()


@app.command()
def main(
    output: Path = typer.Option(
        Path("build/multi_gpu_scaling.csv"),
        "--output",
        "-o",
        help="Output CSV file path",
    ),
    gpu_counts: str = typer.Option(
        "2,4,6,8",
        "--gpu-counts",
        "-g",
        help="Comma-separated list of GPU counts to test",
    ),
):
    """Run multi-GPU scaling benchmark with varying GPU counts.

    Uses CUDA_VISIBLE_DEVICES to control which GPUs are used for each run.
    Combines results into a single CSV file.
    """
    build_dir = bu.get_build_dir(Path(__file__))
    benchmark_exe = build_dir / "benchmark-multi-gpu-scaling"

    bu.validate_executable(benchmark_exe)

    counts = [int(c.strip()) for c in gpu_counts.split(",")]

    benchmarks = []
    for num_gpus in counts:
        # Create GPU list: 0,1 for 2 GPUs, 0,1,2,3 for 4 GPUs, etc.
        gpu_list = ",".join(str(i) for i in range(num_gpus))

        typer.echo(f"\n{'=' * 60}")
        typer.echo(f"Running with {num_gpus} GPUs (CUDA_VISIBLE_DEVICES={gpu_list})")
        typer.echo(f"{'=' * 60}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_list

        benchmarks.append((benchmark_exe, env, None))

    bu.run_benchmarks_and_merge(
        benchmarks,
        output,
        show_progress=False,  # We're showing our own progress messages
    )

    typer.secho(f"\nResults written to {output}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
