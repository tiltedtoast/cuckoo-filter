#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "typer",
# ]
# ///

from pathlib import Path

import benchmark_utils as bu
import typer

app = typer.Typer()


@app.command()
def main(
    output: Path = typer.Option(
        Path("build/benchmark_load_width.csv"),
        "--output",
        "-o",
        help="Output CSV file path",
    ),
):
    """Run 128-bit and 256-bit load width benchmarks and combine results."""
    build_dir = bu.get_build_dir(Path(__file__))

    benchmark_configs = [
        ("benchmark-load-width-256bit", "256bit"),
        ("benchmark-load-width-128bit", "128bit"),
    ]

    benchmarks = [
        (
            build_dir / name,
            None,  # no custom environment
            lambda line, lbl=label: line.replace(
                "LoadWidthFixture", lbl
            ),  # transformer
        )
        for name, label in benchmark_configs
    ]

    bu.run_benchmarks_and_merge(benchmarks, output)

    typer.secho(f"\nResults written to {output}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
