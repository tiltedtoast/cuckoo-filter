#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "typer",
# ]
# ///

import subprocess
import tempfile
from pathlib import Path

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
    build_dir = Path(__file__).parent.parent / "build"

    benchmarks = [
        ("benchmark-load-width-256bit", "256bit"),
        ("benchmark-load-width-128bit", "128bit"),
    ]

    all_lines = []
    header = None

    for bench_name, label in benchmarks:
        bench_path = build_dir / bench_name
        if not bench_path.exists():
            typer.echo(
                f"Error: {bench_path} not found. Did you run 'meson compile -C build'?",
                err=True,
            )
            raise typer.Exit(1)

        typer.echo(f"Running {label} load width benchmark...")

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as f:
            csv_path = f.name

        result = subprocess.run(
            [
                str(bench_path),
                f"--benchmark_out={csv_path}",
                "--benchmark_out_format=csv",
                "--benchmark_format=csv",
            ],
        )

        if result.returncode != 0:
            typer.echo(f"Error running {bench_name}", err=True)
            raise typer.Exit(1)

        with open(csv_path) as f:
            lines = [line.rstrip() for line in f if line.strip()]

        Path(csv_path).unlink()

        if not lines:
            continue

        # Replace fixture name with label
        lines = [line.replace("LoadWidthFixture", label) for line in lines]

        if header is None:
            header = lines[0]
            all_lines.append(header)
            all_lines.extend(lines[1:])
        else:
            if lines[0] == header:
                all_lines.extend(lines[1:])
            else:
                all_lines.extend(lines)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for line in all_lines:
            f.write(line + "\n")

    typer.echo(f"\nResults written to {output}")


if __name__ == "__main__":
    app()
