#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "typer",
# ]
# ///
"""Convert k-mer text files to binary format for benchmarks.

K-mers are DNA sequences over the alphabet {A, C, G, T}. We encode each
nucleotide using 2 bits (A=00, C=01, G=10, T=11).

Binary format:
    - uint64_t: Number of k-mers (N)
    - N x uint64_t: Encoded k-mers
"""

import struct
from pathlib import Path

import typer

NUCLEOTIDE_ENCODING = {
    "A": 0b00,
    "C": 0b01,
    "G": 0b10,
    "T": 0b11,
}


def encode_kmer(kmer: str) -> int:
    """Encode a DNA k-mer string to a uint64_t integer.

    Args:
        kmer: DNA sequence string (A/C/G/T)

    Returns:
        Integer encoding of the k-mer

    Raises:
        ValueError: If k-mer contains invalid characters or is too long
    """
    if len(kmer) > 32:
        raise ValueError(f"K-mer too long: {len(kmer)} (max 32)")

    encoded = 0
    for nucleotide in kmer.upper():
        if nucleotide not in NUCLEOTIDE_ENCODING:
            raise ValueError(f"Invalid nucleotide: {nucleotide}")

        encoded = (encoded << 2) | NUCLEOTIDE_ENCODING[nucleotide]

    return encoded


def convert_kmer_file(input_path: Path, output_path: Path | None = None) -> None:
    """Convert a k-mer text file to binary format.

    Input format: Tab-separated with k-mer and count per line (Jellyfish dump -c -t output)
    Output format: Binary file with count followed by encoded k-mers

    Args:
        input_path: Path to input text file
        output_path: Path to output binary file (default: input_path with .bin extension)
    """
    if output_path is None:
        output_path = input_path.with_suffix(".bin")

    typer.echo(f"Converting {input_path} to {output_path}")

    kmers = []
    invalid_count = 0
    kmer_length = None
    first_kmer = None

    # Read and encode k-mers
    with open(input_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # Parse Jellyfish output format: "kmer\tcount"
            parts = line.split("\t")
            if len(parts) != 2:
                invalid_count += 1
                if invalid_count <= 3:
                    typer.secho(
                        f"Warning: Invalid format at line {line_num} (expected tab-separated)",
                        fg=typer.colors.YELLOW,
                    )
                continue

            kmer_str = parts[0]

            # Store first k-mer as example
            if first_kmer is None:
                first_kmer = kmer_str
                kmer_length = len(kmer_str)

            # Check k-mer length
            if len(kmer_str) > 32:
                if invalid_count == 0:
                    typer.secho(
                        f"\nError: K-mer length {len(kmer_str)} exceeds maximum of 32",
                        fg=typer.colors.RED,
                    )
                    typer.secho(
                        "This script only supports k<=32 (fits in uint64_t with 2-bit encoding)",
                        fg=typer.colors.RED,
                    )
                    typer.secho(
                        "\nTo use k>32, you'll need to modify the benchmark to use larger integer types.",
                        fg=typer.colors.YELLOW,
                    )
                invalid_count += 1
                continue

            try:
                encoded = encode_kmer(kmer_str)
                kmers.append(encoded)
            except ValueError as e:
                invalid_count += 1
                if invalid_count <= 3:
                    typer.secho(
                        f"Warning: {e} at line {line_num}",
                        fg=typer.colors.YELLOW,
                    )

    if kmers:
        typer.echo(f"Successfully encoded {len(kmers):,} k-mers (k={kmer_length})")

    if invalid_count > 0:
        typer.secho(
            f"Skipped {invalid_count:,} invalid k-mers",
            fg=typer.colors.YELLOW,
        )

    if not kmers:
        typer.secho(
            "Error: No valid k-mers found in file!",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    # Write binary file
    with open(output_path, "wb") as f:
        # Write count
        f.write(struct.pack("<Q", len(kmers)))

        # Write k-mers
        for kmer in kmers:
            f.write(struct.pack("<Q", kmer))

    file_size = output_path.stat().st_size
    typer.secho(
        f"Wrote {len(kmers):,} k-mers ({file_size / 1024 / 1024:.2f} MB)",
        fg=typer.colors.GREEN,
    )


def main(
    input_files: list[Path] = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="K-mer text files to convert",
    ),
) -> None:
    """Convert k-mer text files to binary format for CUDA benchmarks."""
    for input_file in input_files:
        try:
            convert_kmer_file(input_file)
        except Exception as e:
            typer.secho(
                f"Error processing {input_file}: {e}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)

    typer.secho("\nAll files converted successfully!", fg=typer.colors.GREEN)


if __name__ == "__main__":
    typer.run(main)
