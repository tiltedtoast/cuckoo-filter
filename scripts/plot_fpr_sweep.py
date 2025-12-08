#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "typer",
# ]
# ///

import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from matplotlib.patches import Patch

app = typer.Typer(help="Plot FPR sweep benchmark results")

FILTER_TYPES = {
    "GPUCF": "GPU Cuckoo",
    "Bloom": "Blocked Bloom",
    "TCF": "TCF",
    "GQF": "GQF",
}

FILTER_COLORS = {
    "GPU Cuckoo": "#2E86AB",
    "Blocked Bloom": "#A23B72",
    "TCF": "#C73E1D",
    "GQF": "#F18F01",
}

FPR_TARGETS = [
    (0.10, r"$\leq 10\%$"),
    (0.01, r"$\leq 1\%$"),
    (0.001, r"$\leq 0.1\%$"),
    (0.0001, r"$\leq 0.01\%$"),
]


def parse_benchmark_name(name: str) -> dict:
    """Extract filter type, fingerprint bits, load factor, and operation from benchmark name."""
    result = {
        "filter": None,
        "fingerprint_bits": None,
        "load_factor": None,
        "operation": "negative_query",  # FPR_Sweep measures negative lookups
    }

    if "_Insert_Sweep" in name:
        result["operation"] = "insert"
    elif "_PositiveQuery_Sweep" in name:
        result["operation"] = "positive_query"

    for prefix, filter_name in FILTER_TYPES.items():
        if name.startswith(prefix):
            result["filter"] = filter_name
            break

    type_to_bits = {
        "uint8_t": 8,
        "uint16_t": 16,
        "uint32_t": 32,
        "uint64_t": 64,
    }

    if "<" in name and ">" in name:
        params = name[name.index("<") + 1 : name.index(">")].split(",")
        params = [p.strip() for p in params]

        if len(params) == 2:
            first_param = params[0]
            if first_param in type_to_bits:
                result["fingerprint_bits"] = type_to_bits[first_param]
            else:
                result["fingerprint_bits"] = int(first_param)
            result["load_factor"] = int(params[1]) / 100.0
        elif len(params) == 1:
            result["load_factor"] = int(params[0]) / 100.0

    return result


def create_fpr_target_comparison(df: pd.DataFrame, output_dir: Path, hit_rate: float):
    """Create grouped bar chart with insert & query bars side by side for each filter."""

    # Separate by operation type
    neg_query_df = df[df["operation"] == "negative_query"].copy()
    pos_query_df = df[df["operation"] == "positive_query"].copy()
    insert_df = df[df["operation"] == "insert"].copy()

    filter_names = [
        f for f in FILTER_COLORS.keys() if f in neg_query_df["filter"].values
    ]

    # Create config key to match data across operations
    def make_config_key(row):
        return f"{row['filter']}_{row.get('fingerprint_bits', '')}_{row.get('load_factor', '')}"

    neg_query_df["config_key"] = neg_query_df.apply(make_config_key, axis=1)

    # Map positive query throughput
    if len(pos_query_df) > 0:
        pos_query_df["config_key"] = pos_query_df.apply(make_config_key, axis=1)
        pos_lookup = pos_query_df.set_index("config_key")["throughput_mops"].to_dict()
        neg_query_df["positive_query_throughput"] = neg_query_df["config_key"].map(
            pos_lookup
        )
    else:
        neg_query_df["positive_query_throughput"] = np.nan

    # Calculate weighted average query throughput based on hit rate
    neg_query_df["avg_query_throughput"] = neg_query_df.apply(
        lambda r: (
            hit_rate * r["positive_query_throughput"]
            + (1 - hit_rate) * r["throughput_mops"]
        )
        if pd.notna(r["positive_query_throughput"])
        else r["throughput_mops"],
        axis=1,
    )

    # Map insert throughput
    if len(insert_df) > 0:
        insert_df["config_key"] = insert_df.apply(make_config_key, axis=1)
        insert_lookup = insert_df.set_index("config_key")["throughput_mops"].to_dict()
        neg_query_df["insert_throughput"] = neg_query_df["config_key"].map(
            insert_lookup
        )
    else:
        neg_query_df["insert_throughput"] = np.nan

    has_insert_data = neg_query_df["insert_throughput"].notna().any()
    has_positive_data = neg_query_df["positive_query_throughput"].notna().any()

    fig, ax = plt.subplots(figsize=(14, 8))

    n_fpr_targets = len(FPR_TARGETS)
    n_filters = len(filter_names)

    # Each FPR target has n_filters groups, each group has 2 bars (query + insert)
    bar_width = 0.35
    group_width = bar_width * 2 + 0.1
    target_width = n_filters * group_width + 0.5

    for target_idx, (fpr_target, fpr_label) in enumerate(FPR_TARGETS):
        target_offset = target_idx * target_width

        for filter_idx, filter_name in enumerate(filter_names):
            filter_data = neg_query_df[neg_query_df["filter"] == filter_name]
            qualifying = filter_data[filter_data["fpr_percentage"] / 100 <= fpr_target]

            # Use weighted average query throughput if positive data available
            if has_positive_data:
                query_tp = (
                    qualifying["avg_query_throughput"].max()
                    if len(qualifying) > 0
                    else 0
                )
            else:
                query_tp = (
                    qualifying["throughput_mops"].max() if len(qualifying) > 0 else 0
                )

            insert_qualifying = qualifying[qualifying["insert_throughput"].notna()]
            insert_tp = (
                insert_qualifying["insert_throughput"].max()
                if len(insert_qualifying) > 0
                else 0
            )

            x_base = target_offset + filter_idx * group_width

            # Query bar (solid)
            ax.bar(
                x_base,
                query_tp if query_tp > 0 else 0.1,
                bar_width,
                color=FILTER_COLORS[filter_name],
                edgecolor="white",
                linewidth=0.5,
            )

            # Insert bar (hatched)
            if has_insert_data:
                ax.bar(
                    x_base + bar_width,
                    insert_tp if insert_tp > 0 else 0.1,
                    bar_width,
                    color=FILTER_COLORS[filter_name],
                    edgecolor="white",
                    linewidth=0.5,
                    hatch="//",
                    alpha=0.7,
                )

    # Set x-axis labels at center of each FPR target group
    target_centers = [
        i * target_width + (n_filters * group_width) / 2 - group_width / 2
        for i in range(n_fpr_targets)
    ]
    ax.set_xticks(target_centers)
    ax.set_xticklabels([label for _, label in FPR_TARGETS])

    ax.set_xlabel("Target FPR")
    ax.set_ylabel("Throughput (MOPS)")
    hit_pct = int(hit_rate * 100)
    ax.set_title(rf"Best Throughput by Target FPR ({hit_pct}% hit rate)")
    # ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    legend_elements = [
        Patch(facecolor=FILTER_COLORS[name], label=name) for name in filter_names
    ]
    legend_elements.append(Patch(facecolor="gray", label="Query"))
    legend_elements.append(
        Patch(facecolor="gray", hatch="//", alpha=0.7, label="Insert")
    )
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    output_path = output_dir / "fpr_sweep_throughput.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.secho(f"Saved throughput comparison to {output_path}", fg=typer.colors.GREEN)


@app.command()
def main(
    csv_file: Path = typer.Argument(
        "-",
        help="Path to benchmark CSV file, or '-' to read from stdin (default: stdin)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots (default: build/)",
    ),
    hit_rate: float = typer.Option(
        50.0,
        "--hit-rate",
        "-h",
        help="Expected percentage of positive queries (0-100, default: 50)",
    ),
):
    try:
        if str(csv_file) == "-":
            df = pd.read_csv(sys.stdin)
        elif not csv_file.exists():
            typer.secho(
                f"Error: File not found: {csv_file}", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(1)
        else:
            df = pd.read_csv(csv_file)
    except Exception as e:
        typer.secho(f"Error parsing CSV: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    df = df[df["name"].str.contains("_median", na=False)]

    parsed = df["name"].apply(parse_benchmark_name)
    df["filter"] = parsed.apply(lambda x: x["filter"])
    df["fingerprint_bits"] = parsed.apply(lambda x: x["fingerprint_bits"])
    df["load_factor"] = parsed.apply(lambda x: x["load_factor"])
    df["operation"] = parsed.apply(lambda x: x["operation"])

    df["throughput_mops"] = df["items_per_second"] / 1e6

    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / "build"

    output_dir.mkdir(parents=True, exist_ok=True)

    create_fpr_target_comparison(df, output_dir, hit_rate / 100.0)


if __name__ == "__main__":
    app()
