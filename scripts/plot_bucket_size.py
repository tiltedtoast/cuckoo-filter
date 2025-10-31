#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "pandas",
#     "seaborn"
# ]
# ///
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns
import json
import re

data = json.load(sys.stdin)
df = pd.DataFrame.from_records(data["benchmarks"])


def parse_benchmark_name(name):
    # Pattern: BM_CuckooFilter_<Operation><<BucketSize>>/<InputSize>
    match = re.match(r"BM_CuckooFilter_(\w+)<\d+>/(\d+)", name)
    if match:
        operation = match.group(1)
        input_size = int(match.group(2))
        return pd.Series(
            {
                "operation": operation,
                "input_size": input_size,
                "exponent": int(np.log2(input_size)),
            }
        )
    return pd.Series({"operation": None, "input_size": None, "exponent": None})


parsed = df["name"].apply(parse_benchmark_name)
df = pd.concat([df, parsed], axis=1)

# bucket_size comes as float from json
df["bucket_size"] = df["bucket_size"].astype(int)

df_filtered = df[df["operation"].isin(["Insert", "Query"])].copy()

df_filtered["time_ms"] = df_filtered["real_time"]
df_filtered["throughput_mops"] = df_filtered["items_per_second"] / 1_000_000

plt.style.use("default")
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
    }
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))


def create_performance_heatmap(df, operation, ax):
    subset = df[df["operation"] == operation].copy()

    subset["normalized_time"] = subset.groupby("exponent")["time_ms"].transform(
        lambda x: x / x.min()
    )

    pivot_table = subset.pivot(
        index="exponent",
        columns="bucket_size",
        values="normalized_time",
    )

    sns.heatmap(
        pivot_table,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="rocket_r",
        cbar_kws={"label": "Performance Ratio (1.0 = Optimal)"},
        vmin=1.0,
        vmax=pivot_table.max().max() if pivot_table.max().max() > 1.0 else 2.0,
    )

    ax.set_title(f"{operation} Performance vs. Bucket Size")
    ax.set_xlabel("Bucket Size")
    ax.set_ylabel("Input Size")
    ax.set_yticklabels([f"$2^{{{int(exp)}}}$" for exp in pivot_table.index], rotation=0)


create_performance_heatmap(df_filtered, "Insert", ax1)
create_performance_heatmap(df_filtered, "Query", ax2)

script_dir = Path(__file__).parent
build_dir = script_dir.parent / "build"
build_dir.mkdir(exist_ok=True)

output_file = build_dir / "benchmark_bucket_size.png"

plt.tight_layout()
plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)

print(f"Bucket size performance plot saved to {output_file}")
