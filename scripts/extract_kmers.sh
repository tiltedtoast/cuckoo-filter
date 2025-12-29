#!/usr/bin/env bash


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DATA_ROOT="${1:-$PROJECT_ROOT/data}"
GENOME_DIR="$DATA_ROOT/genomes"
KMER_DIR="$DATA_ROOT/kmers"
BUILD_DIR="$PROJECT_ROOT/build"

mkdir -p "$KMER_DIR"

K_VALUES=(21 31)

# Check if KMC binaries exist
if [ ! -f "$BUILD_DIR/subprojects/kmc/kmc" ]; then
    echo "Error: kmc not found at $BUILD_DIR/subprojects/kmc/kmc"
    echo "Please run: meson setup build && ninja -C build"
    exit 1
fi

if [ ! -f "$BUILD_DIR/kmc_to_binary" ]; then
    echo "Error: kmc_to_binary not found at $BUILD_DIR/kmc_to_binary"
    echo "Please run: meson setup build && ninja -C build"
    exit 1
fi

KMC="$BUILD_DIR/subprojects/kmc/kmc"
KMC_TO_BINARY="$BUILD_DIR/kmc_to_binary"

echo "Using KMC from: $KMC"
echo "K values: ${K_VALUES[*]}"
echo ""

# Function to extract k-mers for a given genome and k value
extract_kmers() {
    local genome_file=$1
    local genome_name=$2
    local k=$3

    local output_prefix="$KMER_DIR/${genome_name}_${k}"
    local bin_file="${output_prefix}.bin"

    echo "Processing: $genome_name (k=$k)"

    if [ -f "$bin_file" ]; then
        echo "Binary file already exists, skipping"
        return
    fi

    local num_threads
    num_threads=$(nproc 2>/dev/null || echo 4)

    # Create temp directory for KMC
    local kmc_tmp
    kmc_tmp=$(mktemp -d)

    echo "Counting k-mers with KMC (using $num_threads threads)..."

    # KMC count: -ci1 = min count 1, -cs1 = max count 1 (we only care about presence)
    # -fm = multi-fasta mode
    "$KMC" \
        -k"$k" \
        -ci1 \
        -cs1 \
        -t"$num_threads" \
        -m4 \
        -fm \
        "$genome_file" \
        "$output_prefix" \
        "$kmc_tmp"

    # Clean up temp directory
    rm -rf "$kmc_tmp"

    echo "Converting to binary format..."
    "$KMC_TO_BINARY" "$output_prefix" "$bin_file"

    # Clean up KMC database files
    rm -f "${output_prefix}.kmc_pre" "${output_prefix}.kmc_suf"

    echo ""
}

# Extract k-mers for E. coli
if [ -f "$GENOME_DIR/ecoli_k12_mg1655.fna" ]; then
    echo ""
    echo "E. coli K-12 MG1655"
    echo ""
    for k in "${K_VALUES[@]}"; do
        extract_kmers "$GENOME_DIR/ecoli_k12_mg1655.fna" "ecoli" "$k"
    done
    echo ""
else
    echo "Warning: E. coli genome not found, skipping"
    echo "Run ./scripts/download_genomes.sh first"
    echo ""
fi

# Extract k-mers for human chromosome 14
if [ -f "$GENOME_DIR/chr14.fna" ]; then
    echo ""
    echo "Human Chromosome 14"
    echo ""
    for k in "${K_VALUES[@]}"; do
        extract_kmers "$GENOME_DIR/chr14.fna" "chr14" "$k"
    done
    echo ""
else
    echo "Warning: chr14 genome not found, skipping"
    echo "Run ./scripts/download_genomes.sh first"
    echo ""
fi

# Extract k-mers for rice genome
if [ -f "$GENOME_DIR/rice.fna" ]; then
    echo ""
    echo "Rice (Oryza sativa)"
    echo ""
    for k in "${K_VALUES[@]}"; do
        extract_kmers "$GENOME_DIR/rice.fna" "rice" "$k"
    done
    echo ""
else
    echo "Warning: Rice genome not found, skipping"
    echo "Run ./scripts/download_genomes.sh first"
    echo ""
fi

# Extract k-mers for chicken genome
if [ -f "$GENOME_DIR/chicken.fna" ]; then
    echo ""
    echo "Chicken (Gallus gallus)"
    echo ""
    for k in "${K_VALUES[@]}"; do
        extract_kmers "$GENOME_DIR/chicken.fna" "chicken" "$k"
    done
    echo ""
else
    echo "Warning: Chicken genome not found, skipping"
    echo "Run ./scripts/download_genomes.sh first"
    echo ""
fi

# Extract k-mers for full human genome
if [ -f "$GENOME_DIR/human_grch38.fna" ]; then
    echo ""
    echo "Full Human Genome (GRCh38)"
    echo ""
    for k in "${K_VALUES[@]}"; do
        extract_kmers "$GENOME_DIR/human_grch38.fna" "human" "$k"
    done
    echo ""
else
    echo "Warning: Full human genome not found, skipping"
    echo "Run ./scripts/download_genomes.sh first"
    echo ""
fi

# Extract k-mers for wheat genome
if [ -f "$GENOME_DIR/wheat.fna" ]; then
    echo ""
    echo "Wheat Genome (Triticum aestivum)"
    echo ""
    for k in "${K_VALUES[@]}"; do
        extract_kmers "$GENOME_DIR/wheat.fna" "wheat" "$k"
    done
    echo ""
else
    echo "Warning: Wheat genome not found, skipping"
    echo "Run ./scripts/download_genomes.sh first"
    echo ""
fi

echo "Extraction complete"
echo "Binary files are ready for benchmarks in: $KMER_DIR"
