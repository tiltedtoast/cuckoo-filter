#!/usr/bin/env bash
# Extract k-mers from genomic datasets using Jellyfish via Docker

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GENOME_DIR="$PROJECT_ROOT/data/genomes"
KMER_DIR="$PROJECT_ROOT/data/kmers"

mkdir -p "$KMER_DIR"

K_VALUES=(21 31)

JELLYFISH_IMAGE="quay.io/biocontainers/jellyfish:2.2.6--0"

echo "Using Docker image: $JELLYFISH_IMAGE"
echo "K values: ${K_VALUES[*]}"
echo ""

if ! command -v docker &>/dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi
echo "Ensuring Jellyfish Docker image is available..."
docker pull "$JELLYFISH_IMAGE"
echo ""

# Function to extract k-mers for a given genome and k value
extract_kmers() {
    local genome_file=$1
    local genome_name=$2
    local k=$3

    local output_prefix="$KMER_DIR/${genome_name}_${k}"
    local jf_file="${output_prefix}.jf"
    local txt_file="${output_prefix}.txt"

    echo "Processing: $genome_name (k=$k)"

    if [ -f "$txt_file" ]; then
        echo "K-mer file already exists, skipping"
        return
    fi

    local num_threads
    num_threads=$(nproc 2>/dev/null || echo 4)

    echo "Counting k-mers (using $num_threads threads)..."
    docker run --rm \
        -v "$GENOME_DIR:/genomes:ro" \
        -v "$KMER_DIR:/output" \
        "$JELLYFISH_IMAGE" \
        jellyfish count \
        -m "$k" \
        -s 1G \
        -t "$num_threads" \
        -C \
        -o "/output/$(basename "$jf_file")" \
        "/genomes/$(basename "$genome_file")"

    echo "Dumping k-mers to text..."
    docker run --rm \
        -v "$KMER_DIR:/output" \
        "$JELLYFISH_IMAGE" \
        jellyfish dump \
        -c \
        -t \
        "/output/$(basename "$jf_file")" \
        >"$txt_file"

    rm -f "$jf_file"

    local count
    count=$(wc -l <"$txt_file")
    echo "Extracted $count unique k-mers"
    echo ""
}

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

echo "Extraction complete"
echo "Next step: Run ./scripts/convert_kmers_to_binary.py to convert to binary format"
