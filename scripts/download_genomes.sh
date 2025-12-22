#!/usr/bin/env bash
# Download genomic datasets for k-mer benchmarking

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GENOME_DIR="$PROJECT_ROOT/data/genomes"

mkdir -p "$GENOME_DIR"

echo "=== Downloading Genomic Datasets ==="
echo "Output directory: $GENOME_DIR"
echo ""

# E. coli K-12 MG1655 complete genome
echo "[1/3] Downloading E. coli K-12 MG1655 genome..."
if [ -f "$GENOME_DIR/ecoli_k12_mg1655.fna" ]; then
    echo "Already exists, skipping"
else
    wget -O "$GENOME_DIR/ecoli_k12_mg1655.fna.gz" \
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz"
    gunzip "$GENOME_DIR/ecoli_k12_mg1655.fna.gz"
    echo "Downloaded and extracted"
fi

# Human chromosome 14
echo "[2/3] Downloading Human chromosome 14..."
if [ -f "$GENOME_DIR/chr14.fna" ]; then
    echo "Already exists, skipping"
else
    # Using GRCh38 reference
    wget -O "$GENOME_DIR/chr14.fna.gz" \
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_assembly_structure/Primary_Assembly/assembled_chromosomes/FASTA/chr14.fna.gz"
    gunzip "$GENOME_DIR/chr14.fna.gz"
    echo "Downloaded and extracted"
fi

# Full Human Genome (for large-scale benchmarking)
echo "[3/3] Downloading Full Human Genome..."
echo "  This will generate ~8-16 GB of k-mer data (k=31)"
if [ -f "$GENOME_DIR/human_grch38.fna" ]; then
    echo "Already exists, skipping"
else
    wget -O "$GENOME_DIR/human_grch38.fna.gz" \
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.fna.gz"
    gunzip "$GENOME_DIR/human_grch38.fna.gz"
    echo "Downloaded and extracted"
fi

echo ""
echo "All genomes downloaded successfully"
echo "Next step: Run ./scripts/extract_kmers.sh to extract k-mers"
