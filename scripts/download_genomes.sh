#!/usr/bin/env bash


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DATA_ROOT="${1:-$PROJECT_ROOT/data}"
GENOME_DIR="$DATA_ROOT/genomes"

mkdir -p "$GENOME_DIR"

echo "=== Downloading Genomic Datasets ==="
echo "Output directory: $GENOME_DIR"
echo ""

# E. coli K-12 MG1655 complete genome
echo "[1/6] Downloading E. coli K-12 MG1655 genome..."
if [ -f "$GENOME_DIR/ecoli_k12_mg1655.fna" ]; then
    echo "Already exists, skipping"
else
    wget -O "$GENOME_DIR/ecoli_k12_mg1655.fna.gz" \
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz"
    gunzip "$GENOME_DIR/ecoli_k12_mg1655.fna.gz"
    echo "Downloaded and extracted"
fi

# Human chromosome 14
echo "[2/6] Downloading Human chromosome 14..."
if [ -f "$GENOME_DIR/chr14.fna" ]; then
    echo "Already exists, skipping"
else
    wget -O "$GENOME_DIR/chr14.fna.gz" \
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_assembly_structure/Primary_Assembly/assembled_chromosomes/FASTA/chr14.fna.gz"
    gunzip "$GENOME_DIR/chr14.fna.gz"
    echo "Downloaded and extracted"
fi

# Rice Genome (Oryza sativa)
echo "[3/6] Downloading Rice Genome (Oryza sativa)..."
if [ -f "$GENOME_DIR/rice.fna" ]; then
    echo "Already exists, skipping"
else
    wget -O "$GENOME_DIR/rice.fna.gz" \
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/001/433/935/GCF_001433935.1_IRGSP-1.0/GCF_001433935.1_IRGSP-1.0_genomic.fna.gz"
    gunzip "$GENOME_DIR/rice.fna.gz"
    echo "Downloaded and extracted"
fi

# Chicken Genome (Gallus gallus)
echo "[4/6] Downloading Chicken Genome (Gallus gallus)..."
if [ -f "$GENOME_DIR/chicken.fna" ]; then
    echo "Already exists, skipping"
else
    wget -O "$GENOME_DIR/chicken.fna.gz" \
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/016/699/485/GCF_016699485.2_bGalGal1.mat.broiler.GRCg7b/GCF_016699485.2_bGalGal1.mat.broiler.GRCg7b_genomic.fna.gz"
    gunzip "$GENOME_DIR/chicken.fna.gz"
    echo "Downloaded and extracted"
fi

# Full Human Genome
echo "[5/6] Downloading Full Human Genome..."
if [ -f "$GENOME_DIR/human_grch38.fna" ]; then
    echo "Already exists, skipping"
else
    wget -O "$GENOME_DIR/human_grch38.fna.gz" \
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.fna.gz"
    gunzip "$GENOME_DIR/human_grch38.fna.gz"
    echo "Downloaded and extracted"
fi

# Wheat Genome (Triticum aestivum)
echo "[6/6] Downloading Wheat Genome (Triticum aestivum)..."
if [ -f "$GENOME_DIR/wheat.fna" ]; then
    echo "Already exists, skipping"
else
    wget -O "$GENOME_DIR/wheat.fna.gz" \
        "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-59/fasta/triticum_aestivum/dna/Triticum_aestivum.IWGSC.dna.toplevel.fa.gz"
    gunzip "$GENOME_DIR/wheat.fna.gz"
    echo "Downloaded and extracted"
fi

echo ""
echo "All genomes downloaded successfully"
echo "Next step: Run ./scripts/extract_kmers.sh to extract k-mers"
