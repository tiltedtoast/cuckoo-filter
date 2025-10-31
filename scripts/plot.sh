#!/bin/sh

set -xe

./scripts/plot_memory.py < "$1"
./scripts/plot_runtime.py < "$1"
./scripts/plot_throughput.py < "$1"
./scripts/plot_fpr.py < "$1"
./scripts/plot_bucket_size.py < "$1"