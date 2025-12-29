# GPU-Accelerated Cuckoo Filter

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://cuckoo-filter-5b9470.pages.gitlab.rlp.net/)

A high-performance CUDA implementation of the Cuckoo Filter data structure, developed as part of the thesis "Design and Evaluation of a GPU-Accelerated Cuckoo Filter".

## Overview

This library provides a GPU-accelerated Cuckoo Filter implementation optimized for high-throughput batch operations. Cuckoo Filters are space-efficient probabilistic data structures that support insertion, lookup, and deletion operations with a configurable false positive rate.

## Features

-   CUDA-accelerated batch insert, lookup, and delete operations
-   Configurable fingerprint size and bucket size
-   Multiple eviction policies (DFS, BFS)
-   Sorted insertion mode for improved memory coalescing
-   Multi-GPU support via [gossip](https://github.com/Funatiq/gossip)
-   IPC support for cross-process filter sharing
-   Header-only library design

## Requirements

-   CUDA Toolkit
-   C++20 compatible compiler
-   Meson build system (>= 1.3.0)

## Building

```bash
meson setup build
meson compile -C build
```

Benchmarks and tests are built by default. To disable them:

```bash
meson setup build -DBUILD_BENCHMARKS=false -DBUILD_TESTS=false
```

## Usage

```cpp
#include <CuckooFilter.cuh>

// Configure the filter: key type, fingerprint bits, max evictions, block size, bucket size
using Config = CuckooConfig<uint64_t, 16, 500, 256, 16>;

// Create a filter with the desired capacity
CuckooFilter<Config> filter(1 << 20);  // capacity for ~1M items

// Insert keys (d_keys is a device pointer)
filter.insertMany(d_keys, numkeys);

// Or use sorted insertion
filter.insertManySorted(d_keys, numkeys);

// Check membership
filter.containsMany(d_keys, d_results, numkeys);

// Delete keys
filter.deleteMany(d_keys, d_results, numkeys);
```

### Configuration Options

The `CuckooConfig` template accepts the following parameters:

| Parameter         | Description                              | Default              |
| ----------------- | ---------------------------------------- | -------------------- |
| `T`               | Key type                                 | -                    |
| `bitsPerTag`      | Fingerprint size in bits (8, 16, 32)     | -                    |
| `maxEvictions`    | Maximum eviction attempts before failure | 500                  |
| `blockSize`       | CUDA block size                          | 256                  |
| `bucketSize`      | Slots per bucket (must be power of 2)    | 16                   |
| `AltBucketPolicy` | Alternate bucket calculation policy      | `XorAltBucketPolicy` |
| `evictionPolicy`  | Eviction strategy (DFS or BFS)           | `BFS`                |
| `WordType`        | Atomic type (uint32_t or uint64_t)       | `uint64_t`           |

## Multi-GPU Support

For workloads that exceed single GPU capacity:

```cpp
#include <CuckooFilterMultiGPU.cuh>

CuckooFilterMultiGPU<Config> filter(numGPUs, capacityPerGPU);
filter.insertMany(d_keys, numKeys);
filter.containsMany(d_keys, d_results, numKeys);
```

## Project Structure

```
include/           - Header files
  CuckooFilter.cuh           - Main filter implementation
  CuckooFilterMultiGPU.cuh   - Multi-GPU implementation
  CuckooFilterIPC.cuh        - IPC support
  bucket_policies.cuh        - Alternative bucket policies
  helpers.cuh                - Helper functions
src/               - Example applications
benchmark/         - benchmarks
tests/             - Unit tests
scripts/           - Scripts for running/plotting benchmarks
```
