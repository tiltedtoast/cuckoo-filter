#include <benchmark/benchmark.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuckoo/cuckoo_parameter.hpp>
#include <filter.hpp>

#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <CuckooFilter.cuh>
#include <helpers.cuh>
#include "benchmark_common.cuh"
#include "parameter/parameter.hpp"

namespace bm = benchmark;

constexpr double TARGET_LOAD_FACTOR = 0.95;
using GPUConfig = CuckooConfig<uint32_t, 16, 500, 128, 16>;
const size_t L2_CACHE_SIZE = getL2CacheSize();

using CPUFilterParam = filters::cuckoo::Standard4<GPUConfig::bitsPerTag>;

using CPUOptimParam = filters::parameter::PowerOfTwoMurmurScalar64PartitionedMT;

using PartitionedCuckooFilter = filters::
    Filter<filters::FilterType::Cuckoo, CPUFilterParam, GPUConfig::bitsPerTag, CPUOptimParam>;

static void GPU_CuckooFilter_Insert(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<GPUConfig>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);
    CuckooFilter<GPUConfig> filter(capacity);

    size_t filterMemory = filter.sizeInBytes();

    for (auto _ : state) {
        state.PauseTiming();
        filter.clear();
        state.ResumeTiming();

        size_t inserted = filter.insertMany(d_keys);
        cudaDeviceSynchronize();
        bm::DoNotOptimize(inserted);
    }

    setCommonCounters(state, filterMemory, n);
}

static void GPU_CuckooFilter_Query(bm::State& state) {
    const size_t n = state.range(0) * TARGET_LOAD_FACTOR;

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);
    CuckooFilter<GPUConfig> filter(n);
    thrust::device_vector<uint8_t> d_output(n);

    filter.insertMany(d_keys);

    size_t filterMemory = filter.sizeInBytes();
    size_t capacity = filter.capacity();

    for (auto _ : state) {
        filter.containsMany(d_keys, d_output);
        cudaDeviceSynchronize();
        bm::DoNotOptimize(d_output.data().get());
    }

    setCommonCounters(state, filterMemory, n);
}

static void GPU_CuckooFilter_InsertAndQuery(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<GPUConfig>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);
    thrust::device_vector<uint8_t> d_output(n);
    CuckooFilter<GPUConfig> filter(capacity);

    size_t filterMemory = filter.sizeInBytes();

    for (auto _ : state) {
        state.PauseTiming();
        filter.clear();
        state.ResumeTiming();

        size_t inserted = filter.insertMany(d_keys);
        filter.containsMany(d_keys, d_output);

        cudaDeviceSynchronize();

        bm::DoNotOptimize(inserted);
        bm::DoNotOptimize(d_output.data().get());
    }

    setCommonCounters(state, filterMemory, n);
}

static void GPU_CuckooFilter_FalsePositiveRate(bm::State& state) {
    using FPRConfig = CuckooConfig<uint32_t, 16, 500, 128, 4>;

    auto [capacity, n] = calculateCapacityAndSize<FPRConfig>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU<uint32_t>(d_keys, UINT32_MAX);

    CuckooFilter<FPRConfig> filter(capacity);
    filter.insertMany(d_keys);

    size_t fprTestSize = (n < size_t(1'000'000)) ? n : size_t(1'000'000);
    thrust::device_vector<uint32_t> d_neverInserted(fprTestSize);
    thrust::device_vector<uint8_t> d_output(fprTestSize);

    thrust::transform(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(fprTestSize),
        d_neverInserted.begin(),
        [] __device__(size_t idx) {
            thrust::default_random_engine rng(99999);
            thrust::uniform_int_distribution<uint32_t> dist(
                static_cast<uint32_t>(UINT32_MAX) + 1, UINT64_MAX
            );
            rng.discard(idx);
            return dist(rng);
        }
    );

    size_t filterMemory = filter.sizeInBytes();

    for (auto _ : state) {
        filter.containsMany(d_neverInserted, d_output);
        cudaDeviceSynchronize();
        bm::DoNotOptimize(d_output.data().get());
    }

    size_t falsePositives =
        thrust::reduce(d_output.begin(), d_output.end(), 0ULL, cuda::std::plus<size_t>());
    double fpr = static_cast<double>(falsePositives) / static_cast<double>(fprTestSize);

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * fprTestSize));
    state.counters["fpr_percentage"] = bm::Counter(fpr * 100);
    state.counters["false_positives"] = bm::Counter(static_cast<double>(falsePositives));
    state.counters["bits_per_item"] = bm::Counter(
        static_cast<double>(filterMemory * 8) / static_cast<double>(n),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
    );
}

static void PartitionedCPU_CuckooFilter_Insert(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<GPUConfig>(state.range(0), TARGET_LOAD_FACTOR);
    auto keys = generateKeysCPU<uint64_t>(n);

    auto s = static_cast<size_t>(100.0 / TARGET_LOAD_FACTOR);

    size_t expectedSize = (capacity * GPUConfig::bitsPerTag * (static_cast<double>(s) / 100)) / 8.0;
    size_t n_partitions = 1;

    if (expectedSize > L2_CACHE_SIZE) {
        size_t partitionsNeeded = SDIV(expectedSize, L2_CACHE_SIZE);
        while (n_partitions < partitionsNeeded) {
            n_partitions *= 2;
        }
    }

    size_t n_threads = 8;
    size_t n_tasks = 1;

    for (auto _ : state) {
        state.PauseTiming();
        PartitionedCuckooFilter tempFilter(s, 1, n_threads, n_tasks);
        auto construct_keys = keys;
        state.ResumeTiming();

        bool success = tempFilter.construct(construct_keys.data(), construct_keys.size());
        bm::DoNotOptimize(success);
    }

    PartitionedCuckooFilter finalFilter(s, 1, n_threads, n_tasks);
    finalFilter.construct(keys.data(), keys.size());
    size_t filterMemory = finalFilter.size();

    setCommonCounters(state, filterMemory, n);
}

static void PartitionedCPU_CuckooFilter_Query(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<GPUConfig>(state.range(0), TARGET_LOAD_FACTOR);

    auto keys = generateKeysCPU<uint64_t>(n);

    auto s = static_cast<size_t>(100.0 / TARGET_LOAD_FACTOR);

    size_t expectedSize = (capacity * GPUConfig::bitsPerTag * (static_cast<double>(s) / 100)) / 8.0;
    size_t n_partitions = 1;

    if (expectedSize > L2_CACHE_SIZE) {
        size_t partitionsNeeded = SDIV(expectedSize, L2_CACHE_SIZE);
        while (n_partitions < partitionsNeeded) {
            n_partitions *= 2;
        }
    }

    size_t n_threads = 8;
    size_t n_tasks = 1;

    PartitionedCuckooFilter filter(s, 1, n_threads, n_tasks);
    filter.construct(keys.data(), keys.size());

    size_t filterMemory = filter.size();

    for (auto _ : state) {
        size_t found = filter.count(keys.data(), keys.size());
        bm::DoNotOptimize(found);
    }

    setCommonCounters(state, filterMemory, n);
}

static void PartitionedCPU_CuckooFilter_InsertAndQuery(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<GPUConfig>(state.range(0), TARGET_LOAD_FACTOR);

    auto keys = generateKeysCPU<uint64_t>(n);

    auto s = static_cast<size_t>(100.0 / TARGET_LOAD_FACTOR);

    size_t expectedSize = (capacity * GPUConfig::bitsPerTag * (static_cast<double>(s) / 100)) / 8.0;
    size_t n_partitions = 1;

    if (expectedSize > L2_CACHE_SIZE) {
        size_t partitionsNeeded = SDIV(expectedSize, L2_CACHE_SIZE);
        while (n_partitions < partitionsNeeded) {
            n_partitions *= 2;
        }
    }

    size_t n_threads = 8;
    size_t n_tasks = 1;

    for (auto _ : state) {
        state.PauseTiming();
        PartitionedCuckooFilter filter(s, 1, n_threads, n_tasks);
        state.ResumeTiming();

        bool success = filter.construct(keys.data(), keys.size());
        size_t found = filter.count(keys.data(), keys.size());

        bm::DoNotOptimize(success);
        bm::DoNotOptimize(found);
    }

    PartitionedCuckooFilter finalFilter(s, 1, n_threads, n_tasks);
    finalFilter.construct(keys.data(), keys.size());
    size_t filterMemory = finalFilter.size();

    setCommonCounters(state, filterMemory, n);
}

static void PartitionedCPU_CuckooFilter_FalsePositiveRate(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<GPUConfig>(state.range(0), TARGET_LOAD_FACTOR);
    auto keys = generateKeysCPU<uint64_t>(n, 1, 1, UINT32_MAX / 2);

    auto s = static_cast<size_t>(100.0 / TARGET_LOAD_FACTOR);

    size_t expectedSize = (capacity * GPUConfig::bitsPerTag * (static_cast<double>(s) / 100)) / 8.0;
    size_t n_partitions = 1;

    if (expectedSize > L2_CACHE_SIZE) {
        size_t partitionsNeeded = SDIV(expectedSize, L2_CACHE_SIZE);
        while (n_partitions < partitionsNeeded) {
            n_partitions *= 2;
        }
    }

    size_t n_threads = 8;
    size_t n_tasks = 1;

    PartitionedCuckooFilter filter(s, 1, n_threads, n_tasks);
    filter.construct(keys.data(), keys.size());

    size_t fprTestSize = std::min(n, size_t(1'000'000));
    auto neverInserted =
        generateKeysCPU<uint64_t>(fprTestSize, 99999, UINT32_MAX / 2 + 1, UINT32_MAX);

    size_t falsePositives = 0;
    for (auto _ : state) {
        falsePositives = 0;
        for (const auto& k : neverInserted) {
            if (filter.contains(k)) {
                ++falsePositives;
            }
        }
        bm::DoNotOptimize(falsePositives);
    }

    double fpr = static_cast<double>(falsePositives) / static_cast<double>(fprTestSize);
    size_t filterMemory = filter.size();

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * fprTestSize));
    state.counters["fpr_percentage"] = bm::Counter(fpr * 100);
    state.counters["false_positives"] = bm::Counter(static_cast<double>(falsePositives));
    state.counters["bits_per_item"] = bm::Counter(
        static_cast<double>(filterMemory * 8) / static_cast<double>(n),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
    );
}

BENCHMARK(GPU_CuckooFilter_Insert)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);
BENCHMARK(PartitionedCPU_CuckooFilter_Insert)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(GPU_CuckooFilter_Query)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);
BENCHMARK(PartitionedCPU_CuckooFilter_Query)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(GPU_CuckooFilter_InsertAndQuery)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);
BENCHMARK(PartitionedCPU_CuckooFilter_InsertAndQuery)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(GPU_CuckooFilter_FalsePositiveRate)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);
BENCHMARK(PartitionedCPU_CuckooFilter_FalsePositiveRate)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
