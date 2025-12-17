#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <cstdint>
#include <cub/cub.cuh>
#include <CuckooFilter.cuh>
#include <helpers.cuh>
#include "benchmark_common.cuh"

namespace bm = benchmark;

using Config = CuckooConfig<uint64_t, 16, 500, 256, 16, XorAltBucketPolicy>;
using Filter = CuckooFilter<Config>;
using PackedTagType = typename Filter::PackedTagType;

using CF = CuckooFilterFixture<Config>;

BENCHMARK_DEFINE_F(CF, InsertUnsorted)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        cudaDeviceSynchronize();

        timer.start();
        size_t inserted = filter->insertMany(d_keys);
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(CF, InsertSorted)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        cudaDeviceSynchronize();

        timer.start();
        size_t inserted = filter->insertManySorted(d_keys);
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(CF, InsertPresorted)(bm::State& state) {
    PackedTagType* d_packedTags;
    CUDA_CALL(cudaMalloc(&d_packedTags, n * sizeof(PackedTagType)));

    size_t numBlocks = SDIV(n, Config::blockSize);
    computePackedTagsKernel<Config><<<numBlocks, Config::blockSize>>>(
        thrust::raw_pointer_cast(d_keys.data()), d_packedTags, n, filter->numBuckets
    );

    void* d_tempStorage = nullptr;
    size_t tempStorageBytes = 0;

    cub::DeviceRadixSort::SortKeys(d_tempStorage, tempStorageBytes, d_packedTags, d_packedTags, n);

    CUDA_CALL(cudaMalloc(&d_tempStorage, tempStorageBytes));

    cub::DeviceRadixSort::SortKeys(d_tempStorage, tempStorageBytes, d_packedTags, d_packedTags, n);

    CUDA_CALL(cudaFree(d_tempStorage));
    cudaDeviceSynchronize();

    for (auto _ : state) {
        filter->clear();
        cudaDeviceSynchronize();

        timer.start();
        insertKernelSorted<Config><<<numBlocks, Config::blockSize>>>(
            thrust::raw_pointer_cast(d_keys.data()), d_packedTags, n, filter.get()
        );
        cudaDeviceSynchronize();
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(filter->occupiedSlots());
    }

    CUDA_CALL(cudaFree(d_packedTags));
    setCounters(state);
}

#define SORTED_UNSORTED_BENCHMARK_CONFIG \
    ->RangeMultiplier(2)                 \
        ->Range(1 << 16, 1ULL << 28)     \
        ->Unit(benchmark::kMillisecond)  \
        ->UseManualTime()                \
        ->Iterations(20)                 \
        ->Repetitions(10)                \
        ->ReportAggregatesOnly(true)

BENCHMARK_REGISTER_F(CF, InsertUnsorted)
SORTED_UNSORTED_BENCHMARK_CONFIG;

BENCHMARK_REGISTER_F(CF, InsertSorted)
SORTED_UNSORTED_BENCHMARK_CONFIG;

BENCHMARK_REGISTER_F(CF, InsertPresorted)
SORTED_UNSORTED_BENCHMARK_CONFIG;

STANDARD_BENCHMARK_MAIN();
