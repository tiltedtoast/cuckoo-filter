#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <bulk_tcf_host.cuh>
#include <cstdint>
#include <CuckooFilter.cuh>
#include <cuda/std/cstdint>
#include <bucket_policies.cuh>
#include <helpers.cuh>
#include <random>
#include "benchmark_common.cuh"

namespace bm = benchmark;

using Config = CuckooConfig<uint64_t, 16, 500, 128, 16, XorAltBucketPolicy>;
using TCFType = host_bulk_tcf<uint64_t, uint16_t>;
constexpr double TCF_LOAD_FACTOR = 0.85;

class TCFFixture : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    void SetUp(const benchmark::State& state) override {
        auto [cap, num] = calculateCapacityAndSize(state.range(0), TCF_LOAD_FACTOR);
        capacity = cap;
        n = num;

        d_keys.resize(n);
        generateKeysGPU(d_keys);

        cudaMalloc(&d_misses, sizeof(uint64_t));
        filterMemory = capacity * sizeof(uint16_t);
    }

    void TearDown(const benchmark::State&) override {
        if (d_misses) {
            cudaFree(d_misses);
            d_misses = nullptr;
        }
        d_keys.clear();
        d_keys.shrink_to_fit();
    }

    void setCounters(benchmark::State& state) const {
        setCommonCounters(state, filterMemory, n);
    }

    size_t capacity;
    size_t n;
    size_t filterMemory;
    uint64_t* d_misses = nullptr;
    thrust::device_vector<uint64_t> d_keys;
    GPUTimer timer;
};

using CFFixture = CuckooFilterFixture<Config, TCF_LOAD_FACTOR>;

static void CF_FPR(bm::State& state) {
    GPUTimer timer;
    auto [capacity, n] = calculateCapacityAndSize(state.range(0), TCF_LOAD_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);

    auto filter = std::make_unique<CuckooFilter<Config>>(capacity);
    size_t filterMemory = filter->sizeInBytes();
    adaptiveInsert(*filter, d_keys);

    size_t fprTestSize = std::min(n, size_t(1'000'000));
    thrust::device_vector<uint64_t> d_neverInserted(fprTestSize);
    thrust::device_vector<uint8_t> d_output(fprTestSize);

    generateKeysGPURange(
        d_neverInserted, fprTestSize, static_cast<uint64_t>(UINT32_MAX) + 1, UINT64_MAX
    );

    for (auto _ : state) {
        timer.start();
        filter->containsMany(d_neverInserted, d_output);
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
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

BENCHMARK_DEFINE_F(TCFFixture, Insert)(bm::State& state) {
    for (auto _ : state) {
        TCFType* filter = TCFType::host_build_tcf(capacity);
        cudaMemset(d_misses, 0, sizeof(uint64_t));
        cudaDeviceSynchronize();

        timer.start();
        filter->bulk_insert(thrust::raw_pointer_cast(d_keys.data()), n, d_misses);
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        TCFType::host_free_tcf(filter);
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(TCFFixture, Query)(bm::State& state) {
    TCFType* filter = TCFType::host_build_tcf(capacity);
    cudaMemset(d_misses, 0, sizeof(uint64_t));
    filter->bulk_insert(thrust::raw_pointer_cast(d_keys.data()), n, d_misses);
    cudaDeviceSynchronize();

    for (auto _ : state) {
        timer.start();
        bool* d_output = filter->bulk_query(thrust::raw_pointer_cast(d_keys.data()), n);
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output);
        cudaFree(d_output);
    }

    TCFType::host_free_tcf(filter);
    setCounters(state);
}

BENCHMARK_DEFINE_F(TCFFixture, Delete)(bm::State& state) {
    for (auto _ : state) {
        TCFType* filter = TCFType::host_build_tcf(capacity);
        cudaMemset(d_misses, 0, sizeof(uint64_t));
        filter->bulk_insert(thrust::raw_pointer_cast(d_keys.data()), n, d_misses);
        cudaDeviceSynchronize();

        timer.start();
        bool* d_output = filter->bulk_delete(thrust::raw_pointer_cast(d_keys.data()), n);
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output);

        cudaFree(d_output);
        TCFType::host_free_tcf(filter);
    }
    setCounters(state);
}

static void TCF_FPR(bm::State& state) {
    GPUTimer timer;
    auto [capacity, n] = calculateCapacityAndSize(state.range(0), TCF_LOAD_FACTOR);
    size_t filterMemory = capacity * sizeof(uint16_t);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU<uint64_t>(d_keys, UINT32_MAX);

    uint64_t* d_misses;
    cudaMalloc(&d_misses, sizeof(uint64_t));

    TCFType* filter = TCFType::host_build_tcf(capacity);
    cudaMemset(d_misses, 0, sizeof(uint64_t));
    filter->bulk_insert(thrust::raw_pointer_cast(d_keys.data()), n, d_misses);

    size_t fprTestSize = std::min(n, size_t(1'000'000));
    thrust::device_vector<uint64_t> d_neverInserted(fprTestSize);

    generateKeysGPURange(
        d_neverInserted, fprTestSize, static_cast<uint64_t>(UINT32_MAX) + 1, UINT64_MAX
    );

    for (auto _ : state) {
        timer.start();
        bool* d_output =
            filter->bulk_query(thrust::raw_pointer_cast(d_neverInserted.data()), fprTestSize);
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);

        thrust::device_ptr<bool> d_outputPtr(d_output);
        size_t falsePositives =
            thrust::reduce(d_outputPtr, d_outputPtr + fprTestSize, 0ULL, cuda::std::plus<size_t>());
        cudaFree(d_output);

        bm::DoNotOptimize(falsePositives);
    }

    bool* d_output =
        filter->bulk_query(thrust::raw_pointer_cast(d_neverInserted.data()), fprTestSize);
    cudaDeviceSynchronize();
    thrust::device_ptr<bool> d_outputPtr(d_output);
    size_t falsePositives =
        thrust::reduce(d_outputPtr, d_outputPtr + fprTestSize, 0ULL, cuda::std::plus<size_t>());
    cudaFree(d_output);

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

    TCFType::host_free_tcf(filter);
    cudaFree(d_misses);
}

DEFINE_AND_REGISTER_CORE_BENCHMARKS(CFFixture);
REGISTER_CORE_BENCHMARKS(TCFFixture);

REGISTER_FUNCTION_BENCHMARK(CF_FPR);
REGISTER_FUNCTION_BENCHMARK(TCF_FPR);

STANDARD_BENCHMARK_MAIN();
