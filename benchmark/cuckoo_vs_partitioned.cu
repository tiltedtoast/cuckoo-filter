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
using Config = CuckooConfig<uint64_t, 16, 500, 128, 16>;
const size_t L2_CACHE_SIZE = getL2CacheSize();

using CPUFilterParam = filters::cuckoo::Standard4<Config::bitsPerTag>;
using CPUOptimParam = filters::parameter::PowerOfTwoMurmurScalar64PartitionedMT;
using PartitionedCuckooFilter =
    filters::Filter<filters::FilterType::Cuckoo, CPUFilterParam, Config::bitsPerTag, CPUOptimParam>;

using GPUCFFixture = CuckooFilterFixture<Config>;

class PartitionedCFFixture : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    static constexpr double TARGET_LOAD_FACTOR = 0.95;

    void SetUp(const benchmark::State& state) override {
        auto [cap, num] = calculateCapacityAndSize(state.range(0), TARGET_LOAD_FACTOR);
        capacity = cap;
        n = num;

        keys = generateKeysCPU<uint64_t>(n);

        s = static_cast<size_t>(100.0 / TARGET_LOAD_FACTOR);

        size_t expectedSize =
            (capacity * Config::bitsPerTag * (static_cast<double>(s) / 100)) / 8.0;
        n_partitions = 1;

        if (expectedSize > L2_CACHE_SIZE) {
            size_t partitionsNeeded = SDIV(expectedSize, L2_CACHE_SIZE);
            while (n_partitions < partitionsNeeded) {
                n_partitions *= 2;
            }
        }

        n_threads = 8;
        n_tasks = 1;
    }

    void TearDown(const benchmark::State&) override {
        keys.clear();
    }

    void setCounters(benchmark::State& state, size_t filterMemory) const {
        setCommonCounters(state, filterMemory, n);
    }

    size_t capacity;
    size_t n;
    size_t s;
    size_t n_partitions;
    size_t n_threads;
    size_t n_tasks;
    std::vector<uint64_t> keys;
    Timer timer;
};

BENCHMARK_DEFINE_F(GPUCFFixture, Insert)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        cudaDeviceSynchronize();

        timer.start();
        size_t inserted = adaptiveInsert(*filter, d_keys);
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(GPUCFFixture, Query)(bm::State& state) {
    adaptiveInsert(*filter, d_keys);

    for (auto _ : state) {
        timer.start();
        filter->containsMany(d_keys, d_output);
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output.data().get());
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(GPUCFFixture, InsertAndQuery)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        cudaDeviceSynchronize();

        timer.start();
        size_t inserted = adaptiveInsert(*filter, d_keys);
        filter->containsMany(d_keys, d_output);
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);
        bm::DoNotOptimize(d_output.data().get());
    }
    setCounters(state);
}

static void GPUCF_FPR(bm::State& state) {
    Timer timer;
    auto [capacity, n] = calculateCapacityAndSize(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);

    auto filter = std::make_unique<CuckooFilter<Config>>(capacity);
    size_t filterMemory = filter->sizeInBytes();
    adaptiveInsert(*filter, d_keys);

    size_t fprTestSize = std::min(n, size_t(1'000'000));
    thrust::device_vector<uint64_t> d_neverInserted(fprTestSize);
    thrust::device_vector<uint8_t> d_output(fprTestSize);

    generateKeysGPURange(
        d_neverInserted,
        fprTestSize,
        static_cast<uint64_t>(UINT32_MAX + 1),
        static_cast<uint64_t>(UINT64_MAX)
    );

    for (auto _ : state) {
        timer.start();
        filter->containsMany(d_neverInserted, d_output);
        double elapsed = timer.stop();

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

BENCHMARK_DEFINE_F(PartitionedCFFixture, Insert)(bm::State& state) {
    for (auto _ : state) {
        PartitionedCuckooFilter tempFilter(s, n_partitions, n_threads, n_tasks);
        auto constructKeys = keys;

        timer.start();
        bool success = tempFilter.construct(constructKeys.data(), constructKeys.size());
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(success);
    }

    PartitionedCuckooFilter finalFilter(s, n_partitions, n_threads, n_tasks);
    finalFilter.construct(keys.data(), keys.size());
    size_t filterMemory = finalFilter.size();

    setCounters(state, filterMemory);
}

BENCHMARK_DEFINE_F(PartitionedCFFixture, Query)(bm::State& state) {
    PartitionedCuckooFilter filter(s, n_partitions, n_threads, n_tasks);
    filter.construct(keys.data(), keys.size());

    size_t filterMemory = filter.size();

    for (auto _ : state) {
        timer.start();
        size_t found = filter.count(keys.data(), keys.size());
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(found);
    }

    setCounters(state, filterMemory);
}

BENCHMARK_DEFINE_F(PartitionedCFFixture, InsertAndQuery)(bm::State& state) {
    for (auto _ : state) {
        PartitionedCuckooFilter filter(s, n_partitions, n_threads, n_tasks);

        timer.start();
        bool success = filter.construct(keys.data(), keys.size());
        size_t found = filter.count(keys.data(), keys.size());
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(success);
        bm::DoNotOptimize(found);
    }

    PartitionedCuckooFilter finalFilter(s, n_partitions, n_threads, n_tasks);
    finalFilter.construct(keys.data(), keys.size());
    size_t filterMemory = finalFilter.size();

    setCounters(state, filterMemory);
}

static void PartitionedCF_FPR(bm::State& state) {
    Timer timer;
    auto [capacity, n] = calculateCapacityAndSize(state.range(0), TARGET_LOAD_FACTOR);

    auto keys = generateKeysCPU<uint64_t>(n);

    auto s = static_cast<size_t>(100.0 / TARGET_LOAD_FACTOR);
    size_t expectedSize = (capacity * Config::bitsPerTag * (static_cast<double>(s) / 100)) / 8.0;
    size_t n_partitions = 1;

    if (expectedSize > L2_CACHE_SIZE) {
        size_t partitionsNeeded = SDIV(expectedSize, L2_CACHE_SIZE);
        while (n_partitions < partitionsNeeded) {
            n_partitions *= 2;
        }
    }

    size_t n_threads = 8;
    size_t n_tasks = 1;

    PartitionedCuckooFilter filter(s, n_partitions, n_threads, n_tasks);
    filter.construct(keys.data(), keys.size());

    size_t fprTestSize = std::min(n, size_t(1'000'000));
    auto neverInserted = generateKeysCPU<uint64_t>(fprTestSize, 99999, UINT32_MAX + 1, UINT64_MAX);

    size_t falsePositives = 0;
    for (auto _ : state) {
        timer.start();
        falsePositives = 0;
        for (const auto& k : neverInserted) {
            if (filter.contains(k)) {
                ++falsePositives;
            }
        }
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
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

REGISTER_BENCHMARK(GPUCFFixture, Insert);
REGISTER_BENCHMARK(PartitionedCFFixture, Insert);

REGISTER_BENCHMARK(GPUCFFixture, Query);
REGISTER_BENCHMARK(PartitionedCFFixture, Query);

REGISTER_BENCHMARK(GPUCFFixture, InsertAndQuery);
REGISTER_BENCHMARK(PartitionedCFFixture, InsertAndQuery);

REGISTER_FUNCTION_BENCHMARK(GPUCF_FPR);
REGISTER_FUNCTION_BENCHMARK(PartitionedCF_FPR);

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }

    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    fflush(stdout);
    std::cout << std::flush;

    std::_Exit(0);
}
