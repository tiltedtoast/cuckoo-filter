#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cstdint>
#include <CuckooFilter.cuh>
#include <CuckooFilterMultiGPU.cuh>
#include <helpers.cuh>
#include "benchmark_common.cuh"

namespace bm = benchmark;

using Config = CuckooConfig<uint64_t, 16, 500, 256, 16>;

static constexpr size_t CAPACITY_PER_GPU = 1ULL << 30;
static constexpr double LOAD_FACTOR = 0.95;

template <typename ConfigType>
class WeakScalingFixture : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    using KeyType = typename ConfigType::KeyType;

    void SetUp(const benchmark::State&) override {
        CUDA_CALL(cudaGetDeviceCount(&deviceCount));
        numGPUs = static_cast<size_t>(deviceCount);

        // Weak scaling: total capacity grows with GPU count
        totalCapacity = numGPUs * CAPACITY_PER_GPU;
        n = static_cast<size_t>(totalCapacity * LOAD_FACTOR);

        h_keys.resize(n);
        thrust::device_vector<KeyType> d_temp(n);
        generateKeysGPU(d_temp);
        thrust::copy(d_temp.begin(), d_temp.end(), h_keys.begin());

        h_output.resize(n);

        filter = std::make_unique<CuckooFilterMultiGPU<ConfigType>>(numGPUs, totalCapacity);
        filterMemory = filter->sizeInBytes();
    }

    void TearDown(const benchmark::State&) override {
        filter.reset();
        h_keys.clear();
        h_output.clear();
        h_keys.shrink_to_fit();
        h_output.shrink_to_fit();
    }

    void setCounters(benchmark::State& state) {
        setCommonCounters(state, filterMemory, n);
        state.counters["gpus"] = static_cast<double>(numGPUs);
        state.counters["scaling_mode"] = 0;  // 0 = weak
        state.counters["capacity_per_gpu"] =
            static_cast<double>(static_cast<double>(totalCapacity) / numGPUs);
        state.counters["total_capacity"] = static_cast<double>(totalCapacity);
    }

    int deviceCount;
    size_t numGPUs;
    size_t totalCapacity;
    size_t n;
    size_t filterMemory;
    thrust::host_vector<KeyType> h_keys;
    thrust::host_vector<bool> h_output;
    std::unique_ptr<CuckooFilterMultiGPU<ConfigType>> filter;
    CPUTimer timer;
};

template <typename ConfigType>
class StrongScalingFixture : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    using KeyType = typename ConfigType::KeyType;

    void SetUp(const benchmark::State&) override {
        CUDA_CALL(cudaGetDeviceCount(&deviceCount));
        numGPUs = static_cast<size_t>(deviceCount);

        // Strong scaling: total capacity is fixed
        totalCapacity = CAPACITY_PER_GPU;
        n = static_cast<size_t>(totalCapacity * LOAD_FACTOR);

        h_keys.resize(n);
        thrust::device_vector<KeyType> d_temp(n);
        generateKeysGPU(d_temp);
        thrust::copy(d_temp.begin(), d_temp.end(), h_keys.begin());

        h_output.resize(n);

        filter = std::make_unique<CuckooFilterMultiGPU<ConfigType>>(numGPUs, totalCapacity);
        filterMemory = filter->sizeInBytes();
    }

    void TearDown(const benchmark::State&) override {
        filter.reset();
        h_keys.clear();
        h_output.clear();
        h_keys.shrink_to_fit();
        h_output.shrink_to_fit();
    }

    void setCounters(benchmark::State& state) {
        setCommonCounters(state, filterMemory, n);
        state.counters["gpus"] = static_cast<double>(numGPUs);
        state.counters["scaling_mode"] = 1;  // 1 = strong
        state.counters["capacity_per_gpu"] =
            static_cast<double>(static_cast<double>(totalCapacity) / numGPUs);
        state.counters["total_capacity"] = static_cast<double>(totalCapacity);
    }

    int deviceCount;
    size_t numGPUs;
    size_t totalCapacity;
    size_t n;
    size_t filterMemory;
    thrust::host_vector<KeyType> h_keys;
    thrust::host_vector<bool> h_output;
    std::unique_ptr<CuckooFilterMultiGPU<ConfigType>> filter;
    CPUTimer timer;
};

using WeakFixture = WeakScalingFixture<Config>;
using StrongFixture = StrongScalingFixture<Config>;

BENCHMARK_DEFINE_F(WeakFixture, Insert)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        filter->synchronizeAllGPUs();

        timer.start();
        size_t inserted = filter->insertMany(h_keys);
        filter->synchronizeAllGPUs();
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(WeakFixture, Query)(bm::State& state) {
    filter->insertMany(h_keys);
    filter->synchronizeAllGPUs();

    for (auto _ : state) {
        timer.start();
        filter->containsMany(h_keys, h_output);
        filter->synchronizeAllGPUs();
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(h_output.data());
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(WeakFixture, Delete)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        filter->insertMany(h_keys);
        filter->synchronizeAllGPUs();

        timer.start();
        size_t remaining = filter->deleteMany(h_keys, h_output);
        filter->synchronizeAllGPUs();
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(remaining);
        bm::DoNotOptimize(h_output.data());
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(StrongFixture, Insert)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        filter->synchronizeAllGPUs();

        timer.start();
        size_t inserted = filter->insertMany(h_keys);
        filter->synchronizeAllGPUs();
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(StrongFixture, Query)(bm::State& state) {
    filter->insertMany(h_keys);
    filter->synchronizeAllGPUs();

    for (auto _ : state) {
        timer.start();
        filter->containsMany(h_keys, h_output);
        filter->synchronizeAllGPUs();
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(h_output.data());
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(StrongFixture, Delete)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        filter->insertMany(h_keys);
        filter->synchronizeAllGPUs();

        timer.start();
        size_t remaining = filter->deleteMany(h_keys, h_output);
        filter->synchronizeAllGPUs();
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(remaining);
        bm::DoNotOptimize(h_output.data());
    }
    setCounters(state);
}

#define REGISTER_SCALING_BM(fixture, name) \
    BENCHMARK_REGISTER_F(fixture, name)    \
        ->Unit(benchmark::kMillisecond)    \
        ->UseManualTime()                  \
        ->Iterations(5)                    \
        ->Repetitions(3)                   \
        ->ReportAggregatesOnly(true);

REGISTER_SCALING_BM(WeakFixture, Insert);
REGISTER_SCALING_BM(WeakFixture, Query);
REGISTER_SCALING_BM(WeakFixture, Delete);

REGISTER_SCALING_BM(StrongFixture, Insert);
REGISTER_SCALING_BM(StrongFixture, Query);
REGISTER_SCALING_BM(StrongFixture, Delete);

STANDARD_BENCHMARK_MAIN();
