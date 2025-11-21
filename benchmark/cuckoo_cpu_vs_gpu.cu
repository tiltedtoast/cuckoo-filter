#include <benchmark/benchmark.h>
#include <cuckoofilter.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cstdint>
#include <CuckooFilter.cuh>
#include <helpers.cuh>
#include <random>
#include "benchmark_common.cuh"

namespace bm = benchmark;

using Config = CuckooConfig<uint64_t, 16, 500, 128, 16>;
constexpr size_t CPU_BITS_PER_ITEM = Config::bitsPerTag;

using GPUCFFixture = CuckooFilterFixture<Config>;

template <typename KeyType, size_t bitsPerItem, double loadFactor = 0.95>
class CPUCuckooFilterFixture : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    void SetUp(const benchmark::State& state) override {
        auto [cap, num] = calculateCapacityAndSize(state.range(0), loadFactor);
        capacity = cap;
        n = num;

        keys = generateKeysCPU<KeyType>(n);
        filterMemory = 0;
    }

    void TearDown(const benchmark::State&) override {
        keys.clear();
        keys.shrink_to_fit();
    }

    void setCounters(benchmark::State& state) {
        setCommonCounters(state, filterMemory, n);
    }

    size_t capacity;
    size_t n;
    size_t filterMemory;
    std::vector<KeyType> keys;
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

BENCHMARK_DEFINE_F(GPUCFFixture, Delete)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        adaptiveInsert(*filter, d_keys);
        cudaDeviceSynchronize();

        timer.start();
        size_t remaining = filter->deleteMany(d_keys, d_output);
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(remaining);
        bm::DoNotOptimize(d_output.data().get());
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(GPUCFFixture, InsertQueryDelete)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        cudaDeviceSynchronize();

        timer.start();
        size_t inserted = adaptiveInsert(*filter, d_keys);
        filter->containsMany(d_keys, d_output);
        size_t remaining = filter->deleteMany(d_keys, d_output);
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);
        bm::DoNotOptimize(remaining);
        bm::DoNotOptimize(d_output.data().get());
    }
    setCounters(state);
}

using CPUCFFixture = CPUCuckooFilterFixture<uint64_t, CPU_BITS_PER_ITEM>;

BENCHMARK_DEFINE_F(CPUCFFixture, Insert)(bm::State& state) {
    for (auto _ : state) {
        cuckoofilter::CuckooFilter<uint64_t, CPU_BITS_PER_ITEM> tempFilter(capacity);

        timer.start();
        size_t inserted = 0;
        for (const auto& key : keys) {
            if (tempFilter.Add(key) == cuckoofilter::Ok) {
                inserted++;
            }
        }
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);
    }

    filterMemory = cuckoofilter::CuckooFilter<uint64_t, CPU_BITS_PER_ITEM>(capacity).SizeInBytes();
    setCounters(state);
}

BENCHMARK_DEFINE_F(CPUCFFixture, Query)(bm::State& state) {
    cuckoofilter::CuckooFilter<uint64_t, CPU_BITS_PER_ITEM> filter(capacity);
    for (const auto& key : keys) {
        filter.Add(key);
    }
    filterMemory = filter.SizeInBytes();

    for (auto _ : state) {
        timer.start();
        size_t found = 0;
        for (const auto& key : keys) {
            if (filter.Contain(key) == cuckoofilter::Ok) {
                found++;
            }
        }
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(found);
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(CPUCFFixture, Delete)(bm::State& state) {
    for (auto _ : state) {
        cuckoofilter::CuckooFilter<uint64_t, CPU_BITS_PER_ITEM> tempFilter(capacity);
        for (const auto& key : keys) {
            tempFilter.Add(key);
        }

        timer.start();
        size_t deleted = 0;
        for (const auto& key : keys) {
            if (tempFilter.Delete(key) == cuckoofilter::Ok) {
                deleted++;
            }
        }
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(deleted);
    }

    filterMemory = cuckoofilter::CuckooFilter<uint64_t, CPU_BITS_PER_ITEM>(capacity).SizeInBytes();
    setCounters(state);
}

BENCHMARK_DEFINE_F(CPUCFFixture, InsertQueryDelete)(bm::State& state) {
    for (auto _ : state) {
        cuckoofilter::CuckooFilter<uint64_t, CPU_BITS_PER_ITEM> tempFilter(capacity);

        timer.start();
        size_t inserted = 0;
        for (const auto& key : keys) {
            if (tempFilter.Add(key) == cuckoofilter::Ok) {
                inserted++;
            }
        }

        size_t found = 0;
        for (const auto& key : keys) {
            if (tempFilter.Contain(key) == cuckoofilter::Ok) {
                found++;
            }
        }

        size_t deleted = 0;
        for (const auto& key : keys) {
            if (tempFilter.Delete(key) == cuckoofilter::Ok) {
                deleted++;
            }
        }
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);
        bm::DoNotOptimize(found);
        bm::DoNotOptimize(deleted);
    }

    filterMemory = cuckoofilter::CuckooFilter<uint64_t, CPU_BITS_PER_ITEM>(capacity).SizeInBytes();
    setCounters(state);
}

static void GPUCF_FPR(bm::State& state) {
    Timer timer;
    auto [capacity, n] = calculateCapacityAndSize(state.range(0), 0.95);

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

static void CPUCF_FPR(bm::State& state) {
    Timer timer;
    auto [capacity, n] = calculateCapacityAndSize(state.range(0), 0.95);

    auto keys = generateKeysCPU<uint64_t>(n, 42, 1, UINT32_MAX);

    cuckoofilter::CuckooFilter<uint64_t, CPU_BITS_PER_ITEM> filter(capacity);
    for (const auto& k : keys) {
        filter.Add(k);
    }

    size_t fprTestSize = std::min(n, size_t(1'000'000));
    auto neverInserted = generateKeysCPU<uint64_t>(fprTestSize, 99999, UINT32_MAX + 1, UINT64_MAX);

    size_t falsePositives = 0;
    for (auto _ : state) {
        timer.start();
        falsePositives = 0;
        for (const auto& k : neverInserted) {
            if (filter.Contain(k) == cuckoofilter::Ok) {
                ++falsePositives;
            }
        }
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(falsePositives);
    }

    double fpr = static_cast<double>(falsePositives) / static_cast<double>(fprTestSize);
    size_t filterMemory = filter.SizeInBytes();

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
REGISTER_BENCHMARK(CPUCFFixture, Insert);

REGISTER_BENCHMARK(GPUCFFixture, Query);
REGISTER_BENCHMARK(CPUCFFixture, Query);

REGISTER_BENCHMARK(GPUCFFixture, Delete);
REGISTER_BENCHMARK(CPUCFFixture, Delete);

REGISTER_BENCHMARK(GPUCFFixture, InsertQueryDelete);
REGISTER_BENCHMARK(CPUCFFixture, InsertQueryDelete);

REGISTER_FUNCTION_BENCHMARK(GPUCF_FPR);
REGISTER_FUNCTION_BENCHMARK(CPUCF_FPR);

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
