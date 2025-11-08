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

constexpr double TARGET_LOAD_FACTOR = 0.95;
using GPUConfig = CuckooConfig<uint32_t, 16, 500, 128, 16>;
constexpr size_t CPU_BITS_PER_ITEM = GPUConfig::bitsPerTag;

template <typename T>
std::vector<T>
generateKeysCPU(size_t n, unsigned seed = 42, T min = 1, T max = std::numeric_limits<T>::max()) {
    std::vector<T> keys(n);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<T> dist(min, max);
    std::generate(keys.begin(), keys.end(), [&]() { return dist(rng); });
    return keys;
}

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

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(capacity),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
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

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(capacity),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void GPU_CuckooFilter_Delete(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<GPUConfig>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);
    CuckooFilter<GPUConfig> filter(capacity);
    thrust::device_vector<uint8_t> d_output(n);

    size_t filterMemory = filter.sizeInBytes();

    for (auto _ : state) {
        state.PauseTiming();
        filter.clear();
        filter.insertMany(d_keys);
        state.ResumeTiming();

        size_t remaining = filter.deleteMany(d_keys, d_output);
        cudaDeviceSynchronize();
        bm::DoNotOptimize(remaining);
        bm::DoNotOptimize(d_output.data().get());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(capacity),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void GPU_CuckooFilter_InsertQueryDelete(bm::State& state) {
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
        size_t remaining = filter.deleteMany(d_keys, d_output);

        cudaDeviceSynchronize();

        bm::DoNotOptimize(inserted);
        bm::DoNotOptimize(remaining);
        bm::DoNotOptimize(d_output.data().get());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(capacity),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void CPU_CuckooFilter_Insert(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<GPUConfig>(state.range(0), TARGET_LOAD_FACTOR);

    auto keys = generateKeysCPU<uint32_t>(n);
    cuckoofilter::CuckooFilter<uint32_t, CPU_BITS_PER_ITEM> filter(capacity);

    size_t filterMemory = filter.SizeInBytes();
    size_t actualCapacity = filterMemory / (CPU_BITS_PER_ITEM / 8);

    for (auto _ : state) {
        state.PauseTiming();
        cuckoofilter::CuckooFilter<uint32_t, CPU_BITS_PER_ITEM> tempFilter(capacity);
        state.ResumeTiming();

        size_t inserted = 0;
        for (const auto& key : keys) {
            if (tempFilter.Add(key) == cuckoofilter::Ok) {
                inserted++;
            }
        }
        bm::DoNotOptimize(inserted);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(actualCapacity),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void CPU_CuckooFilter_Query(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<GPUConfig>(state.range(0), TARGET_LOAD_FACTOR);

    auto keys = generateKeysCPU<uint32_t>(n);
    cuckoofilter::CuckooFilter<uint32_t, CPU_BITS_PER_ITEM> filter(capacity);

    for (const auto& key : keys) {
        filter.Add(key);
    }

    size_t filterMemory = filter.SizeInBytes();
    size_t actualCapacity = filterMemory / (CPU_BITS_PER_ITEM / 8);

    for (auto _ : state) {
        size_t found = 0;
        for (const auto& key : keys) {
            if (filter.Contain(key) == cuckoofilter::Ok) {
                found++;
            }
        }
        bm::DoNotOptimize(found);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(actualCapacity),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void CPU_CuckooFilter_Delete(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<GPUConfig>(state.range(0), TARGET_LOAD_FACTOR);

    auto keys = generateKeysCPU<uint32_t>(n);
    cuckoofilter::CuckooFilter<uint32_t, CPU_BITS_PER_ITEM> filter(capacity);

    size_t filterMemory = filter.SizeInBytes();
    size_t actualCapacity = filterMemory / (CPU_BITS_PER_ITEM / 8);

    for (auto _ : state) {
        state.PauseTiming();
        cuckoofilter::CuckooFilter<uint32_t, CPU_BITS_PER_ITEM> tempFilter(capacity);
        for (const auto& key : keys) {
            tempFilter.Add(key);
        }
        state.ResumeTiming();

        size_t deleted = 0;
        for (const auto& key : keys) {
            if (tempFilter.Delete(key) == cuckoofilter::Ok) {
                deleted++;
            }
        }
        bm::DoNotOptimize(deleted);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(actualCapacity),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void CPU_CuckooFilter_InsertQueryDelete(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<GPUConfig>(state.range(0), TARGET_LOAD_FACTOR);

    auto keys = generateKeysCPU<uint32_t>(n);
    cuckoofilter::CuckooFilter<uint32_t, CPU_BITS_PER_ITEM> filter(capacity);

    size_t filterMemory = filter.SizeInBytes();
    size_t actualCapacity = filterMemory / (CPU_BITS_PER_ITEM / 8);

    for (auto _ : state) {
        state.PauseTiming();
        cuckoofilter::CuckooFilter<uint32_t, CPU_BITS_PER_ITEM> tempFilter(capacity);
        state.ResumeTiming();

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

        bm::DoNotOptimize(inserted);
        bm::DoNotOptimize(found);
        bm::DoNotOptimize(deleted);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(actualCapacity),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void GPU_CuckooFilter_FalsePositiveRate(bm::State& state) {
    using FPRConfig = CuckooConfig<uint64_t, 16, 500, 128, 4>;

    auto [capacity, n] = calculateCapacityAndSize<FPRConfig>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU<uint64_t>(d_keys, UINT32_MAX);

    CuckooFilter<FPRConfig> filter(capacity);
    filter.insertMany(d_keys);

    size_t fprTestSize = std::min(n, size_t(1'000'000));
    thrust::device_vector<uint64_t> d_neverInserted(fprTestSize);
    thrust::device_vector<uint8_t> d_output(fprTestSize);

    thrust::transform(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(fprTestSize),
        d_neverInserted.begin(),
        [] __device__(size_t idx) {
            thrust::default_random_engine rng(99999);
            thrust::uniform_int_distribution<uint64_t> dist(
                static_cast<uint64_t>(UINT32_MAX) + 1, UINT64_MAX
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
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(n),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
    );
}

static void CPU_CuckooFilter_FalsePositiveRate(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<GPUConfig>(state.range(0), TARGET_LOAD_FACTOR);

    auto keys = generateKeysCPU<uint64_t>(n, 1, UINT32_MAX);
    cuckoofilter::CuckooFilter<uint64_t, CPU_BITS_PER_ITEM> filter(capacity);
    for (const auto& k : keys) {
        filter.Add(k);
    }

    size_t fprTestSize = std::min(n, size_t(1'000'000));
    auto neverInserted = generateKeysCPU<uint64_t>(fprTestSize, 99999, UINT32_MAX + 1, UINT64_MAX);

    size_t falsePositives = 0;
    for (auto _ : state) {
        falsePositives = 0;
        for (const auto& k : neverInserted) {
            if (filter.Contain(k) == cuckoofilter::Ok) {
                ++falsePositives;
            }
        }
        bm::DoNotOptimize(falsePositives);
    }

    double fpr = static_cast<double>(falsePositives) / static_cast<double>(fprTestSize);
    size_t filterMemory = filter.SizeInBytes();

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * fprTestSize));
    state.counters["fpr_percentage"] = bm::Counter(fpr * 100);
    state.counters["false_positives"] = bm::Counter(static_cast<double>(falsePositives));
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(n),
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
BENCHMARK(CPU_CuckooFilter_Insert)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(GPU_CuckooFilter_Query)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);
BENCHMARK(CPU_CuckooFilter_Query)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(GPU_CuckooFilter_Delete)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);
BENCHMARK(CPU_CuckooFilter_Delete)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(GPU_CuckooFilter_InsertQueryDelete)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);
BENCHMARK(CPU_CuckooFilter_InsertQueryDelete)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(GPU_CuckooFilter_FalsePositiveRate)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);
BENCHMARK(CPU_CuckooFilter_FalsePositiveRate)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
