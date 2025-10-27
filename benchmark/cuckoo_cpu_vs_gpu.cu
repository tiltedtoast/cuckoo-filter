#include <benchmark/benchmark.h>
#include <cuckoofilter.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <CuckooFilter.cuh>
#include <helpers.cuh>
#include <random>

namespace bm = benchmark;

constexpr double TARGET_LOAD_FACTOR = 0.95;
using GPUConfig = CuckooConfig<uint32_t, 16, 500, 128, 128>;
constexpr size_t CPU_BITS_PER_ITEM = GPUConfig::bitsPerTag;

template <typename T>
void generateKeysGPU(thrust::device_vector<T>& d_keys, unsigned seed = 42) {
    size_t n = d_keys.size();
    thrust::transform(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(n),
        d_keys.begin(),
        [seed] __device__(size_t idx) {
            thrust::default_random_engine rng(seed);
            thrust::uniform_int_distribution<T> dist(
                1, std::numeric_limits<T>::max()
            );
            rng.discard(idx);
            return dist(rng);
        }
    );
}

template <typename T>
std::vector<T> generateKeysCPU(size_t n, unsigned seed = 42) {
    std::vector<T> keys(n);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<T> dist(1, std::numeric_limits<T>::max());
    std::generate(keys.begin(), keys.end(), [&]() { return dist(rng); });
    return keys;
}

static void BM_GPU_CuckooFilter_Insert(bm::State& state) {
    const size_t n = state.range(0) * TARGET_LOAD_FACTOR;

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);
    CuckooFilter<GPUConfig> filter(n, TARGET_LOAD_FACTOR);

    size_t filterMemory = filter.sizeInBytes();
    size_t capacity = filter.capacity();

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
        static_cast<double>(filterMemory),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(capacity),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void BM_GPU_CuckooFilter_Query(bm::State& state) {
    const size_t n = state.range(0) * TARGET_LOAD_FACTOR;

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);
    CuckooFilter<GPUConfig> filter(n, TARGET_LOAD_FACTOR);
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
        static_cast<double>(filterMemory),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(capacity),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void BM_GPU_CuckooFilter_Delete(bm::State& state) {
    const size_t n = state.range(0) * TARGET_LOAD_FACTOR;

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);
    CuckooFilter<GPUConfig> filter(n, TARGET_LOAD_FACTOR);
    thrust::device_vector<uint8_t> d_output(n);

    size_t filterMemory = filter.sizeInBytes();
    size_t capacity = filter.capacity();

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
        static_cast<double>(filterMemory),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(capacity),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void BM_GPU_CuckooFilter_InsertQueryDelete(bm::State& state) {
    const size_t n = state.range(0) * TARGET_LOAD_FACTOR;

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);
    thrust::device_vector<uint8_t> d_output(n);
    CuckooFilter<GPUConfig> filter(n, TARGET_LOAD_FACTOR);

    size_t filterMemory = filter.sizeInBytes();
    size_t capacity = filter.capacity();

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
        static_cast<double>(filterMemory),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(capacity),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void BM_CPU_CuckooFilter_Insert(bm::State& state) {
    const size_t n = state.range(0) * TARGET_LOAD_FACTOR;

    auto keys = generateKeysCPU<uint32_t>(n);
    cuckoofilter::CuckooFilter<uint32_t, CPU_BITS_PER_ITEM> filter(n);

    size_t filterMemory = filter.SizeInBytes();
    size_t capacity = filterMemory / (CPU_BITS_PER_ITEM / 8);

    for (auto _ : state) {
        state.PauseTiming();
        cuckoofilter::CuckooFilter<uint32_t, CPU_BITS_PER_ITEM> tempFilter(n);
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
        static_cast<double>(filterMemory),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(capacity),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void BM_CPU_CuckooFilter_Query(bm::State& state) {
    const size_t n = state.range(0) * TARGET_LOAD_FACTOR;

    auto keys = generateKeysCPU<uint32_t>(n);
    cuckoofilter::CuckooFilter<uint32_t, CPU_BITS_PER_ITEM> filter(n);

    for (const auto& key : keys) {
        filter.Add(key);
    }

    size_t filterMemory = filter.SizeInBytes();
    size_t capacity = filterMemory / (CPU_BITS_PER_ITEM / 8);

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
        static_cast<double>(filterMemory),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(capacity),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void BM_CPU_CuckooFilter_Delete(bm::State& state) {
    const size_t n = state.range(0) * TARGET_LOAD_FACTOR;

    auto keys = generateKeysCPU<uint32_t>(n);
    cuckoofilter::CuckooFilter<uint32_t, CPU_BITS_PER_ITEM> filter(n);

    size_t filterMemory = filter.SizeInBytes();
    size_t capacity = filterMemory / (CPU_BITS_PER_ITEM / 8);

    for (auto _ : state) {
        state.PauseTiming();
        cuckoofilter::CuckooFilter<uint32_t, CPU_BITS_PER_ITEM> tempFilter(n);
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
        static_cast<double>(filterMemory),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(capacity),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void BM_CPU_CuckooFilter_InsertQueryDelete(bm::State& state) {
    const size_t n = state.range(0) * TARGET_LOAD_FACTOR;

    auto keys = generateKeysCPU<uint32_t>(n);
    cuckoofilter::CuckooFilter<uint32_t, CPU_BITS_PER_ITEM> filter(n);

    size_t filterMemory = filter.SizeInBytes();
    size_t capacity = filterMemory / (CPU_BITS_PER_ITEM / 8);

    for (auto _ : state) {
        state.PauseTiming();
        cuckoofilter::CuckooFilter<uint32_t, CPU_BITS_PER_ITEM> tempFilter(n);
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
        static_cast<double>(filterMemory),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(capacity),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

BENCHMARK(BM_GPU_CuckooFilter_Insert)
    ->Range(1 << 16, 1 << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_GPU_CuckooFilter_Query)
    ->Range(1 << 16, 1 << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_GPU_CuckooFilter_Delete)
    ->Range(1 << 16, 1 << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_GPU_CuckooFilter_InsertQueryDelete)
    ->Range(1 << 16, 1 << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_CPU_CuckooFilter_Insert)
    ->Range(1 << 16, 1 << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_CPU_CuckooFilter_Query)
    ->Range(1 << 16, 1 << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_CPU_CuckooFilter_Delete)
    ->Range(1 << 16, 1 << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_CPU_CuckooFilter_InsertQueryDelete)
    ->Range(1 << 16, 1 << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
