#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <BucketsTableGpu.cuh>
#include <cuco/bloom_filter.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>
#include <cuda/std/cstdint>
#include <helpers.cuh>
#include <random>

namespace bm = benchmark;

template <typename T>
std::vector<T> generateKeys(size_t n, unsigned seed = 42) {
    std::vector<T> keys(n);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<T> dist(1, std::numeric_limits<T>::max());
    std::generate(keys.begin(), keys.end(), [&]() { return dist(rng); });

    return keys;
}

template <typename Filter, size_t bitsPerTag>
size_t cucoNumBlocks(size_t n) {
    constexpr auto bitsPerWord = sizeof(typename Filter::word_type) * 8;

    return (n * bitsPerTag) / (Filter::words_per_block * bitsPerWord);
}

static void BM_CuckooFilter_Insert(bm::State& state) {
    const size_t n = state.range(0);
    const size_t numBuckets = n / 32;

    using Config = CuckooConfig<uint32_t, 16, 1000, 256, 128>;

    auto keys = generateKeys<uint32_t>(n);

    for (auto _ : state) {
        state.PauseTiming();
        BucketsTableGpu<Config> table(numBuckets);
        state.ResumeTiming();

        size_t inserted = table.insertMany(keys.data(), n);

        state.PauseTiming();
        cudaDeviceSynchronize();
        state.ResumeTiming();

        bm::DoNotOptimize(inserted);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations() * n * sizeof(uint32_t))
    );
}

static void BM_CuckooFilter_Query(bm::State& state) {
    const size_t n = state.range(0);
    const size_t numBuckets = n / 32;

    using Config = CuckooConfig<uint32_t, 16, 1000, 256, 128>;
    BucketsTableGpu<Config> table(numBuckets);

    auto keys = generateKeys<uint32_t>(n);
    table.insertMany(keys.data(), n);

    std::vector<uint8_t> output(n);
    bool* outputPtr = reinterpret_cast<bool*>(output.data());

    for (auto _ : state) {
        table.containsMany(keys.data(), n, outputPtr);
        cudaDeviceSynchronize();
        bm::DoNotOptimize(outputPtr);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations() * n * sizeof(uint32_t))
    );
}

static void BM_BloomFilter_Insert(bm::State& state) {
    const size_t n = state.range(0);

    constexpr std::size_t bitsPerTag = 16;

    using BloomFilter = cuco::bloom_filter<uint32_t>;

    const size_t numBlocks = cucoNumBlocks<BloomFilter, bitsPerTag>(n);

    auto keys = generateKeys<uint32_t>(n);
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());

    for (auto _ : state) {
        state.PauseTiming();
        BloomFilter filter(
            cuco::extent{numBlocks},
            cuco::cuda_thread_scope<cuda::thread_scope_device>{}
        );
        state.ResumeTiming();

        filter.add(d_keys.begin(), d_keys.end());
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations() * n * sizeof(uint32_t))
    );
}

static void BM_BloomFilter_Query(bm::State& state) {
    const size_t n = state.range(0);

    constexpr std::size_t bitsPerTag = 16;

    using BloomFilter = cuco::bloom_filter<uint32_t>;

    const size_t numBlocks = cucoNumBlocks<BloomFilter, bitsPerTag>(n);

    BloomFilter filter(
        cuco::extent{numBlocks},
        cuco::cuda_thread_scope<cuda::thread_scope_device>{}
    );

    auto keys = generateKeys<uint32_t>(n);
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());
    thrust::device_vector<uint8_t> d_output(n);

    filter.add(d_keys.begin(), d_keys.end());

    for (auto _ : state) {
        filter.contains(
            d_keys.begin(),
            d_keys.end(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data()))
        );
        cudaDeviceSynchronize();
        bm::DoNotOptimize(d_output.data().get());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations() * n * sizeof(uint32_t))
    );
}

static void BM_CuckooFilter_InsertAndQuery(bm::State& state) {
    const size_t n = state.range(0);
    const size_t numBuckets = n / 32;

    using Config = CuckooConfig<uint32_t, 16, 1000, 256, 128>;

    auto keys = generateKeys<uint32_t>(n);
    std::vector<uint8_t> output(n);
    bool* outputPtr = reinterpret_cast<bool*>(output.data());

    for (auto _ : state) {
        state.PauseTiming();
        BucketsTableGpu<Config> table(numBuckets);
        state.ResumeTiming();

        size_t inserted = table.insertMany(keys.data(), n);
        table.containsMany(keys.data(), n, outputPtr);
        cudaDeviceSynchronize();

        bm::DoNotOptimize(inserted);
        bm::DoNotOptimize(outputPtr);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
}

static void BM_BloomFilter_InsertAndQuery(bm::State& state) {
    const size_t n = state.range(0);

    constexpr std::size_t bitsPerTag = 16;

    using BloomFilter = cuco::bloom_filter<uint32_t>;
    const size_t numBlocks = cucoNumBlocks<BloomFilter, bitsPerTag>(n);

    auto keys = generateKeys<uint32_t>(n);
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());
    thrust::device_vector<uint8_t> d_output(n);

    for (auto _ : state) {
        state.PauseTiming();
        BloomFilter filter(
            cuco::extent{numBlocks},
            cuco::cuda_thread_scope<cuda::thread_scope_device>{}
        );
        state.ResumeTiming();

        filter.add(d_keys.begin(), d_keys.end());
        filter.contains(
            d_keys.begin(),
            d_keys.end(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data()))
        );
        cudaDeviceSynchronize();

        bm::DoNotOptimize(d_output.data().get());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
}

BENCHMARK(BM_CuckooFilter_Insert)
    ->Range(1 << 16, 1 << 24)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_BloomFilter_Insert)
    ->Range(1 << 16, 1 << 24)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_CuckooFilter_Query)
    ->Range(1 << 16, 1 << 24)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_BloomFilter_Query)
    ->Range(1 << 16, 1 << 24)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_CuckooFilter_InsertAndQuery)
    ->Range(1 << 16, 1 << 24)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_BloomFilter_InsertAndQuery)
    ->Range(1 << 16, 1 << 24)
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
