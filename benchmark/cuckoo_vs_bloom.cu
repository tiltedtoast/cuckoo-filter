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

constexpr double TARGET_LOAD_FACTOR = 0.95;
using Config = CuckooConfig<uint32_t, 16, 500, 128, 128>;

template <typename Config>
constexpr double cuckooBitsPerItem() {
    using TagType = typename std::conditional<
        Config::bitsPerTag <= 8,
        uint8_t,
        typename std::
            conditional<Config::bitsPerTag <= 16, uint16_t, uint32_t>::type>::
        type;

    constexpr size_t bucketSize = BucketsTableGpu<Config>::bucketSize;
    constexpr size_t bytesPerBucket = bucketSize * sizeof(TagType);
    constexpr size_t bitsPerBucket = bytesPerBucket * 8;

    return static_cast<double>(bitsPerBucket) /
           (bucketSize * TARGET_LOAD_FACTOR);
}

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

    auto keys = generateKeys<uint32_t>(n);

    for (auto _ : state) {
        state.PauseTiming();
        BucketsTableGpu<Config> table(n, TARGET_LOAD_FACTOR);
        state.ResumeTiming();

        size_t inserted = table.insertMany(keys.data(), n);

        state.PauseTiming();
        cudaDeviceSynchronize();
        state.ResumeTiming();

        bm::DoNotOptimize(inserted);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
}

static void BM_CuckooFilter_Query(bm::State& state) {
    const size_t n = state.range(0);

    BucketsTableGpu<Config> table(n, TARGET_LOAD_FACTOR);

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
}

static void BM_BloomFilter_Insert(bm::State& state) {
    const size_t n = state.range(0);

    constexpr auto bitsPerTag =
        static_cast<size_t>(cuckooBitsPerItem<Config>());

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
}

static void BM_BloomFilter_Query(bm::State& state) {
    const size_t n = state.range(0);

    constexpr auto bitsPerTag =
        static_cast<size_t>(cuckooBitsPerItem<Config>());

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
}

static void BM_CuckooFilter_InsertAndQuery(bm::State& state) {
    const size_t n = state.range(0);

    auto keys = generateKeys<uint32_t>(n);
    std::vector<uint8_t> output(n);
    bool* outputPtr = reinterpret_cast<bool*>(output.data());

    for (auto _ : state) {
        state.PauseTiming();
        BucketsTableGpu<Config> table(n, TARGET_LOAD_FACTOR);
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

    constexpr auto bitsPerTag =
        static_cast<size_t>(cuckooBitsPerItem<Config>());

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
    ->Range(1 << 16, 1 << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_BloomFilter_Insert)
    ->Range(1 << 16, 1 << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_CuckooFilter_Query)
    ->Range(1 << 16, 1 << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_BloomFilter_Query)
    ->Range(1 << 16, 1 << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_CuckooFilter_InsertAndQuery)
    ->Range(1 << 16, 1 << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_BloomFilter_InsertAndQuery)
    ->Range(1 << 16, 1 << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
