#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <bulk_tcf_host.cuh>
#include <cuco/bloom_filter.cuh>
#include <gqf.cuh>
#include <gqf_int.cuh>

#include <cstdint>
#include <CuckooFilter.cuh>
#include <fstream>
#include <helpers.cuh>

#include "benchmark_common.cuh"

namespace bm = benchmark;

using Config = CuckooConfig<uint64_t, 16, 500, 256, 16>;
using TCFType = host_bulk_tcf<uint64_t, uint16_t>;
using BloomFilter = cuco::bloom_filter<uint64_t>;

static constexpr double LOAD_FACTOR = 0.95;

std::vector<uint64_t> loadKmerFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open k-mer file: " + filename);
    }

    uint64_t count;
    file.read(reinterpret_cast<char*>(&count), sizeof(uint64_t));

    if (!file || count == 0) {
        throw std::runtime_error("Failed to read k-mer count from: " + filename);
    }

    std::vector<uint64_t> kmers(count);
    file.read(reinterpret_cast<char*>(kmers.data()), count * sizeof(uint64_t));

    if (!file) {
        throw std::runtime_error("Failed to read k-mers from: " + filename);
    }

    return kmers;
}

size_t getQFSizeHost(QF* d_qf) {
    QF h_qf;
    cudaMemcpy(&h_qf, d_qf, sizeof(QF), cudaMemcpyDeviceToHost);

    qfmetadata h_metadata;
    cudaMemcpy(&h_metadata, h_qf.metadata, sizeof(qfmetadata), cudaMemcpyDeviceToHost);

    return h_metadata.total_size_in_bytes;
}

static std::vector<uint64_t> g_kmerData;
static thrust::device_vector<uint64_t>* g_deviceKeys = nullptr;

void ensureDeviceKeys() {
    if (g_deviceKeys == nullptr) {
        g_deviceKeys = new thrust::device_vector<uint64_t>(g_kmerData.size());
        thrust::copy(g_kmerData.begin(), g_kmerData.end(), g_deviceKeys->begin());
    }
}

static void GPUCF_Insert(bm::State& state) {
    GPUTimer timer;
    ensureDeviceKeys();

    size_t n = g_kmerData.size();
    size_t capacity = static_cast<size_t>(n / LOAD_FACTOR);

    auto filter = std::make_unique<CuckooFilter<Config>>(capacity);
    size_t filterMemory = filter->sizeInBytes();

    for (auto _ : state) {
        filter->clear();
        cudaDeviceSynchronize();

        timer.start();
        size_t inserted = adaptiveInsert(*filter, *g_deviceKeys);
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);
    }

    setCommonCounters(state, filterMemory, n);
}

static void GPUCF_Query(bm::State& state) {
    GPUTimer timer;
    ensureDeviceKeys();

    size_t n = g_kmerData.size();
    auto capacity = static_cast<size_t>(n / LOAD_FACTOR);

    auto filter = std::make_unique<CuckooFilter<Config>>(capacity);
    size_t filterMemory = filter->sizeInBytes();

    adaptiveInsert(*filter, *g_deviceKeys);
    cudaDeviceSynchronize();

    thrust::device_vector<uint8_t> d_output(n);

    for (auto _ : state) {
        timer.start();
        filter->containsMany(*g_deviceKeys, d_output);
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output.data().get());
    }

    setCommonCounters(state, filterMemory, n);
}

static void GPUCF_Delete(bm::State& state) {
    GPUTimer timer;
    ensureDeviceKeys();

    size_t n = g_kmerData.size();
    auto capacity = static_cast<size_t>(n / LOAD_FACTOR);

    auto filter = std::make_unique<CuckooFilter<Config>>(capacity);
    size_t filterMemory = filter->sizeInBytes();

    adaptiveInsert(*filter, *g_deviceKeys);
    cudaDeviceSynchronize();

    thrust::device_vector<uint8_t> d_output(n);

    for (auto _ : state) {
        // Re-insert before each delete iteration
        filter->clear();
        adaptiveInsert(*filter, *g_deviceKeys);
        cudaDeviceSynchronize();

        timer.start();
        filter->deleteMany(*g_deviceKeys, d_output);
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output.data().get());
    }

    setCommonCounters(state, filterMemory, n);
}

static void TCF_Insert(bm::State& state) {
    GPUTimer timer;
    ensureDeviceKeys();

    size_t n = g_kmerData.size();

    constexpr double TCF_CAPACITY_FACTOR = 0.85;
    auto requiredUsableCapacity = static_cast<size_t>(n / LOAD_FACTOR);
    auto capacity = static_cast<size_t>(requiredUsableCapacity / TCF_CAPACITY_FACTOR);

    size_t filterMemory = capacity * sizeof(uint16_t);

    uint64_t* d_misses;
    cudaMalloc(&d_misses, sizeof(uint64_t));

    for (auto _ : state) {
        TCFType* filter = TCFType::host_build_tcf(capacity);
        cudaMemset(d_misses, 0, sizeof(uint64_t));
        cudaDeviceSynchronize();

        timer.start();
        filter->bulk_insert(thrust::raw_pointer_cast(g_deviceKeys->data()), n, d_misses);
        cudaDeviceSynchronize();
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(filter);
        TCFType::host_free_tcf(filter);
    }

    cudaFree(d_misses);
    setCommonCounters(state, filterMemory, n);
}

static void TCF_Query(bm::State& state) {
    GPUTimer timer;
    ensureDeviceKeys();

    size_t n = g_kmerData.size();

    constexpr double TCF_CAPACITY_FACTOR = 0.85;
    auto requiredUsableCapacity = static_cast<size_t>(n / LOAD_FACTOR);
    auto capacity = static_cast<size_t>(requiredUsableCapacity / TCF_CAPACITY_FACTOR);

    size_t filterMemory = capacity * sizeof(uint16_t);

    TCFType* filter = TCFType::host_build_tcf(capacity);

    uint64_t* d_misses;
    cudaMalloc(&d_misses, sizeof(uint64_t));
    cudaMemset(d_misses, 0, sizeof(uint64_t));
    filter->bulk_insert(thrust::raw_pointer_cast(g_deviceKeys->data()), n, d_misses);
    cudaDeviceSynchronize();

    bool* d_output = nullptr;
    for (auto _ : state) {
        timer.start();
        d_output = filter->bulk_query(thrust::raw_pointer_cast(g_deviceKeys->data()), n);
        cudaDeviceSynchronize();
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output);
        cudaFree(d_output);
    }

    cudaFree(d_misses);
    TCFType::host_free_tcf(filter);
    setCommonCounters(state, filterMemory, n);
}

static void TCF_Delete(bm::State& state) {
    GPUTimer timer;
    ensureDeviceKeys();

    size_t n = g_kmerData.size();

    constexpr double TCF_CAPACITY_FACTOR = 0.85;
    auto requiredUsableCapacity = static_cast<size_t>(n / LOAD_FACTOR);
    auto capacity = static_cast<size_t>(requiredUsableCapacity / TCF_CAPACITY_FACTOR);

    size_t filterMemory = capacity * sizeof(uint16_t);

    uint64_t* d_misses;
    cudaMalloc(&d_misses, sizeof(uint64_t));

    for (auto _ : state) {
        TCFType* filter = TCFType::host_build_tcf(capacity);
        cudaMemset(d_misses, 0, sizeof(uint64_t));
        filter->bulk_insert(thrust::raw_pointer_cast(g_deviceKeys->data()), n, d_misses);
        cudaDeviceSynchronize();

        timer.start();
        filter->bulk_delete(thrust::raw_pointer_cast(g_deviceKeys->data()), n);
        cudaDeviceSynchronize();
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(filter);
        TCFType::host_free_tcf(filter);
    }

    cudaFree(d_misses);
    setCommonCounters(state, filterMemory, n);
}

static void GQF_Insert(bm::State& state) {
    GPUTimer timer;
    ensureDeviceKeys();

    size_t n = g_kmerData.size();
    auto capacity = static_cast<size_t>(n / LOAD_FACTOR);

    auto q = static_cast<uint32_t>(std::log2(capacity)) + 1;
    capacity = 1ULL << q;

    for (auto _ : state) {
        QF* qf;
        qf_malloc_device(&qf, q, true);
        cudaDeviceSynchronize();

        timer.start();
        bulk_insert(qf, n, thrust::raw_pointer_cast(g_deviceKeys->data()), 0);
        cudaDeviceSynchronize();
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(qf);
        qf_destroy_device(qf);
    }

    size_t filterMemory = (capacity * QF_BITS_PER_SLOT) / 8;
    setCommonCounters(state, filterMemory, n);
}

static void GQF_Query(bm::State& state) {
    GPUTimer timer;
    ensureDeviceKeys();

    size_t n = g_kmerData.size();
    auto capacity = static_cast<size_t>(n / LOAD_FACTOR);

    auto q = static_cast<uint32_t>(std::log2(capacity)) + 1;
    capacity = 1ULL << q;

    QF* qf;
    qf_malloc_device(&qf, q, true);
    bulk_insert(qf, n, thrust::raw_pointer_cast(g_deviceKeys->data()), 0);
    cudaDeviceSynchronize();

    size_t filterMemory = getQFSizeHost(qf);

    thrust::device_vector<uint64_t> d_results(n);

    for (auto _ : state) {
        timer.start();
        bulk_get(
            qf,
            n,
            thrust::raw_pointer_cast(g_deviceKeys->data()),
            thrust::raw_pointer_cast(d_results.data())
        );
        cudaDeviceSynchronize();
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_results.data().get());
    }

    qf_destroy_device(qf);
    setCommonCounters(state, filterMemory, n);
}

static void GQF_Delete(bm::State& state) {
    GPUTimer timer;
    ensureDeviceKeys();

    size_t n = g_kmerData.size();
    auto capacity = static_cast<size_t>(n / LOAD_FACTOR);

    auto q = static_cast<uint32_t>(std::log2(capacity)) + 1;
    capacity = 1ULL << q;

    size_t filterMemory = (capacity * QF_BITS_PER_SLOT) / 8;

    for (auto _ : state) {
        QF* qf;
        qf_malloc_device(&qf, q, true);
        bulk_insert(qf, n, thrust::raw_pointer_cast(g_deviceKeys->data()), 0);
        cudaDeviceSynchronize();

        timer.start();
        bulk_delete(qf, n, thrust::raw_pointer_cast(g_deviceKeys->data()), QF_NO_LOCK);
        cudaDeviceSynchronize();
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(qf);
        qf_destroy_device(qf);
    }

    setCommonCounters(state, filterMemory, n);
}

static void Bloom_Insert(bm::State& state) {
    GPUTimer timer;
    ensureDeviceKeys();

    size_t n = g_kmerData.size();
    auto capacity = static_cast<size_t>(n / LOAD_FACTOR);

    size_t numBlocks = (capacity * Config::bitsPerTag) /
                       (BloomFilter::words_per_block * sizeof(typename BloomFilter::word_type) * 8);
    if (numBlocks == 0) {
        numBlocks = 1;
    }

    size_t filterMemory =
        numBlocks * BloomFilter::words_per_block * sizeof(typename BloomFilter::word_type);

    for (auto _ : state) {
        auto filter = std::make_unique<BloomFilter>(numBlocks);
        cudaDeviceSynchronize();

        timer.start();
        filter->add(g_deviceKeys->begin(), g_deviceKeys->end());
        cudaDeviceSynchronize();
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(filter.get());
    }

    setCommonCounters(state, filterMemory, n);
}

static void Bloom_Query(bm::State& state) {
    GPUTimer timer;
    ensureDeviceKeys();

    size_t n = g_kmerData.size();
    auto capacity = static_cast<size_t>(n / LOAD_FACTOR);

    size_t numBlocks = (capacity * Config::bitsPerTag) /
                       (BloomFilter::words_per_block * sizeof(typename BloomFilter::word_type) * 8);
    if (numBlocks == 0) {
        numBlocks = 1;
    }

    auto filter = std::make_unique<BloomFilter>(numBlocks);
    filter->add(g_deviceKeys->begin(), g_deviceKeys->end());
    cudaDeviceSynchronize();

    size_t filterMemory = filter->block_extent() * BloomFilter::words_per_block *
                          sizeof(typename BloomFilter::word_type);

    thrust::device_vector<bool> d_output(n);

    for (auto _ : state) {
        timer.start();
        filter->contains(
            g_deviceKeys->begin(), g_deviceKeys->end(), thrust::raw_pointer_cast(d_output.data())
        );
        cudaDeviceSynchronize();
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output.data().get());
    }

    setCommonCounters(state, filterMemory, n);
}

#define KMER_CONFIG                 \
    ->Unit(benchmark::kMillisecond) \
        ->UseManualTime()           \
        ->Iterations(10)            \
        ->Repetitions(5)            \
        ->ReportAggregatesOnly(true)

BENCHMARK(GPUCF_Insert) KMER_CONFIG;
BENCHMARK(GPUCF_Query) KMER_CONFIG;
BENCHMARK(GPUCF_Delete) KMER_CONFIG;
BENCHMARK(TCF_Insert) KMER_CONFIG;
BENCHMARK(TCF_Query) KMER_CONFIG;
BENCHMARK(TCF_Delete) KMER_CONFIG;
BENCHMARK(GQF_Insert) KMER_CONFIG;
BENCHMARK(GQF_Query) KMER_CONFIG;
BENCHMARK(GQF_Delete) KMER_CONFIG;
BENCHMARK(Bloom_Insert) KMER_CONFIG;
BENCHMARK(Bloom_Query) KMER_CONFIG;

int main(int argc, char** argv) {
    std::string kmerFile;

    std::vector<char*> remainingArgs;
    remainingArgs.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--kmer-file" && i + 1 < argc) {
            kmerFile = argv[i + 1];
            ++i;
        } else {
            remainingArgs.push_back(argv[i]);
        }
    }

    if (kmerFile.empty()) {
        std::cerr << "Error: --kmer-file argument required\n";
        std::cerr << "Usage: " << argv[0] << " --kmer-file <path> [benchmark options]\n";
        return 1;
    }

    try {
        g_kmerData = loadKmerFile(kmerFile);
        std::cerr << "Loaded " << g_kmerData.size() << " k-mers for GPU filter benchmarking\n";
    } catch (const std::exception& e) {
        std::cerr << "Error loading k-mer file: " << e.what() << "\n";
        return 1;
    }

    int newArgc = static_cast<int>(remainingArgs.size());
    char** newArgv = remainingArgs.data();

    ::benchmark::Initialize(&newArgc, newArgv);
    if (::benchmark::ReportUnrecognizedArguments(newArgc, newArgv)) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    delete g_deviceKeys;
    g_deviceKeys = nullptr;

    fflush(stdout);
    std::_Exit(0);
}
