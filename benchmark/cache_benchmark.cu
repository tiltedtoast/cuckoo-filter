#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <bulk_tcf_host.cuh>
#include <CLI/CLI.hpp>
#include <cstddef>
#include <cstdint>
#include <CuckooFilter.cuh>
#include <cuco/bloom_filter.cuh>
#include <hash_strategies.cuh>
#include <helpers.cuh>
#include <iostream>
#include <string>
#include "benchmark_common.cuh"

using Config = CuckooConfig<uint64_t, 16, 500, 128, 16, XorAltBucketPolicy>;
using TCFType = host_bulk_tcf<uint64_t, uint16_t>;
using BloomFilter = cuco::bloom_filter<uint64_t>;

template <typename Filter>
size_t cucoNumBlocks(size_t n) {
    constexpr auto bitsPerWord = sizeof(typename Filter::word_type) * 8;
    return SDIV(n * Config::bitsPerTag, Filter::words_per_block * bitsPerWord);
}

void benchmarkCuckooInsert(size_t capacity, double loadFactor) {
    auto n = static_cast<size_t>(capacity * loadFactor);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);

    CuckooFilter<Config> filter(capacity);

    filter.insertMany(d_keys);
    cudaDeviceSynchronize();

    filter.clear();
    cudaDeviceSynchronize();
    filter.insertMany(d_keys);
    cudaDeviceSynchronize();
}

void benchmarkCuckooQuery(size_t capacity, double loadFactor) {
    auto n = static_cast<size_t>(capacity * loadFactor);

    thrust::device_vector<uint64_t> d_keys(n);
    thrust::device_vector<uint8_t> d_output(n);
    generateKeysGPU(d_keys);

    CuckooFilter<Config> filter(capacity);
    filter.insertMany(d_keys);
    cudaDeviceSynchronize();

    filter.containsMany(d_keys, d_output);
    cudaDeviceSynchronize();
}

void benchmarkCuckooDelete(size_t capacity, double loadFactor) {
    auto n = static_cast<size_t>(capacity * loadFactor);

    thrust::device_vector<uint64_t> d_keys(n);
    thrust::device_vector<uint8_t> d_output(n);
    generateKeysGPU(d_keys);

    CuckooFilter<Config> filter(capacity);
    filter.insertMany(d_keys);
    cudaDeviceSynchronize();

    filter.deleteMany(d_keys, d_output);
    cudaDeviceSynchronize();
}

void benchmarkBloomInsert(size_t capacity, double loadFactor) {
    auto n = static_cast<size_t>(capacity * loadFactor);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);

    const size_t numBlocks = cucoNumBlocks<BloomFilter>(capacity);
    BloomFilter filter(numBlocks);

    filter.add(d_keys.begin(), d_keys.end());
    cudaDeviceSynchronize();

    filter.clear();
    cudaDeviceSynchronize();
    filter.add(d_keys.begin(), d_keys.end());
    cudaDeviceSynchronize();
}

void benchmarkBloomQuery(size_t capacity, double loadFactor) {
    auto n = static_cast<size_t>(capacity * loadFactor);

    thrust::device_vector<uint64_t> d_keys(n);
    thrust::device_vector<uint8_t> d_output(n);
    generateKeysGPU(d_keys);

    const size_t numBlocks = cucoNumBlocks<BloomFilter>(capacity);
    BloomFilter filter(numBlocks);
    filter.add(d_keys.begin(), d_keys.end());
    cudaDeviceSynchronize();

    filter.contains(
        d_keys.begin(),
        d_keys.end(),
        reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data()))
    );
    cudaDeviceSynchronize();
}

void benchmarkTcfInsert(size_t capacity, double loadFactor) {
    auto n = static_cast<size_t>(capacity * loadFactor);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);

    uint64_t* d_misses;
    cudaMalloc(&d_misses, sizeof(uint64_t));

    TCFType* filter = TCFType::host_build_tcf(capacity);

    cudaMemset(d_misses, 0, sizeof(uint64_t));
    filter->bulk_insert(thrust::raw_pointer_cast(d_keys.data()), n, d_misses);
    cudaDeviceSynchronize();

    // Rebuild for actual benchmark
    TCFType::host_free_tcf(filter);
    filter = TCFType::host_build_tcf(capacity);

    cudaMemset(d_misses, 0, sizeof(uint64_t));
    filter->bulk_insert(thrust::raw_pointer_cast(d_keys.data()), n, d_misses);
    cudaDeviceSynchronize();

    TCFType::host_free_tcf(filter);
    cudaFree(d_misses);
}

void benchmarkTcfQuery(size_t capacity, double loadFactor) {
    auto n = static_cast<size_t>(capacity * loadFactor);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);

    uint64_t* d_misses;
    cudaMalloc(&d_misses, sizeof(uint64_t));

    TCFType* filter = TCFType::host_build_tcf(capacity);
    cudaMemset(d_misses, 0, sizeof(uint64_t));
    filter->bulk_insert(thrust::raw_pointer_cast(d_keys.data()), n, d_misses);
    cudaDeviceSynchronize();

    bool* d_output = filter->bulk_query(thrust::raw_pointer_cast(d_keys.data()), n);
    cudaDeviceSynchronize();

    cudaFree(d_output);
    TCFType::host_free_tcf(filter);
    cudaFree(d_misses);
}

void benchmarkTcfDelete(size_t capacity, double loadFactor) {
    auto n = static_cast<size_t>(capacity * loadFactor);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);

    uint64_t* d_misses;
    cudaMalloc(&d_misses, sizeof(uint64_t));

    TCFType* filter = TCFType::host_build_tcf(capacity);
    cudaMemset(d_misses, 0, sizeof(uint64_t));
    filter->bulk_insert(thrust::raw_pointer_cast(d_keys.data()), n, d_misses);
    cudaDeviceSynchronize();

    bool* d_output = filter->bulk_delete(thrust::raw_pointer_cast(d_keys.data()), n);
    cudaDeviceSynchronize();

    cudaFree(d_output);
    TCFType::host_free_tcf(filter);
    cudaFree(d_misses);
}

int main(int argc, char** argv) {
    CLI::App app{"GPU Filter Cache Benchmark"};

    std::string filter = "cuckoo";
    std::string operation = "insert";
    size_t exponent = 24;
    double loadFactor = 0.95;

    app.add_option("filter", filter, "Filter type: cuckoo, bloom, tcf")
        ->required()
        ->check(CLI::IsMember({"cuckoo", "bloom", "tcf"}));

    app.add_option("operation", operation, "Operation: insert, query, delete")
        ->required()
        ->check(CLI::IsMember({"insert", "query", "delete"}));

    app.add_option("exponent", exponent, "Exponent for capacity = 2^x")
        ->required()
        ->check(CLI::PositiveNumber);

    app.add_option("-l,--load-factor", loadFactor, "Load factor (0.0-1.0)")
        ->default_val(0.95)
        ->check(CLI::Range(0.0, 1.0));

    CLI11_PARSE(app, argc, argv);

    auto capacity = static_cast<size_t>(1ULL << exponent);

    if (filter == "bloom" && operation == "delete") {
        std::cerr << "Error: Bloom filter does not support delete operation\n";
        return 1;
    }

    auto n = static_cast<size_t>(capacity * loadFactor);

    std::cout << "Filter: " << filter << std::endl;
    std::cout << "Operation: " << operation << std::endl;
    std::cout << "Capacity: " << capacity << std::endl;
    std::cout << "Load Factor: " << loadFactor << std::endl;
    std::cout << "Number of keys: " << n << std::endl;

    if (filter == "cuckoo") {
        if (operation == "insert") {
            benchmarkCuckooInsert(capacity, loadFactor);
        } else if (operation == "query") {
            benchmarkCuckooQuery(capacity, loadFactor);
        } else if (operation == "delete") {
            benchmarkCuckooDelete(capacity, loadFactor);
        }
    } else if (filter == "bloom") {
        if (operation == "insert") {
            benchmarkBloomInsert(capacity, loadFactor);
        } else if (operation == "query") {
            benchmarkBloomQuery(capacity, loadFactor);
        }
    } else if (filter == "tcf") {
        if (operation == "insert") {
            benchmarkTcfInsert(capacity, loadFactor);
        } else if (operation == "query") {
            benchmarkTcfQuery(capacity, loadFactor);
        } else if (operation == "delete") {
            benchmarkTcfDelete(capacity, loadFactor);
        }
    }

    return 0;
}
