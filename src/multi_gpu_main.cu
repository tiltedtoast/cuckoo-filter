#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <chrono>
#include <CLI/CLI.hpp>
#include <cstdint>
#include <ctime>
#include <CuckooFilterMultiGPU.cuh>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <hash_strategies.cuh>
#include <helpers.cuh>
#include <iostream>
#include <random>

int main(int argc, char** argv) {
    CLI::App app{"Multi-GPU Cuckoo Filter Benchmark"};

    int exponent = 20;
    double targetLoadFactor = 0.95;
    int numGPUsToUse = -1;

    app.add_option("exponent", exponent, "Exponent for n = 2^x")
        ->required()
        ->check(CLI::PositiveNumber);

    app.add_option("-l,--load-factor", targetLoadFactor, "Target load factor")
        ->check(CLI::Range(0.0, 1.0))
        ->default_val(0.95);

    app.add_option("-g,--gpus", numGPUsToUse, "Number of GPUs to use (-1 for all available)")
        ->default_val(-1);

    CLI11_PARSE(app, argc, argv);

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (numGPUsToUse == -1) {
        numGPUsToUse = deviceCount;
    } else if (numGPUsToUse > deviceCount) {
        std::cerr << "Requested " << numGPUsToUse << " GPUs but only " << deviceCount
                  << " available. Using " << deviceCount << " GPUs." << std::endl;
        numGPUsToUse = deviceCount;
    }

    std::cout << "Using " << numGPUsToUse << (numGPUsToUse == 1 ? " GPU" : " GPUs") << std::endl;

    using Config = CuckooConfig<uint64_t, 16, 500, 128, 16, XorAltBucketPolicy>;

    size_t capacity = 1ULL << exponent;
    auto n = static_cast<size_t>(capacity * targetLoadFactor);

    std::cout << "Using " << Config::AltBucketPolicy::name << " as the hash strategy" << std::endl;

    thrust::host_vector<uint64_t> h_input(n);

    unsigned int seed = std::random_device{}();
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> dist(1, UINT32_MAX);

    for (size_t i = 0; i < n; ++i) {
        h_input[i] = dist(rng);
    }

    auto filter = CuckooFilterMultiGPU<Config>(static_cast<size_t>(numGPUsToUse), capacity);

    auto start = std::chrono::high_resolution_clock::now();
    size_t count = filter.insertMany(h_input);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Inserted " << count << " / " << n << " items in " << duration << " ms"
              << " (load factor = " << filter.loadFactor() << ")" << std::endl;

    thrust::host_vector<bool> h_output(n);

    start = std::chrono::high_resolution_clock::now();
    filter.containsMany(h_input, h_output);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    size_t found = countOnes(h_output.data(), n);
    std::cout << "Found " << found << " / " << n << " items in " << duration << " ms" << std::endl;

    size_t fprTestSize = std::min(n, size_t(1000000));
    thrust::host_vector<uint64_t> h_neverInserted(fprTestSize);
    thrust::host_vector<bool> h_fprOutput(fprTestSize);

    std::uniform_int_distribution<uint64_t> fprDist(
        static_cast<uint64_t>(UINT32_MAX) + 1, UINT64_MAX
    );

    for (size_t i = 0; i < fprTestSize; ++i) {
        h_neverInserted[i] = fprDist(rng);
    }

    start = std::chrono::high_resolution_clock::now();
    filter.containsMany(h_neverInserted, h_fprOutput);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    size_t falsePositives = countOnes(h_fprOutput.data(), fprTestSize);

    double fpr = static_cast<double>(falsePositives) / static_cast<double>(fprTestSize) * 100.0;
    double theoreticalFPR =
        static_cast<double>(2 * Config::bucketSize) / (1ULL << Config::bitsPerTag);

    std::cout << "False Positive Rate: " << falsePositives << " / " << fprTestSize << " = " << fpr
              << "% (theoretical " << 100 * theoreticalFPR << "% for " << Config::bitsPerTag
              << "-bit tags and " << Config::bucketSize << " tags per buckets)" << std::endl;

    size_t deleteCount = n / 2;
    thrust::host_vector<uint64_t> h_deleteKeys(deleteCount);
    thrust::host_vector<bool> h_deleteOutput(deleteCount);

    for (size_t i = 0; i < deleteCount; ++i) {
        h_deleteKeys[i] = h_input[i];
    }

    start = std::chrono::high_resolution_clock::now();
    size_t remaining = filter.deleteMany(h_deleteKeys, h_deleteOutput);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    size_t deleted = countOnes(h_deleteOutput.data(), deleteCount);
    std::cout << "Deleted " << deleted << " / " << deleteCount << " items in " << duration << " ms"
              << " (load factor = " << filter.loadFactor() << ")" << std::endl;

    filter.containsMany(h_deleteKeys, h_deleteOutput);
    size_t stillFound = countOnes(h_deleteOutput.data(), deleteCount);
    std::cout << "After deletion, " << stillFound << " / " << deleteCount
              << " deleted items still found" << std::endl;

    std::cout << "\nMulti-GPU Statistics:" << std::endl;
    std::cout << "  Total GPUs: " << numGPUsToUse << std::endl;
    std::cout << "  Total capacity: " << filter.totalCapacity() << std::endl;
    std::cout << "  Total occupied: " << filter.totalOccupiedSlots() << std::endl;

    return 0;
}
