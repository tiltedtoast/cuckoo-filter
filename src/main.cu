#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <chrono>
#include <CLI/CLI.hpp>
#include <cstdint>
#include <ctime>
#include <CuckooFilter.cuh>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <bucket_policies.cuh>
#include <helpers.cuh>
#include <iostream>
#include <random>
#include <string_view>
#include <vector>

int main(int argc, char** argv) {
    CLI::App app{"Cuckoo Filter Benchmark"};

    int exponent = 20;
    double targetLoadFactor = 0.95;

    app.add_option("exponent", exponent, "Exponent for n = 2^x")
        ->required()
        ->check(CLI::PositiveNumber);

    app.add_option("-l,--load-factor", targetLoadFactor, "Target load factor")
        ->check(CLI::Range(0.0, 1.0))
        ->default_val(0.95);

    CLI11_PARSE(app, argc, argv);

    using Config = CuckooConfig<uint64_t, 16, 500, 128, 16, XorAltBucketPolicy>;

    size_t capacity = 1ULL << exponent;
    size_t n = capacity * targetLoadFactor;

    std::cout << std::format("Using {}\n", std::string_view(Config::AltBucketPolicy::name));

    thrust::device_vector<uint64_t> d_input(n);
    thrust::device_vector<uint8_t> d_output(n);

    unsigned int seed = std::random_device{}();
    thrust::transform(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(n),
        d_input.begin(),
        [seed] __device__(size_t idx) {
            thrust::default_random_engine rng(seed);
            thrust::uniform_int_distribution<uint64_t> dist(1, UINT32_MAX);
            rng.discard(idx);
            return dist(rng);
        }
    );

    auto filter = CuckooFilter<Config>(capacity);

    auto start = std::chrono::high_resolution_clock::now();
    size_t count = filter.insertMany(d_input);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    float loadFactor = filter.loadFactor();

    std::cout << std::format(
        "Inserted {} / {} items in {} ms (load factor = {})\n", count, n, duration, loadFactor
    );

    start = std::chrono::high_resolution_clock::now();
    filter.containsMany(d_input, d_output);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    thrust::host_vector<uint8_t> output = d_output;

    size_t found = countOnes(reinterpret_cast<bool*>(thrust::raw_pointer_cast(output.data())), n);
    std::cout << std::format("Found {} / {} items in {} ms\n", found, n, duration);

    // size_t occupiedSlots = filter.countOccupiedSlots();
    // std::cout << std::format("Occupied slots: {}\n", occupiedSlots);

    size_t fprTestSize = std::min(n, size_t(1'000'000));
    thrust::device_vector<uint64_t> d_neverInserted(fprTestSize);
    thrust::device_vector<uint8_t> d_fprOutput(fprTestSize);

    thrust::transform(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(fprTestSize),
        d_neverInserted.begin(),
        [seed] __device__(size_t idx) {
            thrust::default_random_engine rng(seed + 1);
            thrust::uniform_int_distribution<uint64_t> dist(
                static_cast<uint64_t>(UINT32_MAX) + 1, UINT64_MAX
            );
            rng.discard(idx);
            return dist(rng);
        }
    );

    start = std::chrono::high_resolution_clock::now();
    filter.containsMany(d_neverInserted, d_fprOutput);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    thrust::host_vector<uint8_t> fprOutput = d_fprOutput;

    size_t falsePositives =
        countOnes(reinterpret_cast<bool*>(thrust::raw_pointer_cast(fprOutput.data())), fprTestSize);

    double fpr = static_cast<double>(falsePositives) / static_cast<double>(fprTestSize) * 100.0;
    double theoreticalFPR =
        (2.0 * Config::bucketSize * loadFactor) / (std::pow(2, int(Config::bitsPerTag)));

    std::cout << std::format(
        "False Positive Rate: {} / {} = {}% (theoretical {}% for f = {}, b = {}, α = {})\n",
        falsePositives,
        fprTestSize,
        fpr,
        100 * theoreticalFPR,
        Config::bitsPerTag,
        filter.bucketSize,
        loadFactor
    );

    size_t deleteCount = n / 2;
    thrust::device_vector<uint64_t> d_deleteKeys(
        d_input.begin(), d_input.begin() + static_cast<ptrdiff_t>(deleteCount)
    );
    thrust::device_vector<uint8_t> d_deleteOutput(deleteCount);

    start = std::chrono::high_resolution_clock::now();
    size_t remaining = filter.deleteMany(d_deleteKeys, d_deleteOutput);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    thrust::host_vector<uint8_t> deleteOutput = d_deleteOutput;

    size_t deleted = countOnes(
        reinterpret_cast<bool*>(thrust::raw_pointer_cast(deleteOutput.data())), deleteCount
    );

    std::cout << std::format(
        "Deleted {} / {} items in {} ms (load factor = {})\n",
        deleted,
        deleteCount,
        duration,
        filter.loadFactor()
    );

    filter.containsMany(d_deleteKeys, d_deleteOutput);
    deleteOutput = d_deleteOutput;
    size_t stillFound = countOnes(
        reinterpret_cast<bool*>(thrust::raw_pointer_cast(deleteOutput.data())), deleteCount
    );
    std::cout << std::format(
        "After deletion, {} / {} deleted items still found\n", stillFound, deleteCount
    );

    size_t nonDeletedCount = n - deleteCount;
    thrust::device_vector<uint64_t> d_nonDeletedKeys(
        d_input.begin() + static_cast<ptrdiff_t>(deleteCount), d_input.end()
    );
    thrust::device_vector<uint8_t> d_nonDeletedOutput(nonDeletedCount);

    filter.containsMany(d_nonDeletedKeys, d_nonDeletedOutput);
    thrust::host_vector<uint8_t> nonDeletedOutput = d_nonDeletedOutput;
    size_t nonDeletedFound = countOnes(
        reinterpret_cast<bool*>(thrust::raw_pointer_cast(nonDeletedOutput.data())), nonDeletedCount
    );
    std::cout << std::format(
        "Non-deleted keys still found: {} / {}\n", nonDeletedFound, nonDeletedCount
    );
}