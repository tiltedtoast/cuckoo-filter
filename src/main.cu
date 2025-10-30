#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <CuckooFilter.cuh>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <helpers.cuh>
#include <iostream>
#include <random>
#include <vector>

constexpr double TARGET_LOAD_FACTOR = 0.95;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <n_exponent> " << std::endl;
        std::cerr << "n_exponent: exponent for n = 2^x" << std::endl;
        return 1;
    }

    int n_exponent = std::atoi(argv[1]);

    if (n_exponent < 1) {
        std::cerr << "Invalid exponent. Use >= 1" << std::endl;
        return 1;
    }

    size_t n = (UINT64_C(1) << n_exponent) * TARGET_LOAD_FACTOR;

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

    using Config = CuckooConfig<uint64_t, 16, 500, 256, 128>;
    auto filter = CuckooFilter<Config>(n);

    auto start = std::chrono::high_resolution_clock::now();
    size_t count = filter.insertMany(d_input);
    filter.containsMany(d_input, d_output);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::vector<uint8_t> output(n);
    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    size_t found = countOnes(reinterpret_cast<bool*>(output.data()), n);
    std::cout << "Inserted " << count << " / " << n << " items, found " << found << " items in "
              << duration << " ms"
              << " (load factor = " << filter.loadFactor() << ")" << std::endl;

    size_t fprTestSize = std::min(n, size_t(1000000));
    thrust::device_vector<uint64_t> d_neverInserted(fprTestSize);
    thrust::device_vector<uint8_t> d_fprOutput(fprTestSize);

    thrust::transform(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(fprTestSize),
        d_neverInserted.begin(),
        [seed] __device__(size_t idx) {
            thrust::default_random_engine rng(seed + 99999);
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

    std::vector<uint8_t> fprOutput(fprTestSize);
    thrust::copy(d_fprOutput.begin(), d_fprOutput.end(), fprOutput.begin());

    size_t falsePositives = countOnes(reinterpret_cast<bool*>(fprOutput.data()), fprTestSize);

    double fpr = static_cast<double>(falsePositives) / static_cast<double>(fprTestSize) * 100.0;
    double theoreticalFPR = 1.0 / (1ULL << Config::bitsPerTag);

    std::cout << "False Positive Rate: " << falsePositives << " / " << fprTestSize << " = " << fpr
              << "% (theoretical " << 100 * theoreticalFPR << "% for " << Config::bitsPerTag
              << "-bit tags)" << std::endl;

    size_t deleteCount = n / 2;
    thrust::device_vector<uint64_t> d_deleteKeys(
        d_input.begin(), d_input.begin() + static_cast<ptrdiff_t>(deleteCount)
    );
    thrust::device_vector<uint8_t> d_deleteOutput(deleteCount);

    start = std::chrono::high_resolution_clock::now();
    size_t remaining = filter.deleteMany(d_deleteKeys, d_deleteOutput);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::vector<uint8_t> deleteOutput(deleteCount);
    thrust::copy(d_deleteOutput.begin(), d_deleteOutput.end(), deleteOutput.begin());

    size_t deleted = countOnes(reinterpret_cast<bool*>(deleteOutput.data()), deleteCount);
    std::cout << "Deleted " << deleted << " / " << deleteCount << " items in " << duration << " ms"
              << " (load factor = " << filter.loadFactor() << ")" << std::endl;

    filter.containsMany(d_deleteKeys, d_deleteOutput);
    thrust::copy(d_deleteOutput.begin(), d_deleteOutput.end(), deleteOutput.begin());
    size_t stillFound = countOnes(reinterpret_cast<bool*>(deleteOutput.data()), deleteCount);
    std::cout << "After deletion, " << stillFound << " / " << deleteCount
              << " deleted items still found" << std::endl;
}
