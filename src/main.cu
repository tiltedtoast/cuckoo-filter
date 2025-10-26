#include <thrust/device_vector.h>
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

    if (n_exponent < 1 || n_exponent > 30) {
        std::cerr << "Invalid exponent. Use 1-30." << std::endl;
        return 1;
    }

    size_t n = (UINT64_C(1) << n_exponent) * TARGET_LOAD_FACTOR;

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<uint32_t> dist(1, UINT32_MAX);

    std::vector<uint32_t> input(n);
    std::generate(input.begin(), input.end(), [&]() { return dist(rng); });

    thrust::device_vector<uint32_t> d_input(input.begin(), input.end());
    thrust::device_vector<uint8_t> d_output(n);

    using Config = CuckooConfig<uint32_t, 16, 500, 128, 128>;
    auto filter = CuckooFilter<Config>(n, TARGET_LOAD_FACTOR);

    auto start = std::chrono::high_resolution_clock::now();
    size_t count = filter.insertMany(d_input);
    filter.containsMany(d_input, d_output);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();

    std::vector<uint8_t> output(n);
    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    size_t found = countOnes(reinterpret_cast<bool*>(output.data()), n);
    std::cout << "Inserted " << count << " / " << n << " items, found " << found
              << " items in " << duration << " ms"
              << " (load factor = " << filter.loadFactor() << ")" << std::endl;

    size_t deleteCount = n / 2;
    thrust::device_vector<uint32_t> d_deleteKeys(
        d_input.begin(), d_input.begin() + static_cast<ptrdiff_t>(deleteCount)
    );
    thrust::device_vector<uint8_t> d_deleteOutput(deleteCount);

    start = std::chrono::high_resolution_clock::now();
    size_t remaining = filter.deleteMany(d_deleteKeys, d_deleteOutput);
    end = std::chrono::high_resolution_clock::now();
    duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();

    std::vector<uint8_t> deleteOutput(deleteCount);
    thrust::copy(
        d_deleteOutput.begin(), d_deleteOutput.end(), deleteOutput.begin()
    );

    size_t deleted =
        countOnes(reinterpret_cast<bool*>(deleteOutput.data()), deleteCount);
    std::cout << "Deleted " << deleted << " / " << deleteCount << " items in "
              << duration << " ms"
              << " (load factor = " << filter.loadFactor() << ")" << std::endl;

    filter.containsMany(d_deleteKeys, d_deleteOutput);
    thrust::copy(
        d_deleteOutput.begin(), d_deleteOutput.end(), deleteOutput.begin()
    );
    size_t stillFound =
        countOnes(reinterpret_cast<bool*>(deleteOutput.data()), deleteCount);
    std::cout << "After deletion, " << stillFound << " / " << deleteCount
              << " deleted items still found (false positives)" << std::endl;
}