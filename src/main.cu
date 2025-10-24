#include <BucketsTableGpu.cuh>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <helpers.cuh>
#include <iostream>
#include <random>

constexpr double TARGET_LOAD_FACTOR = 0.95;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <table_type> " << std::endl;
        std::cerr << "n_exponent: exponent for n = 2^x" << std::endl;
        return 1;
    }

    int n_exponent = std::atoi(argv[1]);

    if (n_exponent < 1 || n_exponent > 30) {
        std::cerr << "Invalid exponent. Use 1-30." << std::endl;
        return 1;
    }

    size_t n = (UINT64_C(1) << n_exponent) * TARGET_LOAD_FACTOR;

    uint32_t* input;
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<uint32_t> dist(1, UINT32_MAX);

    CUDA_CALL(cudaMallocHost(&input, sizeof(uint32_t) * n));

    std::generate(input, input + n, [&]() { return dist(rng); });

    using Config = CuckooConfig<uint32_t, 16, 500, 128, 128>;
    auto table = BucketsTableGpu<Config>(n, TARGET_LOAD_FACTOR);

    bool* output;

    CUDA_CALL(cudaMallocHost(&output, sizeof(bool) * n));

    auto start = std::chrono::high_resolution_clock::now();
    size_t count = table.insertMany(input, n);
    table.containsMany(input, n, output);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();

    size_t found = countOnes(output, n);
    std::cout << "Inserted " << count << " / " << n << " items, found " << found
              << " items in " << duration << " ms"
              << " (load factor = " << table.loadFactor() << ")" << std::endl;

    CUDA_CALL(cudaFreeHost(input));
    CUDA_CALL(cudaFreeHost(output));
}
