#include <cuco/hash_functions.cuh>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/cstdlib>
#include <iostream>
#include <random>

cuda::std::byte* rand_string(size_t size, size_t seed = 0) {
    std::mt19937 mt(seed);
    std::uniform_int_distribution<uint8_t> dist(48, 122);

    cuda::std::byte* string;
    cudaMallocHost(&string, size);

    for (size_t i = 0; i < size; ++i) {
        string[i] = static_cast<cuda::std::byte>(dist(mt));
    }

    return string;
}

__global__ void kernel(
    cuco::default_hash_function<char> hf,
    cuda::std::byte** strings,
    size_t* sizes,
    uint32_t* hash_values,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    auto hash_value = hf.compute_hash(strings[idx], sizes[idx]);
    hash_values[idx] = hash_value;
}

int main() {
    constexpr size_t n = 10;
    constexpr size_t string_size = 10;

    cuda::std::byte* h_strings[n];
    size_t sizes[n];
    uint32_t h_hash_values[n];

    for (size_t i = 0; i < n; ++i) {
        sizes[i] = string_size;
        h_strings[i] = rand_string(sizes[i], i);
    }

    cuda::std::byte* h_d_strings[n];
    size_t* d_sizes;
    uint32_t* d_hash_values;

    for (size_t i = 0; i < n; ++i) {
        cudaMalloc(&h_d_strings[i], string_size);
        cudaMemcpy(
            h_d_strings[i], h_strings[i], string_size, cudaMemcpyHostToDevice
        );
    }

    cuda::std::byte** d_d_strings;
    cudaMalloc(&d_d_strings, n * sizeof(cuda::std::byte*));

    cudaMemcpy(
        d_d_strings,
        h_d_strings,
        n * sizeof(cuda::std::byte*),
        cudaMemcpyHostToDevice
    );

    cudaMalloc(&d_sizes, n * sizeof(size_t));
    cudaMemcpy(d_sizes, sizes, n * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMalloc(&d_hash_values, n * sizeof(uint32_t));

    kernel<<<1, n>>>(
        cuco::default_hash_function<char>(),
        d_d_strings,
        d_sizes,
        d_hash_values,
        n
    );

    cudaDeviceSynchronize();

    cudaMemcpy(
        h_hash_values,
        d_hash_values,
        n * sizeof(uint32_t),
        cudaMemcpyDeviceToHost
    );

    for (size_t i = 0; i < n; ++i) {
        std::cout << "String " << i << ": ";
        for (size_t j = 0; j < sizes[i]; ++j) {
            std::cout << static_cast<char>(h_strings[i][j]);
        }
        std::cout << " -> Hash value: " << h_hash_values[i] << std::endl;
    }

    for (auto& h_d_string : h_d_strings) {
        cudaFree(h_d_string);
    }

    cudaFree(d_d_strings);
    cudaFree(d_sizes);
    cudaFree(d_hash_values);

    for (auto& string : h_strings) {
        cudaFreeHost(string);
    }

    return 0;
}