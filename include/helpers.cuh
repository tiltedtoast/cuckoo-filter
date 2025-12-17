#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>

/**
 * @brief Checks if a number is a power of two.
 * @param n Number to check.
 * @return true if n is a power of two, false otherwise.
 */
constexpr bool powerOfTwo(size_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

/**
 * @brief Calculates the global thread ID in a 1D grid.
 * @return uint32_t Global thread ID.
 */
__host__ __device__ __forceinline__ uint32_t globalThreadId() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

/**
 * @brief Calculates the next power of two greater than or equal to n.
 * @param n Input number.
 * @return size_t Next power of two.
 */
constexpr size_t nextPowerOfTwo(size_t n) {
    if (powerOfTwo(n))
        return n;

    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    n++;

    return n;
}

/**
 * @brief Counts the number of non-zero elements in an array.
 * @tparam T Type of elements.
 * @param data Pointer to the array.
 * @param n Number of elements.
 * @return size_t Number of non-zero elements.
 */
template <typename T>
size_t countOnes(T* data, size_t n) {
    size_t count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (data[i]) {
            count++;
        }
    }
    return count;
}

/**
 * @brief Returns a bitmask indicating which slots in a packed word are zero.
 *
 * Uses SWAR (SIMD Within A Register) to check multiple items in parallel.
 * See https://graphics.stanford.edu/~seander/bithacks.html#ZeroInWord
 *
 * The high bit of each slot that is zero will be set in the result.
 *
 * @tparam TagType The type of the individual items (uint8_t, uint16_t, or uint32_t)
 * @tparam WordType The packed word type (uint32_t or uint64_t)
 * @param v The packed integer
 * @return A bitmask with the high bit of each zero slot set
 */
template <typename TagType, typename WordType>
__host__ __device__ __forceinline__ constexpr WordType getZeroMask(WordType v) {
    static_assert(sizeof(WordType) == 4 || sizeof(WordType) == 8, "WordType must be 32 or 64 bits");

    if constexpr (sizeof(WordType) == 8) {
        if constexpr (sizeof(TagType) == 1) {
            return (v - 0x0101010101010101ULL) & ~v & 0x8080808080808080ULL;
        } else if constexpr (sizeof(TagType) == 2) {
            return (v - 0x0001000100010001ULL) & ~v & 0x8000800080008000ULL;
        } else if constexpr (sizeof(TagType) == 4) {
            return (v - 0x0000000100000001ULL) & ~v & 0x8000000080000000ULL;
        } else {
            return 0;
        }
    } else {
        if constexpr (sizeof(TagType) == 1) {
            return (v - 0x01010101U) & ~v & 0x80808080U;
        } else if constexpr (sizeof(TagType) == 2) {
            return (v - 0x00010001U) & ~v & 0x80008000U;
        } else if constexpr (sizeof(TagType) == 4) {
            return (v - 0x00000001U) & ~v & 0x80000000U;
        } else {
            return 0;
        }
    }
}

/**
 * @brief Checks if a packed word contains a zero slot.
 *
 * @tparam TagType The type of the individual items (uint8_t, uint16_t, or uint32_t)
 * @tparam WordType The packed word type (uint32_t or uint64_t)
 * @param v The packed integer
 * @return true if any of the items in v are zero
 */
template <typename TagType, typename WordType>
__host__ __device__ __forceinline__ constexpr bool hasZero(WordType v) {
    return getZeroMask<TagType, WordType>(v) != 0;
}

/**
 * @brief Replicates a tag value across all slots in a word.
 *
 * @tparam TagType The type of the tag (uint8_t, uint16_t, or uint32_t)
 * @tparam WordType The target word type (uint32_t or uint64_t)
 * @param tag The tag value to replicate
 * @return A word with the tag replicated in every slot
 */
template <typename TagType, typename WordType>
__host__ __device__ __forceinline__ constexpr WordType replicateTag(TagType tag) {
    static_assert(sizeof(WordType) == 4 || sizeof(WordType) == 8, "WordType must be 32 or 64 bits");

    if constexpr (sizeof(WordType) == 8) {
        if constexpr (sizeof(TagType) == 1) {
            return static_cast<uint64_t>(tag) * 0x0101010101010101ULL;
        } else if constexpr (sizeof(TagType) == 2) {
            return static_cast<uint64_t>(tag) * 0x0001000100010001ULL;
        } else if constexpr (sizeof(TagType) == 4) {
            return static_cast<uint64_t>(tag) * 0x0000000100000001ULL;
        } else {
            return tag;
        }
    } else {
        if constexpr (sizeof(TagType) == 1) {
            return static_cast<uint32_t>(tag) * 0x01010101U;
        } else if constexpr (sizeof(TagType) == 2) {
            return static_cast<uint32_t>(tag) * 0x00010001U;
        } else if constexpr (sizeof(TagType) == 4) {
            return static_cast<uint32_t>(tag);
        } else {
            return tag;
        }
    }
}

#if __CUDA_ARCH__ >= 1000

/**
 * @brief Loads 256 bits from global memory using the non-coherent cache path.
 *
 * This function uses inline PTX for 256-bit vectorized loads.
 * For uint64_t: loads 4 values (v4.u64)
 * For uint32_t: loads 8 values (v8.u32)
 *
 * @note Only available on sm_100+ architectures with PTX 8.8.
 *       Use __CUDA_ARCH__ >= 1000 guard at call sites.
 *
 * @tparam T Element type (uint32_t or uint64_t)
 * @param ptr Source pointer (must be 32-byte aligned)
 * @param out Output array (4 elements for uint64_t, 8 for uint32_t)
 */
template <typename T>
__device__ __forceinline__ void load256BitGlobalNC(const T* ptr, T* out) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "T must be uint32_t or uint64_t");

    if constexpr (sizeof(T) == 8) {
        asm volatile("ld.global.nc.v4.u64 {%0, %1, %2, %3}, [%4];"
                     : "=l"(out[0]), "=l"(out[1]), "=l"(out[2]), "=l"(out[3])
                     : "l"(ptr));
    } else {
        asm volatile("ld.global.nc.v8.u32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                     : "=r"(out[0]),
                       "=r"(out[1]),
                       "=r"(out[2]),
                       "=r"(out[3]),
                       "=r"(out[4]),
                       "=r"(out[5]),
                       "=r"(out[6]),
                       "=r"(out[7])
                     : "l"(ptr));
    }
}

#endif

/**
 * @brief Integer division with rounding up (ceiling).
 */
#define SDIV(x, y) (((x) + (y) - 1) / (y))

/**
 * @brief Macro for checking CUDA errors.
 * Prints error message and exits if an error occurs.
 */
#define CUDA_CALL(err)                                                      \
    do {                                                                    \
        cudaError_t err_ = (err);                                           \
        if (err_ == cudaSuccess) [[likely]] {                               \
            break;                                                          \
        }                                                                   \
        printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(err_);                                                         \
    } while (0)

/**
 * @brief Calculates the maximum occupancy grid size for a kernel.
 *
 * @tparam Kernel Type of the kernel function.
 * @param blockSize Block size (threads per block).
 * @param kernel The kernel function.
 * @param dynamicSMemSize Dynamic shared memory size per block.
 * @return size_t The calculated grid size (number of blocks).
 */
template <typename Kernel>
constexpr size_t maxOccupancyGridSize(int32_t blockSize, Kernel kernel, size_t dynamicSMemSize) {
    int device = 0;
    cudaGetDevice(&device);

    int numSM = -1;
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, device);

    int maxActiveBlocksPerSM{};
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocksPerSM, kernel, blockSize, dynamicSMemSize
    );

    return maxActiveBlocksPerSM * numSM;
}
