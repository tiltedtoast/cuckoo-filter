#include <atomic>
#include <cstring>
#include <cuco/hash_functions.cuh>
#include <cuda/std/atomic>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <iostream>
#include <vector>
#include "common.cuh"

template <
    typename T,
    size_t bitsPerTag,
    size_t numSlots,
    size_t maxProbes,
    size_t blockSize>
class NaiveTable;

template <
    typename T,
    size_t bitsPerTag,
    size_t numSlots,
    size_t maxProbes,
    size_t blockSize>
__global__ void containsKernel(
    const T* keys,
    bool* output,
    size_t n,
    typename NaiveTable<T, bitsPerTag, numSlots, maxProbes, blockSize>::
        DeviceTableView table_view
);

template <
    typename T,
    size_t bitsPerTag,
    size_t numSlots = 256,
    size_t maxProbes = 500,
    size_t blockSize = 256>
class NaiveTable {
    static_assert(bitsPerTag <= 32, "The tag cannot be larger than 32 bits");
    static_assert(bitsPerTag >= 1, "The tag must be at least 1 bit");
    static_assert(
        bitsPerTag <= 8 * sizeof(T),
        "The tag cannot be larger than the size of the type"
    );
    static_assert(powerOfTwo(numSlots), "Number of slots must be a power of 2");

    using TagType = typename std::conditional<
        bitsPerTag <= 8,
        uint8_t,
        typename std::conditional<bitsPerTag <= 16, uint16_t, uint32_t>::type>::
        type;

    static constexpr TagType EMPTY = 0;
    static constexpr size_t tagMask = (1ULL << bitsPerTag) - 1;

    TagType* h_slots;
    TagType* d_slots;
    std::atomic<size_t> numOccupied{0};

    template <typename H>
    static __host__ __device__ uint32_t hash(const H& key) {
        auto bytes = reinterpret_cast<const cuda::std::byte*>(&key);
        cuco::xxhash_32<H> hasher;
        return hasher.compute_hash(bytes, sizeof(H));
    }

    static __host__ __device__ TagType fingerprint(const T& key) {
        uint32_t hash_val = hash(key);
        auto fp = static_cast<TagType>(hash_val & tagMask);
        // 0 is reserved for empty slots
        return fp == 0 ? 1 : fp;
    }

    static __host__ __device__ cuda::std::tuple<size_t, size_t, TagType>
    getCandidateSlots(const T& key) {
        TagType fp = fingerprint(key);
        size_t h1 = hash(key) & (numSlots - 1);
        size_t h2 = h1 ^ (hash(fp) & (numSlots - 1));
        return {h1, h2, fp};
    }

    __host__ __device__ size_t getAlternateSlot(size_t slot, TagType fp) const {
        return slot ^ (hash(fp) & (numSlots - 1));
    }

    __host__ bool tryInsertAtSlot(size_t slot, TagType tag) {
        if (h_slots[slot] == EMPTY) {
            h_slots[slot] = tag;
            numOccupied++;
            return true;
        }
        return false;
    }

    __host__ bool insertWithEviction(TagType fp, size_t start_slot) {
        TagType current_fp = fp;
        size_t current_slot = start_slot;

        for (size_t evictions = 0; evictions < maxProbes; ++evictions) {
            if (tryInsertAtSlot(current_slot, current_fp)) {
                return true;
            }

            TagType evicted_fp = h_slots[current_slot];
            h_slots[current_slot] = current_fp;

            current_fp = evicted_fp;
            current_slot = getAlternateSlot(current_slot, evicted_fp);
        }

        return false;
    }

   public:
    explicit NaiveTable() {
        CUDA_CALL(cudaMallocHost(&h_slots, numSlots * sizeof(TagType)));
        CUDA_CALL(cudaMemset(h_slots, 0, numSlots * sizeof(TagType)));
        CUDA_CALL(cudaMalloc(&d_slots, numSlots * sizeof(TagType)));
        CUDA_CALL(cudaMemcpy(
            d_slots, h_slots, numSlots * sizeof(TagType), cudaMemcpyHostToDevice
        ));
    }

    NaiveTable(T* items, size_t n) : NaiveTable(numSlots) {
        for (size_t i = 0; i < n; ++i) {
            insert(items[i]);
        }
    }

    __host__ void syncToDevice() {
        CUDA_CALL(cudaMemcpy(
            d_slots, h_slots, numSlots * sizeof(TagType), cudaMemcpyHostToDevice
        ));
    }

    __host__ bool insert(const T& key) {
        auto [h1, h2, fp] = getCandidateSlots(key);

        if (tryInsertAtSlot(h1, fp) || tryInsertAtSlot(h2, fp)) {
            return true;
        }

        return insertWithEviction(fp, h1);
    }

    __host__ bool contains(const T& key) const {
        auto [h1, h2, fp] = getCandidateSlots(key);
        return (h_slots[h1] == fp) || (h_slots[h2] == fp);
    }

    const bool* containsMany(const T* keys, const size_t n) {
        bool* d_output;
        bool* h_output;
        T* d_keys;

        assert(n <= numSlots && "n may not be larger than numSlots");

        CUDA_CALL(cudaMallocHost(&h_output, n * sizeof(bool)));
        CUDA_CALL(cudaMalloc(&d_output, n * sizeof(bool)));
        CUDA_CALL(cudaMalloc(&d_keys, n * sizeof(T)));

        CUDA_CALL(
            cudaMemcpy(d_keys, keys, n * sizeof(T), cudaMemcpyHostToDevice)
        );

        syncToDevice();

        size_t numBlocks = (n + blockSize - 1) / blockSize;
        containsKernel<T, bitsPerTag, numSlots, maxProbes, blockSize>
            <<<numBlocks, blockSize>>>(
                d_keys, d_output, n, this->get_device_view()
            );

        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaMemcpy(
            h_output, d_output, n * sizeof(bool), cudaMemcpyDeviceToHost
        ));

        CUDA_CALL(cudaFree(d_output));
        CUDA_CALL(cudaFree(d_keys));

        return h_output;
    }

    __host__ bool remove(const T& key) {
        auto [h1, h2, fp] = getCandidateSlots(key);

        if (h_slots[h1] == fp) {
            h_slots[h1] = EMPTY;
            numOccupied--;
            syncToDevice();
            return true;
        }

        if (h_slots[h2] == fp) {
            h_slots[h2] = EMPTY;
            numOccupied--;
            syncToDevice();
            return true;
        }

        return false;
    }

    __host__ void clear() {
        std::memset(h_slots, 0, numSlots * sizeof(TagType));
        CUDA_CALL(cudaMemset(d_slots, 0, numSlots * sizeof(TagType)));
        numOccupied = 0;
    }

    __host__ float loadFactor() const {
        return static_cast<float>(numOccupied.load()) / numSlots;
    }

    __device__ __host__ double expectedFalsePositiveRate() const {
        return 1.0 / (1ULL << bitsPerTag);
    }

    struct DeviceTableView {
        TagType* d_slots;

        __device__ bool contains(const T& key) const {
            auto [h1, h2, fp] = NaiveTable::getCandidateSlots(key);
            return (d_slots[h1] == fp) || (d_slots[h2] == fp);
        }
    };

    __host__ DeviceTableView get_device_view() {
        return DeviceTableView{d_slots};
    }

    ~NaiveTable() {
        if (h_slots) {
            CUDA_CALL(cudaFreeHost(h_slots));
        }
        if (d_slots) {
            CUDA_CALL(cudaFree(d_slots));
        }
    }
};

template <
    typename T,
    size_t bitsPerTag,
    size_t numSlots,
    size_t maxProbes,
    size_t blockSize>
__global__ void containsKernel(
    const T* keys,
    bool* output,
    size_t n,
    typename NaiveTable<T, bitsPerTag, numSlots, maxProbes, blockSize>::
        DeviceTableView table_view
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = table_view.contains(keys[idx]);
    }
}