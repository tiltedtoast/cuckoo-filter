#pragma once

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <thread>
#include <type_traits>
#include <vector>
#include "CuckooFilter.cuh"
#include "helpers.cuh"

#include <nccl.h>

#define NCCL_CALL(cmd)                                                                    \
    do {                                                                                  \
        ncclResult_t r = cmd;                                                             \
        if (r != ncclSuccess) {                                                           \
            printf("NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    } while (0)

template <typename Config>
class CuckooFilterMultiGPU {
   public:
    using T = typename Config::KeyType;

    struct Partitioner {
        size_t numGPUs;

        __host__ __device__ size_t operator()(const T& key) const {
            uint64_t hash = CuckooFilter<Config>::hash64(key);
            return hash % numGPUs;
        }
    };

   private:
    size_t numGPUs;
    size_t capacityPerGPU;
    std::vector<CuckooFilter<Config>*> filters;
    std::vector<cudaStream_t> streams;
    std::vector<ncclComm_t> comms;

    /**
     * @brief Gets the available free memory for each GPU.
     * @return A vector where the i-th element is the free memory on GPU i.
     */
    [[nodiscard]] std::vector<size_t> getGpuMemoryInfo() const {
        std::vector<size_t> freeMem(numGPUs);
        parallelForGPUs([&](size_t gpuId) {
            size_t free, total;
            CUDA_CALL(cudaMemGetInfo(&free, &total));
            freeMem[gpuId] = free;
        });
        return freeMem;
    }

    /**
     * @brief Partitions keys (and optionally an index array) by their target GPU. This function
     * sorts the keys in place.
     */
    void partitionByGPU(
        thrust::device_vector<T>& d_keys,
        thrust::device_vector<size_t>& d_counts,
        thrust::device_vector<size_t>& d_offsets,
        thrust::device_vector<size_t>* d_indices = nullptr
    ) {
        size_t n = d_keys.size();
        if (n == 0) {
            d_counts.assign(numGPUs, 0);
            d_offsets.assign(numGPUs, 0);
            return;
        }

        thrust::device_vector<size_t> d_gpuIndices(n);
        Partitioner partitioner{numGPUs};
        thrust::transform(
            thrust::device, d_keys.begin(), d_keys.end(), d_gpuIndices.begin(), partitioner
        );

        // Sort keys and optionally indices based on the target GPU index
        if (d_indices) {
            thrust::sort_by_key(
                thrust::device,
                d_gpuIndices.begin(),
                d_gpuIndices.end(),
                thrust::make_zip_iterator(thrust::make_tuple(d_keys.begin(), d_indices->begin()))
            );
        } else {
            thrust::sort_by_key(
                thrust::device, d_gpuIndices.begin(), d_gpuIndices.end(), d_keys.begin()
            );
        }

        d_counts.assign(numGPUs, 0);
        d_offsets.assign(numGPUs, 0);

        // Count the number of keys destined for each GPU
        thrust::device_vector<size_t> d_uniqueGPUIds(numGPUs);
        auto end = thrust::reduce_by_key(
            thrust::device,
            d_gpuIndices.begin(),
            d_gpuIndices.end(),
            thrust::make_constant_iterator<size_t>(1),
            d_uniqueGPUIds.begin(),
            d_counts.begin()
        );

        // Scatter the counts to their correct positions in the final counts vector
        size_t numFoundGPUs = thrust::distance(d_uniqueGPUIds.begin(), end.first);
        thrust::device_vector<size_t> d_finalCounts(numGPUs, 0);
        thrust::scatter(
            thrust::device,
            d_counts.begin(),
            d_counts.begin() + numFoundGPUs,
            d_uniqueGPUIds.begin(),
            d_finalCounts.begin()
        );
        d_counts = d_finalCounts;

        // Calculate offsets for data shuffling
        thrust::exclusive_scan(
            thrust::device, d_counts.begin(), d_counts.end(), d_offsets.begin(), 0
        );
    }

    /**
     * @brief Generic function for querying the filter and reordering results.
     * Processes keys in chunks dynamically sized based on available GPU VRAM.
     * @tparam returnOccupied If true, the function queries and returns
     * the total number of occupied slots after the operation.
     * @tparam FilterFunctor A callable that performs the desired CuckooFilter operation (e.g.,
     * contains, delete).
     * @param h_keys Host vector of keys to process.
     * @param h_output Host vector to store the boolean results.
     * @param filterOp The specific filter operation to execute.
     * @return Total occupied slots if returnOccupied is true, otherwise 0.
     */
    template <bool returnOccupied, typename FilterFunctor>
    size_t queryAndReorder(
        const thrust::host_vector<T>& h_keys,
        thrust::host_vector<bool>* h_output,
        FilterFunctor filterOp
    ) {
        size_t n = h_keys.size();
        h_output->resize(n);

        if (n == 0) {
            return returnOccupied ? totalOccupiedSlots() : 0;
        }

        size_t processedCount = 0;
        while (processedCount < n) {
            std::vector<size_t> freeMem = getGpuMemoryInfo();

            // Estimate memory needed per item for sorting, including overhead.
            // (key + original_index + target_gpu_index) * safety_factor
            const size_t memPerItem = sizeof(T) + 2 * sizeof(size_t);

            // Thrust likes to use A LOT of temporary memory
            const float safetyFactor = 3.0f;
            const auto requiredMemPerItem = static_cast<size_t>(memPerItem * safetyFactor);

            std::vector<size_t> chunkSizes(numGPUs);
            size_t totalChunkSize = 0;
            size_t remainingKeys = n - processedCount;

            // Determine how many keys each GPU can handle in this chunk
            for (size_t i = 0; i < numGPUs; ++i) {
                size_t maxItemsForGPU = freeMem[i] / requiredMemPerItem;
                chunkSizes[i] = maxItemsForGPU;
                totalChunkSize += chunkSizes[i];
            }

            totalChunkSize = std::min(totalChunkSize, remainingKeys);

            std::vector<size_t> chunkOffsets(numGPUs + 1, 0);
            // Proportionally resize chunk sizes to match the total we can process
            for (size_t i = 0; i < numGPUs; ++i) {
                auto keysForGPU = static_cast<size_t>(
                    static_cast<double>(chunkSizes[i]) /
                    static_cast<double>(
                        std::accumulate(chunkSizes.begin(), chunkSizes.end(), size_t(0))
                    ) *
                    totalChunkSize
                );
                chunkOffsets[i + 1] = chunkOffsets[i] + keysForGPU;
            }

            // Ensure the last offset covers the entire chunk due to potential rounding
            chunkOffsets[numGPUs] = totalChunkSize;

            parallelForGPUs([&](size_t gpuId) {
                size_t localChunkStart = chunkOffsets[gpuId];
                size_t localChunkEnd = chunkOffsets[gpuId + 1];
                size_t localChunkSize = localChunkEnd - localChunkStart;

                if (localChunkSize == 0) {
                    return;
                }

                thrust::device_vector<T> d_localKeys(
                    h_keys.begin() + processedCount + localChunkStart,
                    h_keys.begin() + processedCount + localChunkEnd
                );

                thrust::device_vector<size_t> d_localIndices(localChunkSize);
                thrust::sequence(thrust::device, d_localIndices.begin(), d_localIndices.end(), 0);

                thrust::device_vector<size_t> d_sendCounts(numGPUs);
                thrust::device_vector<size_t> d_sendOffsets(numGPUs);
                partitionByGPU(d_localKeys, d_sendCounts, d_sendOffsets, &d_localIndices);

                thrust::device_vector<size_t> d_recvCounts(numGPUs);

                exchangeCounts(gpuId, d_sendCounts, d_recvCounts);

                thrust::device_vector<size_t> d_recvOffsets(numGPUs);
                thrust::exclusive_scan(
                    thrust::device,
                    d_recvCounts.begin(),
                    d_recvCounts.end(),
                    d_recvOffsets.begin(),
                    0
                );

                size_t totalToReceive = thrust::reduce(
                    thrust::device, d_recvCounts.begin(), d_recvCounts.end(), size_t(0)
                );

                if (totalToReceive == 0) {
                    return;
                }

                // Shuffle keys between GPUs (All-to-All)
                thrust::device_vector<T> d_receivedKeys(totalToReceive);
                NCCL_CALL(ncclGroupStart());
                for (size_t peerId = 0; peerId < numGPUs; ++peerId) {
                    if (d_sendCounts[peerId] > 0) {
                        NCCL_CALL(ncclSend(
                            d_localKeys.data().get() + d_sendOffsets[peerId],
                            d_sendCounts[peerId] * sizeof(T),
                            ncclChar,
                            peerId,
                            comms[gpuId],
                            streams[gpuId]
                        ));
                    }
                    if (d_recvCounts[peerId] > 0) {
                        NCCL_CALL(ncclRecv(
                            d_receivedKeys.data().get() + d_recvOffsets[peerId],
                            d_recvCounts[peerId] * sizeof(T),
                            ncclChar,
                            peerId,
                            comms[gpuId],
                            streams[gpuId]
                        ));
                    }
                }
                NCCL_CALL(ncclGroupEnd());

                // Perform the filter operation on the received keys
                thrust::device_vector<bool> d_localResults(totalToReceive);
                filterOp(filters[gpuId], d_receivedKeys, d_localResults, streams[gpuId]);

                // Shuffle results back to the original GPUs
                thrust::device_vector<bool> d_resultsToSendBack(localChunkSize);
                NCCL_CALL(ncclGroupStart());
                for (size_t peerId = 0; peerId < numGPUs; ++peerId) {
                    if (d_recvCounts[peerId] > 0) {
                        NCCL_CALL(ncclSend(
                            d_localResults.data().get() + d_recvOffsets[peerId],
                            d_recvCounts[peerId] * sizeof(bool),
                            ncclChar,
                            peerId,
                            comms[gpuId],
                            streams[gpuId]
                        ));
                    }
                    if (d_sendCounts[peerId] > 0) {
                        NCCL_CALL(ncclRecv(
                            d_resultsToSendBack.data().get() + d_sendOffsets[peerId],
                            d_sendCounts[peerId] * sizeof(bool),
                            ncclChar,
                            peerId,
                            comms[gpuId],
                            streams[gpuId]
                        ));
                    }
                }
                NCCL_CALL(ncclGroupEnd());

                // Unsort the results to match the original input order for this chunk
                thrust::device_vector<bool> d_finalChunkOutput(localChunkSize);
                thrust::scatter(
                    thrust::device,
                    d_resultsToSendBack.begin(),
                    d_resultsToSendBack.end(),
                    d_localIndices.begin(),
                    d_finalChunkOutput.begin()
                );

                // Copy the chunk's results back to the correct position
                // in the final host output vector
                thrust::copy(
                    d_finalChunkOutput.begin(),
                    d_finalChunkOutput.end(),
                    h_output->begin() + processedCount + localChunkStart
                );
            });

            synchronizeAllGPUs();
            processedCount += totalChunkSize;
        }

        return returnOccupied ? totalOccupiedSlots() : 0;
    }

    /**
     * @brief Exchanges the send and receive counts between GPUs.
     * This is expected to be called once for each GPU.
     * @param gpuId Current GPU Id
     * @param d_sendCounts The send counts to exchange
     * @param d_recvCounts The receive counts to exchange
     */
    void exchangeCounts(
        size_t gpuId,
        thrust::device_vector<size_t>& d_sendCounts,
        thrust::device_vector<size_t>& d_recvCounts
    ) {
        NCCL_CALL(ncclGroupStart());
        for (size_t peerId = 0; peerId < numGPUs; ++peerId) {
            NCCL_CALL(ncclSend(
                d_sendCounts.data().get() + peerId,
                1,
                ncclUint64,
                peerId,
                comms[gpuId],
                streams[gpuId]
            ));
            NCCL_CALL(ncclRecv(
                d_recvCounts.data().get() + peerId,
                1,
                ncclUint64,
                peerId,
                comms[gpuId],
                streams[gpuId]
            ));
        }
        NCCL_CALL(ncclGroupEnd());
    }

   public:
    CuckooFilterMultiGPU(size_t numGPUs, size_t capacity)
        : numGPUs(numGPUs), capacityPerGPU(static_cast<size_t>(SDIV(capacity, numGPUs))) {
        assert(numGPUs > 0 && "Number of GPUs must be at least 1");

        streams.resize(numGPUs);
        comms.resize(numGPUs);
        filters.resize(numGPUs);

        ncclUniqueId ncclUid;
        NCCL_CALL(ncclGetUniqueId(&ncclUid));

        parallelForGPUs([&](size_t i) {
            CUDA_CALL(cudaSetDevice(i));
            NCCL_CALL(ncclCommInitRank(&comms[i], numGPUs, ncclUid, i));
            CUDA_CALL(cudaStreamCreate(&streams[i]));
            filters[i] = new CuckooFilter<Config>(capacityPerGPU);
        });
        synchronizeAllGPUs();
    }

    ~CuckooFilterMultiGPU() {
        parallelForGPUs([&](size_t i) {
            CUDA_CALL(cudaSetDevice(i));
            delete filters[i];
            NCCL_CALL(ncclCommDestroy(comms[i]));
            CUDA_CALL(cudaStreamDestroy(streams[i]));
        });
    }

    CuckooFilterMultiGPU(const CuckooFilterMultiGPU&) = delete;
    CuckooFilterMultiGPU& operator=(const CuckooFilterMultiGPU&) = delete;

    /**
     * @brief Inserts a batch of keys into the distributed filter.
     * Processes keys in chunks dynamically sized based on available GPU VRAM to prevent OOM errors.
     * @param h_keys The keys to insert.
     * @return The total number of occupied slots across all GPUs after insertion.
     */
    size_t insertMany(const thrust::host_vector<T>& h_keys) {
        size_t n = h_keys.size();
        if (n == 0) {
            return totalOccupiedSlots();
        }

        size_t processedCount = 0;
        while (processedCount < n) {
            std::vector<size_t> freeMem = getGpuMemoryInfo();

            // Estimate memory needed per item, including overhead.
            // (key + target_gpu_index) * safety_factor
            const size_t memPerItem = sizeof(T) + sizeof(size_t);

            // Thrust likes to use A LOT of temporary memory
            const float safetyFactor = 3.0f;
            const auto requiredMemPerItem = static_cast<size_t>(memPerItem * safetyFactor);

            std::vector<size_t> chunkSizes(numGPUs);
            size_t totalChunkSize = 0;
            size_t remainingKeys = n - processedCount;

            for (size_t i = 0; i < numGPUs; ++i) {
                size_t maxItemsForGPU = freeMem[i] / requiredMemPerItem;
                chunkSizes[i] = maxItemsForGPU;
                totalChunkSize += chunkSizes[i];
            }

            totalChunkSize = std::min(totalChunkSize, remainingKeys);

            std::vector<size_t> chunkOffsets(numGPUs + 1, 0);
            for (size_t i = 0; i < numGPUs; ++i) {
                auto keysForGPU = static_cast<size_t>(
                    static_cast<double>(chunkSizes[i]) /
                    static_cast<double>(
                        std::accumulate(chunkSizes.begin(), chunkSizes.end(), size_t(0))
                    ) *
                    totalChunkSize
                );

                chunkOffsets[i + 1] = chunkOffsets[i] + keysForGPU;
            }
            chunkOffsets[numGPUs] = totalChunkSize;

            parallelForGPUs([&](size_t gpuId) {
                size_t localChunkStart = chunkOffsets[gpuId];
                size_t localChunkEnd = chunkOffsets[gpuId + 1];
                size_t localChunkSize = localChunkEnd - localChunkStart;

                if (localChunkSize <= 0) {
                    return;
                }

                thrust::device_vector<T> d_localKeys(
                    h_keys.begin() + processedCount + localChunkStart,
                    h_keys.begin() + processedCount + localChunkEnd
                );

                thrust::device_vector<size_t> d_sendCounts(numGPUs);
                thrust::device_vector<size_t> d_sendOffsets(numGPUs);
                partitionByGPU(d_localKeys, d_sendCounts, d_sendOffsets);

                thrust::device_vector<size_t> d_recvCounts(numGPUs);

                exchangeCounts(gpuId, d_sendCounts, d_recvCounts);

                thrust::device_vector<size_t> d_recvOffsets(numGPUs);
                thrust::exclusive_scan(
                    thrust::device,
                    d_recvCounts.begin(),
                    d_recvCounts.end(),
                    d_recvOffsets.begin(),
                    0
                );
                size_t totalToReceive = thrust::reduce(
                    thrust::device, d_recvCounts.begin(), d_recvCounts.end(), size_t(0)
                );

                if (totalToReceive == 0) {
                    return;
                }

                thrust::device_vector<T> d_receivedKeys(totalToReceive);
                NCCL_CALL(ncclGroupStart());
                for (size_t peerId = 0; peerId < numGPUs; ++peerId) {
                    if (d_sendCounts[peerId] > 0) {
                        NCCL_CALL(ncclSend(
                            d_localKeys.data().get() + d_sendOffsets[peerId],
                            d_sendCounts[peerId] * sizeof(T),
                            ncclChar,
                            peerId,
                            comms[gpuId],
                            streams[gpuId]
                        ));
                    }
                    if (d_recvCounts[peerId] > 0) {
                        NCCL_CALL(ncclRecv(
                            d_receivedKeys.data().get() + d_recvOffsets[peerId],
                            d_recvCounts[peerId] * sizeof(T),
                            ncclChar,
                            peerId,
                            comms[gpuId],
                            streams[gpuId]
                        ));
                    }
                }
                NCCL_CALL(ncclGroupEnd());

                filters[gpuId]->insertMany(d_receivedKeys, streams[gpuId]);
            });

            synchronizeAllGPUs();
            processedCount += totalChunkSize;
        }

        return totalOccupiedSlots();
    }

    void containsMany(const thrust::host_vector<T>& h_keys, thrust::host_vector<bool>& h_output) {
        queryAndReorder<false>(
            h_keys,
            &h_output,
            [](CuckooFilter<Config>* filter,
               const thrust::device_vector<T>& keys,
               thrust::device_vector<bool>& results,
               cudaStream_t stream) { filter->containsMany(keys, results, stream); }
        );
    }

    size_t deleteMany(const thrust::host_vector<T>& h_keys, thrust::host_vector<bool>& h_output) {
        return queryAndReorder<true>(
            h_keys,
            &h_output,
            [](CuckooFilter<Config>* filter,
               const thrust::device_vector<T>& keys,
               thrust::device_vector<bool>& results,
               cudaStream_t stream) { filter->deleteMany(keys, results, stream); }
        );
    }

    float loadFactor() {
        return static_cast<float>(totalOccupiedSlots()) / static_cast<float>(totalCapacity());
    }

    template <typename Func>
    void parallelForGPUs(Func func) const {
        std::vector<std::thread> threads;
        for (size_t i = 0; i < numGPUs; ++i) {
            threads.emplace_back([=]() {
                CUDA_CALL(cudaSetDevice(i));
                func(i);
            });
        }

        for (auto& t : threads) {
            t.join();
        }
    }

    void synchronizeAllGPUs() {
        parallelForGPUs([&](size_t i) { CUDA_CALL(cudaStreamSynchronize(streams[i])); });
    }

    size_t totalOccupiedSlots() {
        std::atomic<size_t> total(0);
        parallelForGPUs([&](size_t i) {
            total.fetch_add(filters[i]->occupiedSlots(), std::memory_order_relaxed);
        });

        return total.load();
    }

    void clear() {
        parallelForGPUs([&](size_t i) { filters[i]->clear(); });
    }

    [[nodiscard]] size_t totalCapacity() const {
        std::atomic<size_t> total(0);
        parallelForGPUs([&](size_t i) {
            total.fetch_add(filters[i]->capacity(), std::memory_order_relaxed);
        });
        return total.load();
    }
};