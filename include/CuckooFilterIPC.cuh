#pragma once

#include <cuda_runtime.h>
#include <fcntl.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <atomic>
#include <cstring>
#include <string>
#include <thread>
#include "CuckooFilter.cuh"
#include "helpers.cuh"

enum class RequestType {
    INSERT = 0,
    CONTAINS = 1,
    DELETE = 2,
    CLEAR = 3,
    SHUTDOWN = 4
};

struct FilterRequest {
    RequestType type;
    uint32_t count;                   // Number of keys in this batch
    cudaIpcMemHandle_t keysHandle;    // handle to device memory containing keys
    cudaIpcMemHandle_t outputHandle;  // optional handle for results (for lookup/deletion)
    uint64_t requestId;               // Unique request identifier
    std::atomic<bool> completed;      // Completion flag
    std::atomic<bool> cancelled;      // Cancellation flag (for force shutdown)
    size_t result;                    // Updated number of occupied slots after insert/delete
};

struct SharedRingBuffer {
    static constexpr size_t QUEUE_SIZE = 256;
    static_assert(powerOfTwo(QUEUE_SIZE), "queue size must be a power of two");

    std::atomic<uint64_t> head;  // Producer index
    std::atomic<uint64_t> tail;  // Consumer index
    sem_t producerSem;           // Semaphore for available slots
    sem_t consumerSem;           // Semaphore for pending requests

    FilterRequest requests[QUEUE_SIZE];

    bool initialised;
    std::atomic<bool> shuttingDown;

    FilterRequest* enqueue() {
        // Check if server is shutting down before trying to do anything
        if (shuttingDown.load(std::memory_order_acquire)) {
            return nullptr;
        }

        // Wait for available slot
        if (sem_wait(&producerSem) != 0) {
            return nullptr;
        }

        // Double-check after acquiring semaphore
        if (shuttingDown.load(std::memory_order_acquire)) {
            sem_post(&producerSem);
            return nullptr;
        }

        uint64_t headIdx = head.fetch_add(1, std::memory_order_acq_rel);
        FilterRequest& req = requests[headIdx & (QUEUE_SIZE - 1)];

        return &req;
    }

    // new request is ready for processing
    void signalEnqueued() {
        sem_post(&consumerSem);
    }

    FilterRequest* dequeue() {
        // Wait for a request
        if (sem_wait(&consumerSem) != 0) {
            if (errno == EINTR) {
                return nullptr;
            }
            return nullptr;
        }

        uint64_t tailIdx = tail.load(std::memory_order_acquire);
        FilterRequest& req = requests[tailIdx & (QUEUE_SIZE - 1)];

        return &req;
    }

    // new request can be submitted
    void signalDequeued() {
        tail.fetch_add(1, std::memory_order_release);
        sem_post(&producerSem);
    }

    [[nodiscard]] size_t pendingRequests() const {
        return head.load(std::memory_order_acquire) - tail.load(std::memory_order_acquire);
    }

    void initiateShutdown() {
        shuttingDown.store(true, std::memory_order_release);
    }

    [[nodiscard]] bool isShuttingDown() const {
        return shuttingDown.load(std::memory_order_acquire);
    }

    size_t cancelPendingRequests() {
        uint64_t currentTail = tail.load(std::memory_order_acquire);
        uint64_t currentHead = head.load(std::memory_order_acquire);
        size_t cancelled = 0;

        for (uint64_t i = currentTail; i < currentHead; i++) {
            FilterRequest& req = requests[i & (QUEUE_SIZE - 1)];
            if (!req.completed.load(std::memory_order_acquire)) {
                req.cancelled.store(true, std::memory_order_release);
                req.completed.store(true, std::memory_order_release);
                req.result = 0;
                cancelled++;
            }
        }

        return cancelled;
    }
};

template <typename Config>
class CuckooFilterIPCServer {
   private:
    CuckooFilter<Config>* filter;
    SharedRingBuffer* ring;
    int shmFd;
    std::string shmName;
    bool running;
    std::thread workerThread;

    void processRequests() {
        bool shutdownReceived = false;

        while (true) {
            // Finally break once queue is drained after shutdown
            if (shutdownReceived && ring->pendingRequests() == 0) {
                break;
            }

            FilterRequest* reqPtr = ring->dequeue();

            // check if we should continue
            if (!reqPtr) {
                if (errno == EINTR) {
                    continue;
                }
                break;
            }

            FilterRequest& req = *reqPtr;

            if (req.type == RequestType::SHUTDOWN) {
                shutdownReceived = true;
                req.completed.store(true, std::memory_order_release);
                ring->signalDequeued();
                continue;
            }

            if (req.type == RequestType::CLEAR) {
                filter->clear();
                req.result = 0;
                req.completed.store(true, std::memory_order_release);
                ring->signalDequeued();
                continue;
            }

            // Skip cancelled requests (from force shutdown)
            if (req.cancelled.load(std::memory_order_acquire)) {
                req.completed.store(true, std::memory_order_release);
                ring->signalDequeued();
                continue;
            }

            try {
                processRequest(req);
            } catch (const std::exception& e) {
                std::cerr << "Error processing request: " << e.what() << std::endl;
                req.result = 0;
            }

            req.completed.store(true, std::memory_order_release);

            ring->signalDequeued();
        }
    }

    void processRequest(FilterRequest& req) {
        using T = typename Config::KeyType;

        T* d_keys = nullptr;
        bool* d_output = nullptr;

        cudaIpcMemHandle_t zeroHandle = {0};
        bool hasKeys = memcmp(&req.keysHandle, &zeroHandle, sizeof(cudaIpcMemHandle_t)) != 0;

        if (hasKeys) {
            CUDA_CALL(
                cudaIpcOpenMemHandle((void**)&d_keys, req.keysHandle, cudaIpcMemLazyEnablePeerAccess)
            );
        }

        bool hasOutput = false;
        if (req.type == RequestType::CONTAINS || req.type == RequestType::DELETE) {
            bool handleValid =
                memcmp(&req.outputHandle, &zeroHandle, sizeof(cudaIpcMemHandle_t)) != 0;

            if (handleValid) {
                CUDA_CALL(cudaIpcOpenMemHandle(
                    (void**)&d_output, req.outputHandle, cudaIpcMemLazyEnablePeerAccess
                ));
                hasOutput = true;
            }
        }

        switch (req.type) {
            case RequestType::INSERT:
                req.result = filter->insertMany(d_keys, req.count);
                break;

            case RequestType::CONTAINS:
                filter->containsMany(d_keys, req.count, d_output);
                req.result = 0;
                break;

            case RequestType::DELETE:
                req.result = filter->deleteMany(d_keys, req.count, d_output);
                break;

            default:
                req.result = 0;
                break;
        }

        if (hasKeys) {
            CUDA_CALL(cudaIpcCloseMemHandle(d_keys));
        }
        if (hasOutput) {
            CUDA_CALL(cudaIpcCloseMemHandle(d_output));
        }
    }

   public:
    CuckooFilterIPCServer(const std::string& name, size_t capacity)
        : shmName("/cuckoo_filter_" + name), running(false) {
        filter = new CuckooFilter<Config>(capacity);

        // shared memory for ring buffer
        shmFd = shm_open(shmName.c_str(), O_CREAT | O_RDWR, 0666);
        if (shmFd == -1) {
            throw std::runtime_error("Failed to create shared memory");
        }

        if (ftruncate(shmFd, sizeof(SharedRingBuffer)) == -1) {
            close(shmFd);
            throw std::runtime_error("Failed to set shared memory size");
        }

        ring = static_cast<SharedRingBuffer*>(
            mmap(nullptr, sizeof(SharedRingBuffer), PROT_READ | PROT_WRITE, MAP_SHARED, shmFd, 0)
        );

        if (ring == MAP_FAILED) {
            close(shmFd);
            throw std::runtime_error("Failed to map shared memory");
        }

        ring->head.store(0, std::memory_order_release);
        ring->tail.store(0, std::memory_order_release);
        ring->shuttingDown.store(false, std::memory_order_release);

        sem_init(&ring->producerSem, 1, SharedRingBuffer::QUEUE_SIZE);
        sem_init(&ring->consumerSem, 1, 0);

        for (auto& request : ring->requests) {
            request.completed.store(false, std::memory_order_release);
            request.cancelled.store(false, std::memory_order_release);
        }

        ring->initialised = true;
    }

    ~CuckooFilterIPCServer() {
        stop();

        if (ring != MAP_FAILED) {
            sem_destroy(&ring->producerSem);
            sem_destroy(&ring->consumerSem);
            munmap(ring, sizeof(SharedRingBuffer));
        }

        if (shmFd != -1) {
            close(shmFd);
            shm_unlink(shmName.c_str());
        }

        delete filter;
    }

    void start() {
        if (running) {
            return;
        }

        running = true;
        workerThread = std::thread(&CuckooFilterIPCServer::processRequests, this);
    }

    void stop(bool force = false) {
        if (!running) {
            return;
        }

        running = false;

        if (force) {
            size_t cancelled = ring->cancelPendingRequests();
            if (cancelled > 0) {
                std::cout << "Force shutdown: cancelled " << cancelled << " pending requests"
                          << std::endl;
            }
        } else {
            size_t pending = ring->pendingRequests();
            if (pending > 0) {
                std::cout << "Graceful shutdown: draining " << pending << " pending requests..."
                          << std::endl;
            }
        }

        // Send shutdown request to the worker thread before
        // setting shutdown flag so the enqueue() doesn't reject it
        FilterRequest* reqPtr = ring->enqueue();
        if (reqPtr) {
            FilterRequest& req = *reqPtr;
            req.type = RequestType::SHUTDOWN;
            req.completed.store(false, std::memory_order_release);
            req.cancelled.store(false, std::memory_order_release);

            ring->signalEnqueued();
        }

        ring->initiateShutdown();

        if (workerThread.joinable()) {
            workerThread.join();
        }
    }

    CuckooFilter<Config>* getFilter() {
        return filter;
    }
};

template <typename Config>
class CuckooFilterIPCClient {
   private:
    SharedRingBuffer* ring;
    int shmFd;
    std::string shmName;
    uint64_t nextRequestId;

   public:
    using T = typename Config::KeyType;

    explicit CuckooFilterIPCClient(const std::string& name)
        : shmName("/cuckoo_filter_" + name), nextRequestId(0) {
        shmFd = shm_open(shmName.c_str(), O_RDWR, 0666);
        if (shmFd == -1) {
            throw std::runtime_error("Failed to open shared memory, is the server running?");
        }

        ring = static_cast<SharedRingBuffer*>(
            mmap(nullptr, sizeof(SharedRingBuffer), PROT_READ | PROT_WRITE, MAP_SHARED, shmFd, 0)
        );

        if (ring == MAP_FAILED) {
            close(shmFd);
            throw std::runtime_error("Failed to map shared memory");
        }

        if (!ring->initialised) {
            munmap(ring, sizeof(SharedRingBuffer));
            close(shmFd);
            throw std::runtime_error("Ring buffer not initialised, server may not be ready");
        }
    }

    ~CuckooFilterIPCClient() {
        if (ring != MAP_FAILED) {
            munmap(ring, sizeof(SharedRingBuffer));
        }
        if (shmFd != -1) {
            close(shmFd);
        }
    }

    size_t insertMany(const T* d_keys, size_t count) {
        return submitRequest(RequestType::INSERT, d_keys, count, nullptr);
    }

    void containsMany(const T* d_keys, size_t count, bool* d_output) {
        submitRequest(RequestType::CONTAINS, d_keys, count, d_output);
    }

    size_t deleteMany(const T* d_keys, size_t count, bool* d_output = nullptr) {
        return submitRequest(RequestType::DELETE, d_keys, count, d_output);
    }

    void clear() {
        submitRequest(RequestType::CLEAR, nullptr, 0, nullptr);
    }

    void requestShutdown() {
        if (ring->isShuttingDown()) {
            return;
        }

        ring->initiateShutdown();

        FilterRequest* reqPtr = ring->enqueue();
        if (reqPtr) {
            FilterRequest& req = *reqPtr;
            req.type = RequestType::SHUTDOWN;
            req.completed.store(false, std::memory_order_release);
            req.cancelled.store(false, std::memory_order_release);

            ring->signalEnqueued();

            while (!req.completed.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
        }
    }

   private:
    size_t submitRequest(RequestType type, const T* d_keys, size_t count, bool* d_output) {
        FilterRequest* reqPtr = ring->enqueue();
        if (!reqPtr) {
            throw std::runtime_error("Server is shutting down, not accepting new requests");
        }

        FilterRequest& req = *reqPtr;

        if (d_keys != nullptr) {
            CUDA_CALL(cudaIpcGetMemHandle(&req.keysHandle, const_cast<T*>(d_keys)));
        } else {
            memset(&req.keysHandle, 0, sizeof(cudaIpcMemHandle_t));
        }

        if (d_output != nullptr) {
            CUDA_CALL(cudaIpcGetMemHandle(&req.outputHandle, d_output));
        } else {
            memset(&req.outputHandle, 0, sizeof(cudaIpcMemHandle_t));
        }

        req.type = type;
        req.count = count;
        req.requestId = nextRequestId++;
        req.completed.store(false, std::memory_order_release);
        req.cancelled.store(false, std::memory_order_release);
        req.result = 0;

        ring->signalEnqueued();

        while (!req.completed.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }

        if (req.cancelled.load(std::memory_order_acquire)) {
            throw std::runtime_error("Request cancelled, server is shutting down");
        }

        return req.result;
    }
};

#ifdef CUCKOO_FILTER_HAS_THRUST
template <typename Config>
class CuckooFilterIPCClientThrust {
   private:
    CuckooFilterIPCClient<Config> client;

   public:
    using T = typename Config::KeyType;

    explicit CuckooFilterIPCClientThrust(const std::string& name) : client(name) {
    }

    size_t insertMany(const thrust::device_vector<T>& d_keys) {
        return client.insertMany(thrust::raw_pointer_cast(d_keys.data()), d_keys.size());
    }

    void
    containsMany(const thrust::device_vector<T>& d_keys, thrust::device_vector<bool>& d_output) {
        if (d_output.size() != d_keys.size()) {
            d_output.resize(d_keys.size());
        }
        client.containsMany(
            thrust::raw_pointer_cast(d_keys.data()),
            d_keys.size(),
            thrust::raw_pointer_cast(d_output.data())
        );
    }

    void
    containsMany(const thrust::device_vector<T>& d_keys, thrust::device_vector<uint8_t>& d_output) {
        if (d_output.size() != d_keys.size()) {
            d_output.resize(d_keys.size());
        }
        client.containsMany(
            thrust::raw_pointer_cast(d_keys.data()),
            d_keys.size(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data()))
        );
    }

    size_t
    deleteMany(const thrust::device_vector<T>& d_keys, thrust::device_vector<bool>& d_output) {
        if (d_output.size() != d_keys.size()) {
            d_output.resize(d_keys.size());
        }
        return client.deleteMany(
            thrust::raw_pointer_cast(d_keys.data()),
            d_keys.size(),
            thrust::raw_pointer_cast(d_output.data())
        );
    }

    size_t
    deleteMany(const thrust::device_vector<T>& d_keys, thrust::device_vector<uint8_t>& d_output) {
        if (d_output.size() != d_keys.size()) {
            d_output.resize(d_keys.size());
        }
        return client.deleteMany(
            thrust::raw_pointer_cast(d_keys.data()),
            d_keys.size(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data()))
        );
    }

    size_t deleteMany(const thrust::device_vector<T>& d_keys) {
        return client.deleteMany(thrust::raw_pointer_cast(d_keys.data()), d_keys.size(), nullptr);
    }

    void clear() {
        client.clear();
    }
};
#endif