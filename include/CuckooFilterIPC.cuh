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

/**
 * @brief Type of request that can be sent to the IPC server.
 */
enum class RequestType {
    INSERT = 0,
    CONTAINS = 1,
    DELETE = 2,
    CLEAR = 3,
    SHUTDOWN = 4
};

/**
 * @brief Structure representing a filter operation request.
 *
 * Contains all information needed to process a filter operation
 * through IPC, including memory handles for keys and results.
 */
struct FilterRequest {
    RequestType type;
    uint32_t count;                   // Number of keys in this batch
    cudaIpcMemHandle_t keysHandle;    // optional handle to device memory containing keys
    cudaIpcMemHandle_t outputHandle;  // optional handle for results (for lookup/deletion)
    uint64_t requestId;               // Unique request identifier
    std::atomic<bool> completed;      // Completion flag
    std::atomic<bool> cancelled;      // Cancellation flag (for force shutdown)
    size_t result;                    // Updated number of occupied slots after insert/delete
};

/**
 * @brief A shared memory queue implementation for Inter-Process Communication.
 *
 * This structure manages a ring buffer of requests in shared memory,
 * using semaphores for synchronization between the producer (client)
 * and consumer (server).
 */
struct SharedQueue {
    static constexpr size_t QUEUE_SIZE = 256;
    static_assert(powerOfTwo(QUEUE_SIZE), "queue size must be a power of two");

    std::atomic<uint64_t> head;  // Producer index
    std::atomic<uint64_t> tail;  // Consumer index
    sem_t producerSem;           // Semaphore for available slots
    sem_t consumerSem;           // Semaphore for pending requests

    FilterRequest requests[QUEUE_SIZE];

    std::atomic<bool> initialised;
    std::atomic<bool> shuttingDown;

    /**
     * @brief Attempts to acquire a slot in the queue for a new request.
     *
     * This function blocks until a slot is available or the server shuts down.
     *
     * @return FilterRequest* Pointer to the acquired request slot, or nullptr if shutting down.
     */
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

    /**
     * @brief Signals that a new request has been enqueued and is ready for processing.
     */
    void signalEnqueued() {
        sem_post(&consumerSem);
    }

    /**
     * @brief Dequeues the next pending request.
     *
     * This function blocks until a request is available.
     *
     * @return FilterRequest* Pointer to the request to process, or nullptr on error/interrupt.
     */
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

    /**
     * @brief Signals that a request has been processed and the slot is free.
     */
    void signalDequeued() {
        tail.fetch_add(1, std::memory_order_release);
        sem_post(&producerSem);
    }

    /**
     * @brief Returns the number of pending requests in the queue.
     * @return size_t Number of pending requests.
     */
    [[nodiscard]] size_t pendingRequests() const {
        return head.load(std::memory_order_acquire) - tail.load(std::memory_order_acquire);
    }

    /**
     * @brief Initiates the shutdown process for the queue.
     */
    void initiateShutdown() {
        shuttingDown.store(true, std::memory_order_release);
    }

    /**
     * @brief Checks if the queue is in shutdown mode.
     * @return true if shutting down, false otherwise.
     */
    [[nodiscard]] bool isShuttingDown() const {
        return shuttingDown.load(std::memory_order_acquire);
    }

    /**
     * @brief Cancels all pending requests in the queue.
     *
     * Marks all uncompleted requests as cancelled and completed.
     *
     * @return size_t Number of requests cancelled.
     */
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

/**
 * @brief Server implementation for the IPC Cuckoo Filter.
 *
 * This class manages the shared memory segment and processes requests
 * from clients. It runs a worker thread that polls the shared queue
 * and executes filter operations on the GPU.
 *
 * @tparam Config The configuration structure for the Cuckoo Filter.
 */
template <typename Config>
class CuckooFilterIPCServer {
   private:
    CuckooFilter<Config>* filter;
    SharedQueue* queue;
    int shmFd;
    std::string shmName;
    bool running;
    std::thread workerThread;

    void processRequests() {
        bool shutdownReceived = false;

        while (true) {
            // Finally break once queue is drained after shutdown
            if (shutdownReceived && queue->pendingRequests() == 0) {
                break;
            }

            FilterRequest* req = queue->dequeue();

            // check if we should continue
            if (!req) {
                if (errno == EINTR) {
                    continue;
                }
                break;
            }

            if (req->type == RequestType::SHUTDOWN) {
                shutdownReceived = true;
                req->completed.store(true, std::memory_order_release);
                queue->signalDequeued();
                continue;
            }

            if (req->type == RequestType::CLEAR) {
                filter->clear();
                req->result = 0;
                req->completed.store(true, std::memory_order_release);
                queue->signalDequeued();
                continue;
            }

            // Skip cancelled requests (from force shutdown)
            if (req->cancelled.load(std::memory_order_acquire)) {
                req->completed.store(true, std::memory_order_release);
                queue->signalDequeued();
                continue;
            }

            try {
                processRequest(req);
            } catch (const std::exception& e) {
                std::cerr << "Error processing request: " << e.what() << std::endl;
                req->result = 0;
            }

            req->completed.store(true, std::memory_order_release);

            queue->signalDequeued();
        }
    }

    void processRequest(FilterRequest* req) {
        using T = typename Config::KeyType;

        T* d_keys = nullptr;
        bool* d_output = nullptr;

        cudaIpcMemHandle_t zeroHandle = {0};
        bool hasKeys = memcmp(&req->keysHandle, &zeroHandle, sizeof(cudaIpcMemHandle_t)) != 0;

        if (hasKeys) {
            CUDA_CALL(cudaIpcOpenMemHandle(
                (void**)&d_keys, req->keysHandle, cudaIpcMemLazyEnablePeerAccess
            ));
        }

        bool hasOutput = false;
        if (req->type == RequestType::CONTAINS || req->type == RequestType::DELETE) {
            bool handleValid =
                memcmp(&req->outputHandle, &zeroHandle, sizeof(cudaIpcMemHandle_t)) != 0;

            if (handleValid) {
                CUDA_CALL(cudaIpcOpenMemHandle(
                    (void**)&d_output, req->outputHandle, cudaIpcMemLazyEnablePeerAccess
                ));
                hasOutput = true;
            }
        }

        switch (req->type) {
            case RequestType::INSERT:
                req->result = filter->insertMany(d_keys, req->count);
                break;

            case RequestType::CONTAINS:
                filter->containsMany(d_keys, req->count, d_output);
                req->result = 0;
                break;

            case RequestType::DELETE:
                req->result = filter->deleteMany(d_keys, req->count, d_output);
                break;

            default:
                req->result = 0;
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
    /**
     * @brief Constructs a new CuckooFilterIPCServer.
     *
     * Creates the shared memory segment and initializes the shared queue.
     *
     * @param name Unique name for the shared memory segment.
     * @param capacity Capacity of the Cuckoo Filter.
     */
    CuckooFilterIPCServer(const std::string& name, size_t capacity)
        : shmName("/cuckoo_filter_" + name), running(false) {
        filter = new CuckooFilter<Config>(capacity);

        // shared memory for queue
        shmFd = shm_open(shmName.c_str(), O_CREAT | O_RDWR, 0666);
        if (shmFd == -1) {
            throw std::runtime_error("Failed to create shared memory");
        }

        if (ftruncate(shmFd, sizeof(SharedQueue)) == -1) {
            close(shmFd);
            throw std::runtime_error("Failed to set shared memory size");
        }

        queue = static_cast<SharedQueue*>(
            mmap(nullptr, sizeof(SharedQueue), PROT_READ | PROT_WRITE, MAP_SHARED, shmFd, 0)
        );

        if (queue == MAP_FAILED) {
            close(shmFd);
            throw std::runtime_error("Failed to map shared memory");
        }

        queue->head.store(0, std::memory_order_release);
        queue->tail.store(0, std::memory_order_release);
        queue->shuttingDown.store(false, std::memory_order_release);

        sem_init(&queue->producerSem, 1, SharedQueue::QUEUE_SIZE);
        sem_init(&queue->consumerSem, 1, 0);

        for (auto& request : queue->requests) {
            request.completed.store(false, std::memory_order_release);
            request.cancelled.store(false, std::memory_order_release);
        }

        queue->initialised.store(true, std::memory_order_release);
    }

    /**
     * @brief Destroys the CuckooFilterIPCServer.
     *
     * Stops the server, cleans up shared memory and resources.
     */
    ~CuckooFilterIPCServer() {
        stop();

        if (queue != MAP_FAILED) {
            sem_destroy(&queue->producerSem);
            sem_destroy(&queue->consumerSem);
            munmap(queue, sizeof(SharedQueue));
        }

        if (shmFd != -1) {
            close(shmFd);
            shm_unlink(shmName.c_str());
        }

        delete filter;
    }

    /**
     * @brief Starts the worker thread to process requests.
     */
    void start() {
        if (running) {
            return;
        }

        running = true;
        workerThread = std::thread(&CuckooFilterIPCServer::processRequests, this);
    }

    /**
     * @brief Stops the server.
     *
     * @param force If true, cancels pending requests immediately. If false, waits for pending
     * requests to complete.
     */
    void stop(bool force = false) {
        if (!running) {
            return;
        }

        running = false;

        if (force) {
            size_t cancelled = queue->cancelPendingRequests();
            if (cancelled > 0) {
                std::cout << "Force shutdown: cancelled " << cancelled << " pending requests"
                          << std::endl;
            }
        } else {
            size_t pending = queue->pendingRequests();
            if (pending > 0) {
                std::cout << "Graceful shutdown: draining " << pending << " pending requests..."
                          << std::endl;
            }
        }

        // Send shutdown request to the worker thread before
        // setting shutdown flag so the enqueue() doesn't reject it
        FilterRequest* req = queue->enqueue();
        if (req) {
            req->type = RequestType::SHUTDOWN;
            req->completed.store(false, std::memory_order_release);
            req->cancelled.store(false, std::memory_order_release);

            queue->signalEnqueued();
        }

        queue->initiateShutdown();

        if (workerThread.joinable()) {
            workerThread.join();
        }
    }

    /**
     * @brief Returns a pointer to the underlying CuckooFilter instance.
     * @return CuckooFilter<Config>* Pointer to the filter.
     */
    CuckooFilter<Config>* getFilter() {
        return filter;
    }
};

/**
 * @brief Client implementation for the IPC Cuckoo Filter.
 *
 * This class connects to an existing shared memory segment created by
 * a server and allows submitting filter operations. It handles the
 * details of mapping memory handles for CUDA IPC.
 *
 * @tparam Config The configuration structure for the Cuckoo Filter.
 */
template <typename Config>
class CuckooFilterIPCClient {
   private:
    SharedQueue* queue;
    int shmFd;
    std::string shmName;
    uint64_t nextRequestId;

   public:
    using T = typename Config::KeyType;

    /**
     * @brief Constructs a new CuckooFilterIPCClient.
     *
     * Connects to the shared memory segment.
     *
     * @param name Name of the shared memory segment to connect to.
     */
    explicit CuckooFilterIPCClient(const std::string& name)
        : shmName("/cuckoo_filter_" + name), nextRequestId(0) {
        shmFd = shm_open(shmName.c_str(), O_RDWR, 0666);
        if (shmFd == -1) {
            throw std::runtime_error("Failed to open shared memory, is the server running?");
        }

        queue = static_cast<SharedQueue*>(
            mmap(nullptr, sizeof(SharedQueue), PROT_READ | PROT_WRITE, MAP_SHARED, shmFd, 0)
        );

        if (queue == MAP_FAILED) {
            close(shmFd);
            throw std::runtime_error("Failed to map shared memory");
        }

        if (!queue->initialised.load(std::memory_order_acquire)) {
            munmap(queue, sizeof(SharedQueue));
            close(shmFd);
            throw std::runtime_error("Ring buffer not initialised, server may not be ready");
        }
    }

    /**
     * @brief Destroys the CuckooFilterIPCClient.
     *
     * Unmaps the shared memory.
     */
    ~CuckooFilterIPCClient() {
        if (queue != MAP_FAILED) {
            munmap(queue, sizeof(SharedQueue));
        }
        if (shmFd != -1) {
            close(shmFd);
        }
    }

    /**
     * @brief Inserts multiple keys into the filter.
     *
     * @param d_keys Pointer to device memory containing keys.
     * @param count Number of keys to insert.
     * @return size_t Total number of occupied slots after insertion.
     */
    size_t insertMany(const T* d_keys, size_t count) {
        return submitRequest(RequestType::INSERT, d_keys, count, nullptr);
    }

    /**
     * @brief Checks for existence of multiple keys.
     *
     * @param d_keys Pointer to device memory containing keys.
     * @param count Number of keys to check.
     * @param d_output Pointer to device memory to store results (true/false).
     */
    void containsMany(const T* d_keys, size_t count, bool* d_output) {
        submitRequest(RequestType::CONTAINS, d_keys, count, d_output);
    }

    /**
     * @brief Deletes multiple keys from the filter.
     *
     * @param d_keys Pointer to device memory containing keys.
     * @param count Number of keys to delete.
     * @param d_output Optional pointer to device memory to store results (true if deleted).
     * @return size_t Total number of occupied slots after deletion.
     */
    size_t deleteMany(const T* d_keys, size_t count, bool* d_output = nullptr) {
        return submitRequest(RequestType::DELETE, d_keys, count, d_output);
    }

    /**
     * @brief Clears the filter.
     */
    void clear() {
        submitRequest(RequestType::CLEAR, nullptr, 0, nullptr);
    }

    /**
     * @brief Requests the server to shut down.
     */
    void requestShutdown() {
        if (queue->isShuttingDown()) {
            return;
        }

        queue->initiateShutdown();

        FilterRequest* req = queue->enqueue();
        if (req) {
            req->type = RequestType::SHUTDOWN;
            req->completed.store(false, std::memory_order_release);
            req->cancelled.store(false, std::memory_order_release);

            queue->signalEnqueued();

            while (!req->completed.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
        }
    }

   private:
    size_t submitRequest(RequestType type, const T* d_keys, size_t count, bool* d_output) {
        FilterRequest* req = queue->enqueue();
        if (!req) {
            throw std::runtime_error("Server is shutting down, not accepting new requests");
        }

        if (d_keys != nullptr) {
            CUDA_CALL(cudaIpcGetMemHandle(&req->keysHandle, const_cast<T*>(d_keys)));
        } else {
            memset(&req->keysHandle, 0, sizeof(cudaIpcMemHandle_t));
        }

        if (d_output != nullptr) {
            CUDA_CALL(cudaIpcGetMemHandle(&req->outputHandle, d_output));
        } else {
            memset(&req->outputHandle, 0, sizeof(cudaIpcMemHandle_t));
        }

        req->type = type;
        req->count = count;
        req->requestId = nextRequestId++;
        req->completed.store(false, std::memory_order_release);
        req->cancelled.store(false, std::memory_order_release);
        req->result = 0;

        queue->signalEnqueued();

        while (!req->completed.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }

        if (req->cancelled.load(std::memory_order_acquire)) {
            throw std::runtime_error("Request cancelled, server is shutting down");
        }

        return req->result;
    }
};

#ifdef CUCKOO_FILTER_HAS_THRUST
/**
 * @brief Thrust-compatible wrapper for the IPC Client.
 *
 * This class provides a convenient interface for using the IPC client
 * with Thrust vectors, automatically handling pointer casting and
 * vector resizing.
 *
 * @tparam Config The configuration structure for the Cuckoo Filter.
 */
template <typename Config>
class CuckooFilterIPCClientThrust {
   private:
    CuckooFilterIPCClient<Config> client;

   public:
    using T = typename Config::KeyType;

    /**
     * @brief Constructs a new CuckooFilterIPCClientThrust.
     * @param name Name of the shared memory segment.
     */
    explicit CuckooFilterIPCClientThrust(const std::string& name) : client(name) {
    }

    /**
     * @brief Inserts keys from a Thrust device vector.
     * @param d_keys Vector of keys to insert.
     * @return size_t Total number of occupied slots.
     */
    size_t insertMany(const thrust::device_vector<T>& d_keys) {
        return client.insertMany(thrust::raw_pointer_cast(d_keys.data()), d_keys.size());
    }

    /**
     * @brief Checks for existence of keys in a Thrust device vector.
     * @param d_keys Vector of keys to check.
     * @param d_output Vector to store results (bool). Resized if necessary.
     */
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

    /**
     * @brief Checks for existence of keys in a Thrust device vector (uint8_t output).
     * @param d_keys Vector of keys to check.
     * @param d_output Vector to store results (uint8_t). Resized if necessary.
     */
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

    /**
     * @brief Deletes keys in a Thrust device vector.
     * @param d_keys Vector of keys to delete.
     * @param d_output Vector to store results (bool). Resized if necessary.
     * @return size_t Total number of occupied slots.
     */
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

    /**
     * @brief Deletes keys in a Thrust device vector (uint8_t output).
     * @param d_keys Vector of keys to delete.
     * @param d_output Vector to store results (uint8_t). Resized if necessary.
     * @return size_t Total number of occupied slots.
     */
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

    /**
     * @brief Deletes keys in a Thrust device vector without outputting results.
     * @param d_keys Vector of keys to delete.
     * @return size_t Total number of occupied slots.
     */
    size_t deleteMany(const thrust::device_vector<T>& d_keys) {
        return client.deleteMany(thrust::raw_pointer_cast(d_keys.data()), d_keys.size(), nullptr);
    }

    /**
     * @brief Clears the filter.
     */
    void clear() {
        client.clear();
    }
};
#endif