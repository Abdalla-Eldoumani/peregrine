// PCIe transfer probe: rate and fixed cost of host<->device copies for pinned
// and pageable host memory, timed on the GPU's own clock with event pairs.

#include "json.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using pg_ceil::join;
using pg_ceil::json_str;
using pg_ceil::num;
using pg_ceil::obj;

void check(cudaError_t e, const char* call, int line) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(cudaGetErrorName(e)) + " from " + call +
                                 " (pcie.cu:" + std::to_string(line) + ")");
    }
}

#define PG_CHECK(x) check((x), #x, __LINE__)

constexpr std::int64_t kMaxBytes = std::int64_t{256} << 20;
constexpr std::int64_t kWarmups = 3;
constexpr std::int64_t kReps = 20;

} // namespace

std::string pcie_report() {
    const std::int64_t sizes[] = {std::int64_t{4} << 10,  std::int64_t{64} << 10,
                                  std::int64_t{1} << 20,  std::int64_t{4} << 20,
                                  std::int64_t{16} << 20, std::int64_t{64} << 20, kMaxBytes};

    PG_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    PG_CHECK(cudaGetDeviceProperties(&prop, 0));
    int runtime = 0;
    int driver = 0;
    PG_CHECK(cudaRuntimeGetVersion(&runtime));
    PG_CHECK(cudaDriverGetVersion(&driver));

    // One buffer of the largest size serves every size, which keeps pinned
    // memory at the 256 MB cap. An error path leaks these until process exit,
    // when the driver reclaims them.
    void* pinned = nullptr;
    PG_CHECK(cudaHostAlloc(&pinned, kMaxBytes, cudaHostAllocDefault));
    std::memset(pinned, 1, kMaxBytes);
    // Value-initialized so every pageable page is committed before timing.
    std::vector<unsigned char> pageable(kMaxBytes, 1);
    void* device = nullptr;
    PG_CHECK(cudaMalloc(&device, kMaxBytes));
    PG_CHECK(cudaMemset(device, 0, kMaxBytes));
    cudaStream_t stream = nullptr;
    PG_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    PG_CHECK(cudaEventCreate(&start));
    PG_CHECK(cudaEventCreate(&stop));

    struct host_kind {
        const char* name;
        void* ptr;
    };
    const host_kind hosts[] = {{"pinned", pinned}, {"pageable", pageable.data()}};
    std::vector<std::string> rows;
    std::vector<std::string> latency;
    for (const host_kind& h : hosts) {
        for (const bool h2d : {true, false}) {
            const char* dir = h2d ? "h2d" : "d2h";
            double small_median_us = 0.0;
            double large_best_gbs = 0.0;
            for (const std::int64_t bytes : sizes) {
                std::vector<double> ms;
                for (std::int64_t r = 0; r < kWarmups + kReps; ++r) {
                    PG_CHECK(cudaEventRecord(start, stream));
                    if (h2d) {
                        PG_CHECK(cudaMemcpyAsync(device, h.ptr, static_cast<std::size_t>(bytes),
                                                 cudaMemcpyHostToDevice, stream));
                    } else {
                        PG_CHECK(cudaMemcpyAsync(h.ptr, device, static_cast<std::size_t>(bytes),
                                                 cudaMemcpyDeviceToHost, stream));
                    }
                    PG_CHECK(cudaEventRecord(stop, stream));
                    PG_CHECK(cudaEventSynchronize(stop));
                    float t = 0.0f;
                    PG_CHECK(cudaEventElapsedTime(&t, start, stop));
                    if (r >= kWarmups) {
                        ms.push_back(t);
                    }
                }
                std::sort(ms.begin(), ms.end());
                const double median_ms = 0.5 * (ms[kReps / 2 - 1] + ms[kReps / 2]);
                const double best_gbs = static_cast<double>(bytes) / (ms.front() * 1e-3) / 1e9;
                const double median_gbs = static_cast<double>(bytes) / (median_ms * 1e-3) / 1e9;
                if (bytes == sizes[0]) {
                    small_median_us = median_ms * 1e3;
                }
                if (bytes == kMaxBytes) {
                    large_best_gbs = best_gbs;
                }
                rows.push_back(obj({{"host", json_str(h.name)},
                                    {"direction", json_str(dir)},
                                    {"bytes", std::to_string(bytes)},
                                    {"best_gbs", num(best_gbs)},
                                    {"median_gbs", num(median_gbs)},
                                    {"best_us", num(ms.front() * 1e3)},
                                    {"median_us", num(median_ms * 1e3)}}));
            }
            // 4 KB moves in well under a microsecond at the streaming rate, so
            // nearly all of its time is the fixed per-transfer cost.
            const double stream_us = static_cast<double>(sizes[0]) / (large_best_gbs * 1e9) * 1e6;
            latency.push_back(obj({{"host", json_str(h.name)},
                                   {"direction", json_str(dir)},
                                   {"latency_us", num(small_median_us - stream_us)}}));
        }
    }

    PG_CHECK(cudaEventDestroy(stop));
    PG_CHECK(cudaEventDestroy(start));
    PG_CHECK(cudaStreamDestroy(stream));
    PG_CHECK(cudaFree(device));
    PG_CHECK(cudaFreeHost(pinned));

    return obj({{"device", json_str(prop.name)},
                {"cuda_runtime_version", std::to_string(runtime)},
                {"cuda_driver_version", std::to_string(driver)},
                {"timing", json_str("cudaEventElapsedTime around one cudaMemcpyAsync on one "
                                    "non-blocking stream")},
                {"gb", json_str("1e9 bytes")},
                {"warmups", std::to_string(kWarmups)},
                {"reps", std::to_string(kReps)},
                {"latency_method", json_str("median 4 KB time minus 4 KB at the best 256 MB rate")},
                {"transfers", join(rows)},
                {"latency", join(latency)}});
}
