// std::getenv is the standard portable read and PEREGRINE_ALLOW_TF32 and
// PEREGRINE_PREFLIGHT_FREE_CAP are only ever compared or parsed as a number,
// never used to build a path or a command, so MSVC's /W4 C4996
// deprecation (which steers toward getenv_s) is noise here; silence it
// TU-locally, exactly as src/cpu/feature_detect.cpp does for PEREGRINE_DISABLE_AVX2.
#define _CRT_SECURE_NO_WARNINGS

#include "cuda/context.cuh"

#include "cuda/check.cuh"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace pg::cuda {
namespace {

// The device the backend targets. Single-GPU box; multi-GPU selection is backlog.
constexpr int kDevice = 0;

// Compute-capability floor for "usable". sm_70 is the first tensor-core
// generation; everything this backend relies on (cudaMallocAsync mempools,
// cublasLt, the f32 GEMM path) is present at and above it. The reference machine
// is sm_86, comfortably above the floor.
constexpr int kMinCcMajor = 7;

// Teardown runs at interpreter exit via atexit, NOT as a static destructor. A
// static-storage Context whose destructor calls cublasDestroy/cudaStreamDestroy
// runs during static deinit, and its order against the CUDA runtime's own atexit
// handlers is unspecified: if the driver unloaded first, the destroy calls hit a
// torn-down runtime and crash or return cudaErrorCudartUnloading (the
// destruction-order trap). So teardown is a plain function registered with
// std::atexit at build time, and it must NEVER use the throwing CHECK macros --
// a throw escaping atexit terminates the process. It checks return codes by hand
// and swallows cudaErrorCudartUnloading and already-freed states: leaking a
// handle into a process that is exiting anyway is strictly better than aborting
// the interpreter. The pointers live in this TU so teardown can reach them
// without touching the static Context during deinit.
cublasLtHandle_t g_cublaslt = nullptr;
cublasHandle_t g_cublas = nullptr;
cudaStream_t g_streams[3] = {nullptr, nullptr, nullptr};

__global__ void trap_kernel() {
    __trap();
}

} // namespace

// Set true at the top of teardown() and read by free_device and the pinned-pool
// deleter (transfer.cu). After teardown destroys ctx.stream, ctx.stream is a
// DANGLING handle the static Context still names, so an Array still alive at
// finalization whose ~Array calls free_device would otherwise cudaFreeAsync on
// that destroyed stream. The flag turns that post-teardown free into a no-op: the
// mempool buffer is reclaimed by the driver at process exit anyway, so leaking it
// is correct, while freeing it on a dead stream is a use-after-teardown. Defined
// in the named namespace (not the anonymous one) so transfer.cu can read it via
// context.cuh.
std::atomic<bool> g_torn_down{false};

void teardown() {
    // First, publish that teardown has begun so any ~Array running after this
    // point (Python GC at finalization) sees g_torn_down and skips its
    // cudaFreeAsync rather than touching a stream this function is about to
    // destroy. Ordered before the destroys so there is no window where the
    // streams are gone but the flag is still false.
    g_torn_down.store(true);

    // Order: handles before streams (a handle may have work queued on its
    // stream), and every result is discarded. cudaErrorCudartUnloading here is
    // the expected benign case, not a failure. No PG_*_CHECK: see above.
    if (g_cublaslt != nullptr) {
        cublasLtDestroy(g_cublaslt);
        g_cublaslt = nullptr;
    }
    if (g_cublas != nullptr) {
        cublasDestroy(g_cublas);
        g_cublas = nullptr;
    }
    for (cudaStream_t& s : g_streams) {
        if (s != nullptr) {
            cudaStreamDestroy(s);
            s = nullptr;
        }
    }
}

void inject_fault() {
    trap_kernel<<<1, 1, 0, context().stream>>>();
    PG_CUDA_CHECK(cudaGetLastError());
}

namespace {

// Reads PEREGRINE_ALLOW_TF32 exactly once, here, so the answer is folded into the
// Context for the process lifetime and every later GEMM sees one math mode. Any
// value except null or a literal "0" counts as enabled, matching the
// PEREGRINE_DISABLE_AVX2 convention in feature_detect.cpp. Default OFF: TF32's 10-bit
// mantissa is ~830x worse abs error than DEFAULT_MATH at n=512 (measured), which
// shatters the tolerance contract, so it is opt-in only and never in headline
// numbers.
bool read_tf32_flag() {
    const char* allow = std::getenv("PEREGRINE_ALLOW_TF32");
    return allow != nullptr && std::strcmp(allow, "0") != 0;
}

// A non-negative decimal byte count, else no cap.
int64_t read_free_cap() {
    const char* v = std::getenv("PEREGRINE_PREFLIGHT_FREE_CAP");
    if (v == nullptr || *v == '\0') {
        return INT64_MAX;
    }
    char* end = nullptr;
    const long long cap = std::strtoll(v, &end, 10);
    return (*end == '\0' && cap >= 0) ? static_cast<int64_t>(cap) : INT64_MAX;
}

Context build_context() {
    Context ctx{};
    ctx.device_id = kDevice;

    PG_CUDA_CHECK(cudaSetDevice(ctx.device_id));
    PG_CUDA_CHECK(cudaGetDeviceProperties(&ctx.props, ctx.device_id));

    // One plain blocking stream for everything. Two streams made every device op
    // wait on the host twice to order an allocation or copy on one against the
    // GEMM on the other; the priority/non-blocking knobs buy nothing here. The
    // side streams serve the staged GEMM alone, which orders them against this one
    // with events rather than host waits.
    PG_CUDA_CHECK(cudaStreamCreate(&ctx.stream));
    PG_CUDA_CHECK(cudaStreamCreate(&ctx.upload_stream));
    PG_CUDA_CHECK(cudaStreamCreate(&ctx.download_stream));
    ctx.free_cap = read_free_cap();

    // The device default mempool with the release threshold raised to UINT64_MAX:
    // cudaFreeAsync returns memory to the pool instead of the OS, so the next
    // matmul reuses it rather than re-allocating (set the release threshold high so
    // repeated matmuls reuse memory). The stream-ordered allocations
    // (cudaMallocAsync) rely on this policy the pool keeps.
    PG_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&ctx.pool, ctx.device_id));
    // Non-const: cudaMemPoolSetAttribute takes a void* value pointer (it is a
    // generic getter/setter), so a const-qualified threshold fails to bind.
    std::uint64_t release_threshold = UINT64_MAX;
    PG_CUDA_CHECK(cudaMemPoolSetAttribute(
        ctx.pool, cudaMemPoolAttrReleaseThreshold, &release_threshold));

    // One cublas v2 handle, bound to the stream so its GEMMs serialize with
    // everything else on it. cublasCreate is the millisecond cost the singleton
    // exists to pay exactly once.
    PG_CUBLAS_CHECK(cublasCreate(&ctx.cublas));
    PG_CUBLAS_CHECK(cublasSetStream(ctx.cublas, ctx.stream));

    // Fold the TF32 flag once and set the handle's math mode to match here, so
    // the policy is decided at init and cannot be toggled mid-session into a
    // result that violates the tolerance contract. DEFAULT_MATH is the f32 path;
    // TF32_TENSOR_OP only when explicitly opted in. The GEMM also consults
    // ctx.tf32_enabled, but the handle already carries the decided mode.
    ctx.tf32_enabled = read_tf32_flag();
    PG_CUBLAS_CHECK(cublasSetMathMode(
        ctx.cublas,
        ctx.tf32_enabled ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));

    // cublasLt handle held for future use (batched / epilogue control). The GEMM
    // uses the v2 cublasSgemm/Dgemm, which is simpler and equally fast for plain
    // GEMM; this handle is created now so that work never pays a per-call create
    // later.
    PG_CUBLAS_CHECK(cublasLtCreate(&ctx.cublaslt));

    // Publish the raw handles for teardown and register it. atexit runs the
    // registered functions in reverse order at normal process exit, before the
    // CUDA runtime's own static teardown in the common case; in the racy case
    // teardown swallows cudaErrorCudartUnloading. Registered here, inside the
    // once-built static init, so it is registered exactly once.
    g_streams[0] = ctx.stream;
    g_streams[1] = ctx.upload_stream;
    g_streams[2] = ctx.download_stream;
    g_cublas = ctx.cublas;
    g_cublaslt = ctx.cublaslt;
    std::atexit(teardown);

    return ctx;
}

} // namespace

Context& context() {
    // Built once on first use and reused for the process lifetime (mirrors
    // src/cpu feature_detect's memoized detect()). The first device operation
    // pays the init; module import does not, honoring the no-CUDA-init-at-import
    // rule. Not const: callers use the handles and streams mutably.
    static Context the_ctx = build_context();
    return the_ctx;
}

namespace {

cuda_device_info probe_once() {
    // Never-throwing: this is what has_cuda and the test gate call to decide
    // "is a usable device here" without paying full context init. Every CUDA
    // result is inspected by hand, not via the throwing CHECK macros, so a
    // driverless or device-less machine gets a not-present verdict instead of an
    // exception. Reason strings name the specific failure mode.
    cuda_device_info info{};
    info.present = false;
    info.device_id = kDevice;

    int count = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (count_err != cudaSuccess) {
        // cudaErrorInsufficientDriver lands here: the runtime is present but the
        // installed driver is too old to run it.
        info.reason = (count_err == cudaErrorInsufficientDriver)
                          ? "driver too old"
                          : std::string("no device (") +
                                cudaGetErrorName(count_err) + ")";
        return info;
    }
    if (count <= 0) {
        info.reason = "no device";
        return info;
    }

    // Single attribute reads, not cudaGetDeviceProperties, decide presence:
    // the property query fills every field of the struct and is much slower
    // than reading the two attributes the verdict needs.
    cudaError_t attr_err = cudaDeviceGetAttribute(
        &info.cc_major, cudaDevAttrComputeCapabilityMajor, kDevice);
    if (attr_err == cudaSuccess) {
        attr_err = cudaDeviceGetAttribute(
            &info.cc_minor, cudaDevAttrComputeCapabilityMinor, kDevice);
    }
    if (attr_err != cudaSuccess) {
        info.reason =
            std::string("no device (") + cudaGetErrorName(attr_err) + ")";
        return info;
    }
    if (info.cc_major < kMinCcMajor) {
        info.reason = "compute capability too low";
        return info;
    }

    // The runtime API exposes the device name only through the property struct.
    // It keys the calibration cache, so it is paid here, once per process.
    cudaDeviceProp props{};
    const cudaError_t prop_err = cudaGetDeviceProperties(&props, kDevice);
    if (prop_err != cudaSuccess) {
        info.reason =
            std::string("no device (") + cudaGetErrorName(prop_err) + ")";
        return info;
    }
    info.name = props.name;

    info.present = true;
    return info;
}

} // namespace

cuda_device_info device_probe() {
    // Device presence and capability cannot change within a process, and the
    // runtime already caches a failed driver init, so the first verdict stands.
    // Function-local static initialization is thread-safe in C++11 and later.
    static const cuda_device_info info = probe_once();
    return info;
}

cuda_device_info context_device_info() {
    // The Python-facing entry that drives context() init and reports the bound
    // device. Probe first so a machine without a usable device returns the
    // not-present verdict WITHOUT building (and therefore without initializing
    // CUDA or registering teardown) -- calling this on a driverless box cannot
    // throw. With a usable device, force the singleton and copy its props into
    // the CUDA-free POD the binding returns.
    cuda_device_info info = device_probe();
    if (!info.present) {
        return info;
    }

    const Context& ctx = context();
    info.device_id = ctx.device_id;
    info.cc_major = ctx.props.major;
    info.cc_minor = ctx.props.minor;
    info.name = ctx.props.name;
    return info;
}

} // namespace pg::cuda
