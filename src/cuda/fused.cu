#include "cuda/fused.cuh"

#include "cuda/check.cuh"
#include "cuda/context.cuh"

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace fme::cuda {
namespace {

// RAII owners for the chain timer's device scratch and cudaEvents. time_fused_chain
// wraps every allocation, record, sync, and elapsed-time query in a throwing
// FME_CUDA_CHECK; without an owner a throw unwinds past the trailing
// cudaFreeAsync/cudaEventDestroy and leaks the scratch into the context mempool
// (release threshold UINT64_MAX -- never returned to the OS) and leaks the events.
// The destructors run on the unwind path, so they discard the return code the way
// free_device does and clear the sticky error rather than using the throwing CHECK.
struct device_buf {
    void* ptr = nullptr;
    cudaStream_t stream = nullptr;
    device_buf() = default;
    device_buf(void* p, cudaStream_t s) : ptr(p), stream(s) {}
    ~device_buf() {
        if (ptr != nullptr) {
            cudaFreeAsync(ptr, stream);
            cudaGetLastError();
        }
    }
    device_buf(const device_buf&) = delete;
    device_buf& operator=(const device_buf&) = delete;
};

struct cuda_event {
    cudaEvent_t ev = nullptr;
    ~cuda_event() {
        if (ev != nullptr) {
            cudaEventDestroy(ev);
            cudaGetLastError();
        }
    }
    cuda_event() = default;
    cuda_event(const cuda_event&) = delete;
    cuda_event& operator=(const cuda_event&) = delete;
};

// One templated grid-stride kernel + one functor per op (the shared-skeleton
// recommendation, RESEARCH Alternatives). Each functor is the per-element
// arithmetic; the kernel owns the float4-prefix / scalar-tail traversal so the
// tail-safety logic lives in exactly one place.
//
// float4 (f32) / double2 (f64) vectorized loads/stores are used ONLY on the
// n & ~3 (f32) / n & ~1 (f64) aligned prefix: a vector load past the last full
// pack, or on a misaligned base pointer, is an out-of-bounds device access (the
// float4 odd-tail trap, RESEARCH Pitfall 4). cudaMallocAsync allocations are
// 16-byte aligned (so the f32 float4 and f64 double2 bases are aligned for a
// full-array op), but the ELEMENT COUNT need not be a multiple of the pack width,
// so the n % width tail always falls to a scalar grid-stride loop. The vector
// width is the largest 16-byte pack for the dtype: 4 floats or 2 doubles.

struct AxpbyOp {
    float a;
    float b;
    __device__ float operator()(float x, float y) const { return a * x + b * y; }
};
struct AxpbyOpD {
    double a;
    double b;
    __device__ double operator()(double x, double y) const { return a * x + b * y; }
};

struct Fma3Op {
    // Single rounding: fmaf is the device true FMA, so inf*0 + z = NaN and the
    // result can be closer to truth than the unfused two-rounding x*y + z, exactly
    // like the CPU _mm256_fmadd path. The oracle's two-sided tolerance allows it.
    __device__ float operator()(float x, float y, float z) const {
        return fmaf(x, y, z);
    }
};
struct Fma3OpD {
    __device__ double operator()(double x, double y, double z) const {
        return fma(x, y, z);
    }
};

// scaled_relu must propagate NaN exactly like np.maximum: the device fmaxf/fmax
// follow IEEE maxNum (NaN-quieting), so fmaxf(NaN, 0) returns 0 -- the same trap
// the CPU bare _mm256_max has. Restore the NaN per element (isnan(v) ? v : max),
// so it is correct in BOTH the vectorized and the scalar path.
struct ScaledReluOp {
    float scale;
    __device__ float operator()(float x) const {
        const float v = scale * x;
        return isnan(v) ? v : fmaxf(v, 0.0f);
    }
};
struct ScaledReluOpD {
    double scale;
    __device__ double operator()(double x) const {
        const double v = scale * x;
        return isnan(v) ? v : fmax(v, 0.0);
    }
};

// vec4: the 16-byte vector type for a dtype (float4 for float, double2 for
// double -- both 16 bytes, the widest aligned device transaction). Used to load
// and store the aligned prefix one pack per thread-step.
template <typename T>
struct vec4_traits;
template <>
struct vec4_traits<float> {
    using type = float4;
    static constexpr int width = 4;
};
template <>
struct vec4_traits<double> {
    using type = double2;
    static constexpr int width = 2;
};

// Apply a unary op pack-wise (scaled_relu).
template <typename T, class Op>
__device__ typename vec4_traits<T>::type apply_pack(
    typename vec4_traits<T>::type vx, Op op);

template <>
__device__ float4 apply_pack<float>(float4 vx, ScaledReluOp op) {
    return make_float4(op(vx.x), op(vx.y), op(vx.z), op(vx.w));
}
template <>
__device__ double2 apply_pack<double>(double2 vx, ScaledReluOpD op) {
    return make_double2(op(vx.x), op(vx.y));
}

// Unary grid-stride kernel (scaled_relu): float4/double2 over the aligned prefix,
// scalar grid-stride for the [prefix, n) tail. nvec is the number of full packs
// (n / width); prefix = nvec * width is the first tail index.
template <typename T, class Op>
__global__ void fused_unary_kernel(const T* x, T* out, int64_t n, int64_t nvec,
                                   Op op) {
    using V = typename vec4_traits<T>::type;
    constexpr int W = vec4_traits<T>::width;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t i0 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const V* xv = reinterpret_cast<const V*>(x);
    V* outv = reinterpret_cast<V*>(out);
    for (int64_t v = i0; v < nvec; v += stride) {
        outv[v] = apply_pack<T>(xv[v], op);
    }
    const int64_t prefix = nvec * W;
    for (int64_t i = prefix + i0; i < n; i += stride) {
        out[i] = op(x[i]);
    }
}

template <typename T, class Op>
__device__ typename vec4_traits<T>::type apply_pack2(
    typename vec4_traits<T>::type vx, typename vec4_traits<T>::type vy, Op op);

template <>
__device__ float4 apply_pack2<float>(float4 vx, float4 vy, AxpbyOp op) {
    return make_float4(op(vx.x, vy.x), op(vx.y, vy.y), op(vx.z, vy.z),
                       op(vx.w, vy.w));
}
template <>
__device__ double2 apply_pack2<double>(double2 vx, double2 vy, AxpbyOpD op) {
    return make_double2(op(vx.x, vy.x), op(vx.y, vy.y));
}

// Binary grid-stride kernel (axpby).
template <typename T, class Op>
__global__ void fused_binary_kernel(const T* x, const T* y, T* out, int64_t n,
                                    int64_t nvec, Op op) {
    using V = typename vec4_traits<T>::type;
    constexpr int W = vec4_traits<T>::width;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t i0 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const V* xv = reinterpret_cast<const V*>(x);
    const V* yv = reinterpret_cast<const V*>(y);
    V* outv = reinterpret_cast<V*>(out);
    for (int64_t v = i0; v < nvec; v += stride) {
        outv[v] = apply_pack2<T>(xv[v], yv[v], op);
    }
    const int64_t prefix = nvec * W;
    for (int64_t i = prefix + i0; i < n; i += stride) {
        out[i] = op(x[i], y[i]);
    }
}

template <typename T, class Op>
__device__ typename vec4_traits<T>::type apply_pack3(
    typename vec4_traits<T>::type vx, typename vec4_traits<T>::type vy,
    typename vec4_traits<T>::type vz, Op op);

template <>
__device__ float4 apply_pack3<float>(float4 vx, float4 vy, float4 vz, Fma3Op op) {
    return make_float4(op(vx.x, vy.x, vz.x), op(vx.y, vy.y, vz.y),
                       op(vx.z, vy.z, vz.z), op(vx.w, vy.w, vz.w));
}
template <>
__device__ double2 apply_pack3<double>(double2 vx, double2 vy, double2 vz,
                                       Fma3OpD op) {
    return make_double2(op(vx.x, vy.x, vz.x), op(vx.y, vy.y, vz.y));
}

// Ternary grid-stride kernel (fma3).
template <typename T, class Op>
__global__ void fused_ternary_kernel(const T* x, const T* y, const T* z, T* out,
                                     int64_t n, int64_t nvec, Op op) {
    using V = typename vec4_traits<T>::type;
    constexpr int W = vec4_traits<T>::width;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t i0 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const V* xv = reinterpret_cast<const V*>(x);
    const V* yv = reinterpret_cast<const V*>(y);
    const V* zv = reinterpret_cast<const V*>(z);
    V* outv = reinterpret_cast<V*>(out);
    for (int64_t v = i0; v < nvec; v += stride) {
        outv[v] = apply_pack3<T>(xv[v], yv[v], zv[v], op);
    }
    const int64_t prefix = nvec * W;
    for (int64_t i = prefix + i0; i < n; i += stride) {
        out[i] = op(x[i], y[i], z[i]);
    }
}

// Block count: 4-8 blocks/SM x 256 threads, capped at the elements actually
// present (one pack per thread-step is enough). threads=256, blocks =
// min((n+255)/256, 8 * SMs) per the cuda-sm86 occupancy notes. GA106 has 30 SMs
// so the cap is 240 blocks; for small n the (n+255)/256 term keeps the launch
// from over-subscribing.
int launch_blocks(int64_t n) {
    const Context& ctx = context();
    const int64_t want = (n + 255) / 256;
    const int64_t cap = static_cast<int64_t>(8) * ctx.props.multiProcessorCount;
    return static_cast<int>(std::min<int64_t>(std::max<int64_t>(want, 1), cap));
}

// The aligned-prefix pack count for a dtype: nvec full packs, prefix = nvec*W
// scalar-tail boundary. A misaligned base would force nvec=0 (scalar everywhere),
// but device buffers from cudaMallocAsync are 16-byte aligned, so a full-array op
// always vectorizes the prefix; the binding never hands a misaligned sub-view.
template <typename T>
int64_t pack_count(int64_t n) {
    return n / vec4_traits<T>::width;
}

} // namespace

template <typename T>
void fused_axpby(const T* x, const T* y, T* out, int64_t n, T a, T b) {
    if (n == 0) {
        return; // zero-dim guard FIRST: no launch, matching gemm + NumPy.
    }
    const int threads = 256;
    const int blocks = launch_blocks(n);
    const int64_t nvec = pack_count<T>(n);
    if constexpr (std::is_same_v<T, float>) {
        AxpbyOp op{a, b};
        fused_binary_kernel<float, AxpbyOp>
            <<<blocks, threads, 0, context().compute>>>(x, y, out, n, nvec, op);
    } else {
        AxpbyOpD op{a, b};
        fused_binary_kernel<double, AxpbyOpD>
            <<<blocks, threads, 0, context().compute>>>(x, y, out, n, nvec, op);
    }
    FME_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void fused_fma3(const T* x, const T* y, const T* z, T* out, int64_t n) {
    if (n == 0) {
        return;
    }
    const int threads = 256;
    const int blocks = launch_blocks(n);
    const int64_t nvec = pack_count<T>(n);
    if constexpr (std::is_same_v<T, float>) {
        Fma3Op op{};
        fused_ternary_kernel<float, Fma3Op>
            <<<blocks, threads, 0, context().compute>>>(x, y, z, out, n, nvec, op);
    } else {
        Fma3OpD op{};
        fused_ternary_kernel<double, Fma3OpD>
            <<<blocks, threads, 0, context().compute>>>(x, y, z, out, n, nvec, op);
    }
    FME_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void fused_scaled_relu(const T* x, T* out, int64_t n, T scale) {
    if (n == 0) {
        return;
    }
    const int threads = 256;
    const int blocks = launch_blocks(n);
    const int64_t nvec = pack_count<T>(n);
    if constexpr (std::is_same_v<T, float>) {
        ScaledReluOp op{scale};
        fused_unary_kernel<float, ScaledReluOp>
            <<<blocks, threads, 0, context().compute>>>(x, out, n, nvec, op);
    } else {
        ScaledReluOpD op{scale};
        fused_unary_kernel<double, ScaledReluOpD>
            <<<blocks, threads, 0, context().compute>>>(x, out, n, nvec, op);
    }
    FME_CUDA_CHECK(cudaGetLastError());
}

template void fused_axpby<float>(const float*, const float*, float*, int64_t, float, float);
template void fused_axpby<double>(const double*, const double*, double*, int64_t, double, double);
template void fused_fma3<float>(const float*, const float*, const float*, float*, int64_t);
template void fused_fma3<double>(const double*, const double*, const double*, double*, int64_t);
template void fused_scaled_relu<float>(const float*, float*, int64_t, float);
template void fused_scaled_relu<double>(const double*, double*, int64_t, double);

template <typename T>
float time_fused_chain(const T* x, const T* y, const T* z, int64_t n, int reps,
                       int warmups) {
    // Clone the time_matmul SHAPE (reject reps<=0/n==0, alloc scratch on the
    // compute stream, events, warmups + sync, timed reps, elapsed, free), but the
    // timed body is the 3-op CHAIN, not one op. Two scratch buffers (t = the
    // axpby/fma3 intermediate, out = the chain result) are allocated once outside
    // the timed window so no allocation lands inside it; the chain runs
    // transfer-free on the compute stream. Fixed a/b/scale so the timed chain
    // matches what plan 05 benches; correctness is the @gpu oracle test's job.
    if (reps <= 0) {
        throw ::fme::cuda_error("cuda time_fused_chain: reps must be positive");
    }
    if (n == 0) {
        throw ::fme::cuda_error(
            "cuda time_fused_chain: empty input has no work to time");
    }

    Context& ctx = context();
    const cudaStream_t stream = ctx.compute;
    const size_t bytes = static_cast<size_t>(n) * sizeof(T);

    void* t_raw = nullptr;
    void* out_raw = nullptr;
    FME_CUDA_CHECK(cudaMallocAsync(&t_raw, bytes, stream));
    device_buf t_buf{t_raw, stream};
    FME_CUDA_CHECK(cudaMallocAsync(&out_raw, bytes, stream));
    device_buf out_buf{out_raw, stream};
    T* t = static_cast<T*>(t_buf.ptr);
    T* out = static_cast<T*>(out_buf.ptr);

    cuda_event e0;
    cuda_event e1;
    FME_CUDA_CHECK(cudaEventCreate(&e0.ev));
    FME_CUDA_CHECK(cudaEventCreate(&e1.ev));

    const T a = static_cast<T>(2);
    const T b = static_cast<T>(3);
    const T scale = static_cast<T>(1);

    // One chain iteration: t = axpby(x, y, a, b); t = fma3(t, y, z) = t*y + z;
    // out = scaled_relu(t, scale). Three launches on the compute stream, each
    // CHECK'd. fma3 is the true 3-operand x*y + z, so the middle step reuses y and
    // z as the multiply and add operands -- this is the same axpby -> fma3 ->
    // scaled_relu composition plan 05 benches, and the in-place t (read and write
    // the same buffer) is safe because the op is elementwise (no cross-element
    // dependency, each thread reads then writes its own index).
    auto chain = [&]() {
        fused_axpby<T>(x, y, t, n, a, b);
        fused_fma3<T>(t, y, z, t, n);
        fused_scaled_relu<T>(t, out, n, scale);
    };

    for (int i = 0; i < warmups; ++i) {
        chain();
    }

    FME_CUDA_CHECK(cudaStreamSynchronize(stream));
    FME_CUDA_CHECK(cudaEventRecord(e0.ev, stream));
    for (int i = 0; i < reps; ++i) {
        chain();
    }
    FME_CUDA_CHECK(cudaEventRecord(e1.ev, stream));
    FME_CUDA_CHECK(cudaEventSynchronize(e1.ev));

    float ms = 0.0f;
    FME_CUDA_CHECK(cudaEventElapsedTime(&ms, e0.ev, e1.ev));

    // t_buf, out_buf, e0, e1 are released by their owners on return; ms is read
    // into the return value before any destructor runs.
    return ms / static_cast<float>(reps);
}

template float time_fused_chain<float>(const float*, const float*, const float*, int64_t, int, int);
template float time_fused_chain<double>(const double*, const double*, const double*, int64_t, int, int);

} // namespace fme::cuda
