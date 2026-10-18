// Dev-only probe of this machine's hardware ceilings: FMA peak, DRAM
// bandwidth, and, when built with CUDA, PCIe transfer rates. It runs fixed
// kernels on hand-pinned std::threads so the numbers describe the hardware,
// not an OpenMP runtime or the library's dispatch.

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <pdh.h>
#include <pdhmsg.h>
#else
#include <pthread.h>
#include <sched.h>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#include <immintrin.h>

#include "json.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <format>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#define PG_NOINLINE __declspec(noinline)
#else
// noipa rather than noinline: GCC would otherwise clone a kernel with its
// constant arguments propagated, which can turn register operands into memory
// operands inside the timed loop.
#define PG_NOINLINE __attribute__((noipa))
#endif

#if defined(PG_CEILINGS_HAS_CUDA)
std::string pcie_report(); // pcie.cu
#endif

namespace {

using pg_ceil::join;
using pg_ceil::json_str;
using pg_ceil::num;
using pg_ceil::obj;
using clock_type = std::chrono::steady_clock;

constexpr const char* kToolVersion = "1.0.0";
constexpr const char* kUsage =
    "usage: pg_ceilings [options]\n"
    "  --threads LIST   comma-separated worker counts, one worker per physical core\n"
    "                   (default 1,<physical cores>)\n"
    "  --all-threads    every count from 1 to the physical core count\n"
    "  --seconds S      FMA timing window per run (default 2.0)\n"
    "  --warmup S       FMA warmup before each window (default 0.5)\n"
    "  --mem-mb N       size of each bandwidth array in MiB (default 256)\n"
    "  --trials N       bandwidth trials per kernel (default 10)\n"
    "  --out PATH       also write the JSON report to PATH\n"
    "  --skip-fma  --skip-mem  --skip-pcie\n";

// Written once per measurement after its window closes, so the compiler must
// keep every accumulator and loaded value live.
volatile double g_sink = 0.0;

double secs(clock_type::duration d) { return std::chrono::duration<double>(d).count(); }

std::string boolean(bool b) { return b ? "true" : "false"; }

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::int64_t n = static_cast<std::int64_t>(v.size());
    return n % 2 != 0 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ---------------------------------------------------------------- topology

void cpuid(unsigned leaf, unsigned r[4]) {
#if defined(_MSC_VER)
    int t[4];
    __cpuid(t, static_cast<int>(leaf));
    std::memcpy(r, t, sizeof t);
#else
    __cpuid(leaf, r[0], r[1], r[2], r[3]);
#endif
}

std::string cpu_brand() {
    unsigned r[4];
    cpuid(0x80000000u, r);
    if (r[0] < 0x80000004u) {
        return "unknown";
    }
    std::string brand;
    for (unsigned leaf = 0x80000002u; leaf <= 0x80000004u; ++leaf) {
        cpuid(leaf, r);
        brand.append(reinterpret_cast<const char*>(r), sizeof r);
    }
    brand.resize(brand.find('\0') == std::string::npos ? brand.size() : brand.find('\0'));
    const auto first = brand.find_first_not_of(' ');
    const auto last = brand.find_last_not_of(' ');
    return first == std::string::npos ? "unknown" : brand.substr(first, last - first + 1);
}

struct core_slot {
    int group;    // processor group; always 0 on Linux
    int logical;  // first logical processor of the core, numbered within its group
    int siblings; // logical processors sharing the core
};

struct topology {
    std::vector<core_slot> cores;
    std::int64_t logical_total = 0;
};

topology probe_topology() {
    topology t;
#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<unsigned char> buf(len);
    if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &len)) {
        throw std::runtime_error(
            std::format("GetLogicalProcessorInformationEx failed: {}", GetLastError()));
    }
    for (DWORD off = 0; off < len;) {
        const auto* info =
            reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
        // A core never spans processor groups, so GroupMask[0] holds all of it.
        const GROUP_AFFINITY& g = info->Processor.GroupMask[0];
        unsigned long first = 0;
        _BitScanForward64(&first, g.Mask);
        const int count = static_cast<int>(__popcnt64(g.Mask));
        t.cores.push_back({static_cast<int>(g.Group), static_cast<int>(first), count});
        t.logical_total += count;
        off += info->Size;
    }
#else
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof allowed, &allowed) != 0) {
        throw std::runtime_error("sched_getaffinity failed");
    }
    std::map<std::pair<int, int>, std::int64_t> slot_of; // (package, core_id) -> index
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) {
            continue;
        }
        const auto topo_value = [cpu](const char* leaf) {
            std::ifstream f(std::format("/sys/devices/system/cpu/cpu{}/topology/{}", cpu, leaf));
            // A missing file yields a value unique to this cpu, so an unreadable
            // topology degrades to one core per logical cpu instead of merging.
            int v = -1 - cpu;
            f >> v;
            return v;
        };
        const auto key = std::make_pair(topo_value("physical_package_id"), topo_value("core_id"));
        const auto [it, fresh] = slot_of.try_emplace(key, static_cast<std::int64_t>(t.cores.size()));
        if (fresh) {
            t.cores.push_back({0, cpu, 0});
        }
        ++t.cores[it->second].siblings;
        ++t.logical_total;
    }
#endif
    if (t.cores.empty()) {
        throw std::runtime_error("no physical cores found");
    }
    return t;
}

bool pin_current_thread(const core_slot& c) {
#if defined(_WIN32)
    // SetThreadGroupAffinity, not SetThreadAffinityMask: the mask form can only
    // move a thread within its current group, so on a multi-group machine it
    // would pin to the wrong processor. On one group the two are equivalent.
    GROUP_AFFINITY ga{};
    ga.Mask = KAFFINITY{1} << c.logical;
    ga.Group = static_cast<WORD>(c.group);
    return SetThreadGroupAffinity(GetCurrentThread(), &ga, nullptr) != 0;
#else
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(c.logical, &s);
    return pthread_setaffinity_np(pthread_self(), sizeof s, &s) == 0;
#endif
}

int current_cpu() {
#if defined(_WIN32)
    PROCESSOR_NUMBER pn{};
    GetCurrentProcessorNumberEx(&pn);
    return pn.Number;
#else
    return sched_getcpu();
#endif
}

// ---------------------------------------------------------------- workers

// One pinned worker per physical core. Idle workers spin on a generation
// counter instead of sleeping so a release reaches every core within
// nanoseconds; that keeps per-trial timing on the coordinator honest.
class pool {
public:
    explicit pool(std::vector<core_slot> cores)
        : cores_(std::move(cores)), observed_(cores_.size(), -1) {
        for (std::int64_t i = 0; i < size(); ++i) {
            threads_.emplace_back([this, i] { worker(i); });
        }
        wait();
    }
    ~pool() {
        quit_ = true;
        gen_.fetch_add(1, std::memory_order_release);
        for (std::thread& t : threads_) {
            t.join();
        }
    }
    pool(const pool&) = delete;
    pool& operator=(const pool&) = delete;

    std::int64_t size() const { return static_cast<std::int64_t>(cores_.size()); }
    const core_slot& core(std::int64_t i) const { return cores_[i]; }
    int observed(std::int64_t i) const { return observed_[i]; } // -1 when pinning failed

    void start(std::function<void(std::int64_t)> job) {
        job_ = std::move(job);
        done_.store(0, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);
    }
    void wait() const {
        while (done_.load(std::memory_order_acquire) < size()) {
            _mm_pause();
        }
    }
    // Seconds from releasing the workers until the last one finishes.
    double run(std::function<void(std::int64_t)> job) {
        const auto t0 = clock_type::now();
        start(std::move(job));
        wait();
        return secs(clock_type::now() - t0);
    }

private:
    void worker(std::int64_t i) {
        if (pin_current_thread(cores_[i])) {
            std::this_thread::yield();
            observed_[i] = current_cpu();
        }
        done_.fetch_add(1, std::memory_order_release);
        std::int64_t seen = 0;
        for (;;) {
            std::int64_t g = 0;
            while ((g = gen_.load(std::memory_order_acquire)) == seen) {
                _mm_pause();
            }
            seen = g;
            if (quit_) {
                return;
            }
            job_(i);
            done_.fetch_add(1, std::memory_order_release);
        }
    }

    std::vector<core_slot> cores_;
    std::vector<int> observed_;
    std::function<void(std::int64_t)> job_;
    bool quit_ = false; // published to workers by the release on gen_
    std::atomic<std::int64_t> gen_{0};
    std::atomic<std::int64_t> done_{0};
    std::vector<std::thread> threads_;
};

// ---------------------------------------------------------------- PDH

class pdh_window {
public:
#if defined(_WIN32)
    pdh_window() {
        status_ = PdhOpenQueryW(nullptr, 0, &query_);
        if (status_ == ERROR_SUCCESS) {
            status_ = PdhAddEnglishCounterW(
                query_, L"\\Processor Information(_Total)\\% Processor Performance", 0, &perf_);
        }
        if (status_ == ERROR_SUCCESS) {
            status_ = PdhAddEnglishCounterW(
                query_, L"\\Processor Information(_Total)\\% Processor Time", 0, &time_);
        }
    }
    ~pdh_window() {
        if (query_ != nullptr) {
            PdhCloseQuery(query_);
        }
    }
    pdh_window(const pdh_window&) = delete;
    pdh_window& operator=(const pdh_window&) = delete;

    // Called at window start and end; both counters are rates, so the
    // formatted value is the average between the last two samples.
    void sample() {
        if (status_ == ERROR_SUCCESS) {
            status_ = PdhCollectQueryData(query_);
        }
    }
    std::string report() const {
        if (status_ != ERROR_SUCCESS) {
            return obj({{"error", json_str(std::format("PDH status 0x{:08X}",
                                                       static_cast<unsigned>(status_)))}});
        }
        return obj({{"processor_performance_pct", value(perf_)},
                    {"processor_time_pct", value(time_)}});
    }

private:
    static std::string value(PDH_HCOUNTER c) {
        PDH_FMT_COUNTERVALUE v{};
        const PDH_STATUS s =
            PdhGetFormattedCounterValue(c, PDH_FMT_DOUBLE | PDH_FMT_NOCAP100, nullptr, &v);
        const bool valid = s == ERROR_SUCCESS &&
                           (v.CStatus == PDH_CSTATUS_VALID_DATA || v.CStatus == PDH_CSTATUS_NEW_DATA);
        return valid ? num(v.doubleValue) : "null";
    }

    PDH_HQUERY query_ = nullptr;
    PDH_HCOUNTER perf_ = nullptr;
    PDH_HCOUNTER time_ = nullptr;
    PDH_STATUS status_ = ERROR_SUCCESS;
#else
    void sample() {}
    std::string report() const { return "\"unavailable\""; }
#endif
};

// ---------------------------------------------------------------- options

struct options {
    std::vector<std::int64_t> threads;
    double seconds = 2.0;
    double warmup = 0.5;
    std::int64_t mem_mb = 256;
    std::int64_t trials = 10;
    std::string out;
    bool skip_fma = false;
    bool skip_mem = false;
    bool skip_pcie = false;
    bool help = false;
};

template <typename T>
T to_num(std::string_view flag, std::string_view v) {
    T x{};
    const auto [end, ec] = std::from_chars(v.data(), v.data() + v.size(), x);
    if (ec != std::errc{} || end != v.data() + v.size()) {
        throw std::runtime_error(std::format("bad value '{}' for {}", v, flag));
    }
    return x;
}

options parse(int argc, char** argv, std::int64_t cores) {
    options o;
    o.threads = cores > 1 ? std::vector<std::int64_t>{1, cores} : std::vector<std::int64_t>{1};
    bool all = false;
    for (std::int64_t i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        const auto value = [&]() -> std::string_view {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::format("{} needs a value", a));
            }
            return argv[++i];
        };
        if (a == "--threads") {
            o.threads.clear();
            std::string_view list = value();
            for (std::size_t pos = 0; pos <= list.size();) {
                const std::size_t comma = std::min(list.find(',', pos), list.size());
                o.threads.push_back(to_num<std::int64_t>(a, list.substr(pos, comma - pos)));
                pos = comma + 1;
            }
        } else if (a == "--all-threads") {
            all = true;
        } else if (a == "--seconds") {
            o.seconds = to_num<double>(a, value());
        } else if (a == "--warmup") {
            o.warmup = to_num<double>(a, value());
        } else if (a == "--mem-mb") {
            o.mem_mb = to_num<std::int64_t>(a, value());
        } else if (a == "--trials") {
            o.trials = to_num<std::int64_t>(a, value());
        } else if (a == "--out") {
            o.out = value();
        } else if (a == "--skip-fma") {
            o.skip_fma = true;
        } else if (a == "--skip-mem") {
            o.skip_mem = true;
        } else if (a == "--skip-pcie") {
            o.skip_pcie = true;
        } else if (a == "--help" || a == "-h") {
            o.help = true;
        } else {
            throw std::runtime_error(std::format("unknown argument {}", a));
        }
    }
    if (all) {
        o.threads.clear();
        for (std::int64_t n = 1; n <= cores; ++n) {
            o.threads.push_back(n);
        }
    }
    for (const std::int64_t n : o.threads) {
        if (n < 1 || n > cores) {
            throw std::runtime_error(
                std::format("thread count {} is outside 1..{} physical cores", n, cores));
        }
    }
    if (!(o.seconds > 0.0) || !(o.warmup >= 0.0) || o.mem_mb < 1 || o.trials < 1) {
        throw std::runtime_error("--seconds must be > 0, --warmup >= 0, --mem-mb and --trials >= 1");
    }
    return o;
}

// ---------------------------------------------------------------- FMA peak

struct f32_ops {
    using T = float;
    using V = __m256;
    static constexpr std::int64_t lanes = 8;
    static V set1(T x) { return _mm256_set1_ps(x); }
    static V load(const T* p) { return _mm256_load_ps(p); }
    static void store(T* p, V v) { _mm256_store_ps(p, v); }
    static V fma(V a, V b, V c) { return _mm256_fmadd_ps(a, b, c); }
};

struct f64_ops {
    using T = double;
    using V = __m256d;
    static constexpr std::int64_t lanes = 4;
    static V set1(T x) { return _mm256_set1_pd(x); }
    static V load(const T* p) { return _mm256_load_pd(p); }
    static void store(T* p, V v) { _mm256_store_pd(p, v); }
    static V fma(V a, V b, V c) { return _mm256_fmadd_pd(a, b, c); }
};

// FMA latency is 4 cycles on 2 ports, so at least 8 independent chains must be
// in flight to saturate them. 12 accumulators plus the 2 operands use 14 of the
// 16 ymm registers, which leaves the loop free of spills.
constexpr std::int64_t kChains = 12;
constexpr std::int64_t kFmaPorts = 2;
// About 50k cycles per call (6 cycles per iteration at 2 FMAs per cycle). The
// phase flag is polled between calls, so each window edge is off by at most
// one call, roughly 12 microseconds.
constexpr std::int64_t kBlockIters = std::int64_t{1} << 13;

// acc = acc * mul + add with mul < 1 converges to add / (1 - mul), so values
// stay normal forever: no overflow, and no denormal slow path.
template <typename Ops>
PG_NOINLINE void fma_block(typename Ops::T* s, typename Ops::T mul, typename Ops::T add,
                           std::int64_t iters) {
    using V = typename Ops::V;
    constexpr std::int64_t L = Ops::lanes;
    const V m = Ops::set1(mul);
    const V c = Ops::set1(add);
    V a0 = Ops::load(s + 0 * L), a1 = Ops::load(s + 1 * L), a2 = Ops::load(s + 2 * L);
    V a3 = Ops::load(s + 3 * L), a4 = Ops::load(s + 4 * L), a5 = Ops::load(s + 5 * L);
    V a6 = Ops::load(s + 6 * L), a7 = Ops::load(s + 7 * L), a8 = Ops::load(s + 8 * L);
    V a9 = Ops::load(s + 9 * L), a10 = Ops::load(s + 10 * L), a11 = Ops::load(s + 11 * L);
    for (std::int64_t i = 0; i < iters; ++i) {
        a0 = Ops::fma(a0, m, c);
        a1 = Ops::fma(a1, m, c);
        a2 = Ops::fma(a2, m, c);
        a3 = Ops::fma(a3, m, c);
        a4 = Ops::fma(a4, m, c);
        a5 = Ops::fma(a5, m, c);
        a6 = Ops::fma(a6, m, c);
        a7 = Ops::fma(a7, m, c);
        a8 = Ops::fma(a8, m, c);
        a9 = Ops::fma(a9, m, c);
        a10 = Ops::fma(a10, m, c);
        a11 = Ops::fma(a11, m, c);
    }
    Ops::store(s + 0 * L, a0);
    Ops::store(s + 1 * L, a1);
    Ops::store(s + 2 * L, a2);
    Ops::store(s + 3 * L, a3);
    Ops::store(s + 4 * L, a4);
    Ops::store(s + 5 * L, a5);
    Ops::store(s + 6 * L, a6);
    Ops::store(s + 7 * L, a7);
    Ops::store(s + 8 * L, a8);
    Ops::store(s + 9 * L, a9);
    Ops::store(s + 10 * L, a10);
    Ops::store(s + 11 * L, a11);
}

template <typename Ops>
std::string run_fma(pool& p, const options& o, pdh_window& pdh, const char* dtype) {
    using T = typename Ops::T;
    enum : int { idle, warmup, measure, stop };
    const std::int64_t n = p.size();
    std::atomic<int> phase{idle};
    std::atomic<std::int64_t> ready{0};
    std::vector<std::int64_t> blocks(n, 0);
    std::vector<double> sums(n, 0.0);

    p.start([&](std::int64_t t) {
        alignas(64) T state[kChains * Ops::lanes];
        for (std::int64_t k = 0; k < kChains * Ops::lanes; ++k) {
            state[k] = T(1) + T(k + t) * T(1e-3);
        }
        ready.fetch_add(1, std::memory_order_release);
        while (phase.load(std::memory_order_acquire) == idle) {
            _mm_pause();
        }
        std::int64_t counted = 0;
        for (int ph = 0; (ph = phase.load(std::memory_order_acquire)) != stop;) {
            fma_block<Ops>(state, T(0.999), T(0.001), kBlockIters);
            if (ph == measure) {
                ++counted;
            }
        }
        blocks[t] = counted;
        double sum = 0.0;
        for (const T x : state) {
            sum += x;
        }
        sums[t] = sum;
    });

    // Barrier: every worker is pinned and spinning before anyone starts.
    while (ready.load(std::memory_order_acquire) < n) {
        _mm_pause();
    }
    phase.store(warmup, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::duration<double>(o.warmup));
    pdh.sample();
    phase.store(measure, std::memory_order_release);
    const auto t0 = clock_type::now();
    // The coordinator sleeps through the window so it never competes for an
    // FMA port on a worker's hyperthread sibling.
    std::this_thread::sleep_for(std::chrono::duration<double>(o.seconds));
    pdh.sample();
    const auto t1 = clock_type::now();
    phase.store(stop, std::memory_order_release);
    p.wait();

    const double window = secs(t1 - t0);
    const double flops_per_block = static_cast<double>(kChains * Ops::lanes * 2 * kBlockIters);
    const std::int64_t flops_per_cycle = kFmaPorts * Ops::lanes * 2;
    double total = 0.0;
    std::vector<std::string> per;
    for (std::int64_t t = 0; t < n; ++t) {
        const double gf = static_cast<double>(blocks[t]) * flops_per_block / window / 1e9;
        total += gf;
        g_sink = g_sink + sums[t];
        per.push_back(obj({{"worker", std::to_string(t)},
                           {"logical_cpu", std::to_string(p.core(t).logical)},
                           {"gflops", num(gf)},
                           {"effective_ghz", num(gf / static_cast<double>(flops_per_cycle))}}));
    }
    return obj({{"dtype", json_str(dtype)},
                {"threads", std::to_string(n)},
                {"chains", std::to_string(kChains)},
                {"lanes", std::to_string(Ops::lanes)},
                {"iterations_per_block", std::to_string(kBlockIters)},
                {"flops_per_cycle_per_core", std::to_string(flops_per_cycle)},
                {"warmup_seconds", num(o.warmup)},
                {"window_seconds", num(window)},
                {"total_gflops", num(total)},
                {"per_thread", join(per)},
                {"pdh", pdh.report()}});
}

// ---------------------------------------------------------------- bandwidth

template <bool NT>
inline void store_pd(double* p, __m256d v) {
    if constexpr (NT) {
        _mm256_stream_pd(p, v);
    } else {
        _mm256_store_pd(p, v);
    }
}

// Every kernel takes a 32-byte-aligned span whose length is a multiple of 16
// doubles, so the loops need no remainder handling.

PG_NOINLINE double k_read(const double* a, std::int64_t n) {
    // Four accumulators keep add latency off the critical path; loads bound it.
    __m256d s0 = _mm256_setzero_pd(), s1 = s0, s2 = s0, s3 = s0;
    for (std::int64_t i = 0; i < n; i += 16) {
        s0 = _mm256_add_pd(s0, _mm256_load_pd(a + i));
        s1 = _mm256_add_pd(s1, _mm256_load_pd(a + i + 4));
        s2 = _mm256_add_pd(s2, _mm256_load_pd(a + i + 8));
        s3 = _mm256_add_pd(s3, _mm256_load_pd(a + i + 12));
    }
    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, _mm256_add_pd(_mm256_add_pd(s0, s1), _mm256_add_pd(s2, s3)));
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}

// Streaming stores are weakly ordered; the sfence makes them globally visible
// before the worker reports done, so the trial time covers them.
template <bool NT>
PG_NOINLINE void k_write(double* a, double s, std::int64_t n) {
    const __m256d v = _mm256_set1_pd(s);
    for (std::int64_t i = 0; i < n; i += 16) {
        store_pd<NT>(a + i, v);
        store_pd<NT>(a + i + 4, v);
        store_pd<NT>(a + i + 8, v);
        store_pd<NT>(a + i + 12, v);
    }
    if constexpr (NT) {
        _mm_sfence();
    }
}

template <bool NT>
PG_NOINLINE void k_copy(double* a, const double* b, std::int64_t n) {
    for (std::int64_t i = 0; i < n; i += 16) {
        store_pd<NT>(a + i, _mm256_load_pd(b + i));
        store_pd<NT>(a + i + 4, _mm256_load_pd(b + i + 4));
        store_pd<NT>(a + i + 8, _mm256_load_pd(b + i + 8));
        store_pd<NT>(a + i + 12, _mm256_load_pd(b + i + 12));
    }
    if constexpr (NT) {
        _mm_sfence();
    }
}

template <bool NT>
PG_NOINLINE void k_triad(double* a, const double* b, const double* c, double s, std::int64_t n) {
    const __m256d v = _mm256_set1_pd(s);
    for (std::int64_t i = 0; i < n; i += 16) {
        store_pd<NT>(a + i, _mm256_fmadd_pd(v, _mm256_load_pd(c + i), _mm256_load_pd(b + i)));
        store_pd<NT>(a + i + 4,
                     _mm256_fmadd_pd(v, _mm256_load_pd(c + i + 4), _mm256_load_pd(b + i + 4)));
        store_pd<NT>(a + i + 8,
                     _mm256_fmadd_pd(v, _mm256_load_pd(c + i + 8), _mm256_load_pd(b + i + 8)));
        store_pd<NT>(a + i + 12,
                     _mm256_fmadd_pd(v, _mm256_load_pd(c + i + 12), _mm256_load_pd(b + i + 12)));
    }
    if constexpr (NT) {
        _mm_sfence();
    }
}

struct aligned_free {
    void operator()(double* p) const { ::operator delete(p, std::align_val_t{64}); }
};
using dbuf = std::unique_ptr<double[], aligned_free>;

dbuf make_buf(std::int64_t n) {
    return dbuf(static_cast<double*>(
        ::operator new(static_cast<std::size_t>(n) * sizeof(double), std::align_val_t{64})));
}

std::string run_mem(pool& p, const options& o, pdh_window& pdh) {
    const std::int64_t n = o.mem_mb * (std::int64_t{1} << 20) / static_cast<std::int64_t>(sizeof(double));
    const std::int64_t workers = p.size();
    // Whole multiples of 16 doubles per worker; the last worker takes the rest,
    // which is also a multiple of 16 because n is.
    const std::int64_t chunk = n / workers / 16 * 16;
    const auto lo = [&](std::int64_t t) { return t * chunk; };
    const auto len = [&](std::int64_t t) { return (t == workers - 1 ? n : (t + 1) * chunk) - lo(t); };

    const dbuf a = make_buf(n), b = make_buf(n), c = make_buf(n);
    double* const pa = a.get();
    double* const pb = b.get();
    double* const pc = c.get();
    // First touch by the same pinned workers that time the kernels, so every
    // page is committed before timing and placed by the thread that uses it.
    p.run([&](std::int64_t t) {
        for (std::int64_t i = lo(t); i < lo(t) + len(t); ++i) {
            pa[i] = 1.0;
            pb[i] = 2.0;
            pc[i] = 0.5;
        }
    });

    constexpr double s = 3.0;
    std::vector<double> part(workers, 0.0);
    struct kernel {
        const char* name;
        const char* stores;
        std::int64_t bytes_per_elem; // explicit reads plus writes
        std::function<void(std::int64_t)> body;
    };
    // read runs first, while a still holds its first-touch 1.0, so its sum
    // must equal n exactly; triad_nt runs last and leaves a == 2 + 3 * 0.5.
    const std::vector<kernel> kernels = {
        {"read", "none", 8, [&](std::int64_t t) { part[t] = k_read(pa + lo(t), len(t)); }},
        {"write", "regular", 8, [&](std::int64_t t) { k_write<false>(pa + lo(t), s, len(t)); }},
        {"write", "nontemporal", 8, [&](std::int64_t t) { k_write<true>(pa + lo(t), s, len(t)); }},
        {"copy", "regular", 16,
         [&](std::int64_t t) { k_copy<false>(pa + lo(t), pb + lo(t), len(t)); }},
        {"copy", "nontemporal", 16,
         [&](std::int64_t t) { k_copy<true>(pa + lo(t), pb + lo(t), len(t)); }},
        {"triad", "regular", 24,
         [&](std::int64_t t) { k_triad<false>(pa + lo(t), pb + lo(t), pc + lo(t), s, len(t)); }},
        {"triad", "nontemporal", 24,
         [&](std::int64_t t) { k_triad<true>(pa + lo(t), pb + lo(t), pc + lo(t), s, len(t)); }},
    };

    std::vector<std::string> rows;
    for (const kernel& k : kernels) {
        const std::int64_t bytes = k.bytes_per_elem * n;
        std::vector<double> gbs;
        pdh.sample();
        for (std::int64_t trial = 0; trial < o.trials; ++trial) {
            gbs.push_back(static_cast<double>(bytes) / p.run(k.body) / 1e9);
            if (std::string_view(k.name) == "read") {
                double total = 0.0;
                for (const double x : part) {
                    total += x;
                }
                if (total != static_cast<double>(n)) {
                    throw std::runtime_error("read kernel did not cover every element once");
                }
                g_sink = g_sink + total;
            }
        }
        pdh.sample();
        std::vector<std::string> trials;
        for (const double g : gbs) {
            trials.push_back(num(g));
        }
        rows.push_back(obj({{"kernel", json_str(k.name)},
                            {"stores", json_str(k.stores)},
                            {"bytes_per_trial", std::to_string(bytes)},
                            {"best_gbs", num(*std::max_element(gbs.begin(), gbs.end()))},
                            {"median_gbs", num(median(gbs))},
                            {"trial_gbs", join(trials)},
                            {"pdh", pdh.report()}}));
    }
    if (!std::all_of(pa, pa + n, [](double x) { return x == 3.5; })) {
        throw std::runtime_error("triad kernel left a wrong value in a");
    }
    return obj({{"threads", std::to_string(workers)}, {"kernels", join(rows)}});
}

std::string timestamp_utc() {
    return std::format("{:%Y-%m-%dT%H:%M:%SZ}",
                       std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));
}

} // namespace

int main(int argc, char** argv) {
    try {
        const topology topo = probe_topology();
        const std::int64_t cores = static_cast<std::int64_t>(topo.cores.size());
        const options o = parse(argc, argv, cores);
        if (o.help) {
            std::fputs(kUsage, stdout);
            return 0;
        }

        pdh_window pdh;
        std::vector<std::string> placement, fma, mem;
        for (const std::int64_t n : o.threads) {
            pool p(std::vector<core_slot>(topo.cores.begin(), topo.cores.begin() + n));
            std::vector<std::string> workers;
            for (std::int64_t t = 0; t < n; ++t) {
                if (p.observed(t) < 0) {
                    throw std::runtime_error(std::format("could not pin worker {} to logical cpu {}",
                                                         t, p.core(t).logical));
                }
                workers.push_back(obj({{"worker", std::to_string(t)},
                                       {"group", std::to_string(p.core(t).group)},
                                       {"logical_cpu", std::to_string(p.core(t).logical)},
                                       {"observed_cpu", std::to_string(p.observed(t))}}));
            }
            placement.push_back(obj({{"threads", std::to_string(n)}, {"workers", join(workers)}}));
            if (!o.skip_fma) {
                fma.push_back(run_fma<f32_ops>(p, o, pdh, "float32"));
                fma.push_back(run_fma<f64_ops>(p, o, pdh, "float64"));
            }
            if (!o.skip_mem) {
                mem.push_back(run_mem(p, o, pdh));
            }
        }

        std::string pcie = json_str("not built");
#if defined(PG_CEILINGS_HAS_CUDA)
        if (o.skip_pcie) {
            pcie = json_str("skipped");
        } else {
            try {
                pcie = pcie_report();
            } catch (const std::exception& e) {
                pcie = obj({{"error", json_str(e.what())}});
            }
        }
        const std::string nvcc = json_str(PG_CEILINGS_NVCC_VERSION);
#else
        const std::string nvcc = "null";
#endif

        std::vector<std::string> core_rows, thread_list;
        for (std::int64_t i = 0; i < cores; ++i) {
            core_rows.push_back(obj({{"core", std::to_string(i)},
                                     {"group", std::to_string(topo.cores[i].group)},
                                     {"first_logical", std::to_string(topo.cores[i].logical)},
                                     {"logical_count", std::to_string(topo.cores[i].siblings)}}));
        }
        for (const std::int64_t n : o.threads) {
            thread_list.push_back(std::to_string(n));
        }

        const std::string report = obj({
            {"tool", json_str("pg_ceilings")},
            {"version", json_str(kToolVersion)},
            {"timestamp_utc", json_str(timestamp_utc())},
            {"cpu", obj({{"brand", json_str(cpu_brand())},
                         {"physical_cores", std::to_string(cores)},
                         {"logical_processors", std::to_string(topo.logical_total)},
                         {"cores", join(core_rows)}})},
            {"build", obj({{"compiler_id", json_str(PG_CEILINGS_COMPILER_ID)},
                           {"compiler_version", json_str(PG_CEILINGS_COMPILER_VERSION)},
                           {"config", json_str(PG_CEILINGS_CONFIG)},
                           {"cxx_flags", json_str(PG_CEILINGS_FLAGS)},
                           {"nvcc_version", nvcc}})},
            {"params", obj({{"threads", join(thread_list)},
                            {"seconds", num(o.seconds)},
                            {"warmup_seconds", num(o.warmup)},
                            {"mem_mb", std::to_string(o.mem_mb)},
                            {"trials", std::to_string(o.trials)},
                            {"skip_fma", boolean(o.skip_fma)},
                            {"skip_mem", boolean(o.skip_mem)},
                            {"skip_pcie", boolean(o.skip_pcie)}})},
            {"placement", join(placement)},
            {"fma", o.skip_fma ? json_str("skipped") : join(fma)},
            {"memory",
             o.skip_mem
                 ? json_str("skipped")
                 : obj({{"byte_counting",
                         json_str("STREAM convention: bytes each kernel explicitly reads plus "
                                  "writes; write-allocate traffic of regular stores is not counted")},
                        {"element", json_str("float64")},
                        {"gb", json_str("1e9 bytes")},
                        {"array_bytes", std::to_string(o.mem_mb << 20)},
                        {"trials", std::to_string(o.trials)},
                        {"runs", join(mem)}})},
            {"pcie", pcie},
        });

        std::fputs(report.c_str(), stdout);
        std::fputc('\n', stdout);
        if (!o.out.empty()) {
            std::ofstream f(o.out, std::ios::binary);
            f << report << '\n';
            if (!f) {
                throw std::runtime_error(std::format("could not write {}", o.out));
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "pg_ceilings: %s\n", e.what());
        return 1;
    }
}
