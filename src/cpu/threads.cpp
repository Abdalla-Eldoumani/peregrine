// std::getenv is the standard portable read and the value is only parsed as a
// number, so MSVC's /W4 C4996 deprecation (which steers toward getenv_s) is
// noise here; silence it TU-locally.
#define _CRT_SECURE_NO_WARNINGS

#include "cpu/threads.hpp"

#include "cpu/topology.hpp"

#include <atomic>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <xmmintrin.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace pg::cpu {
namespace {

// 0 means not set: get_num_threads falls back to the resolved value.
std::atomic<int> g_setting{0};

// A positive decimal int with optional surrounding whitespace, else 0. With
// first_of_list the value may be followed by a comma and more entries, which
// is how OpenMP reads OMP_NUM_THREADS (one count per nesting level).
int env_threads(const char* name, bool first_of_list) noexcept {
    const char* s = std::getenv(name);
    if (s == nullptr) {
        return 0;
    }
    while (std::isspace(static_cast<unsigned char>(*s))) {
        ++s;
    }
    if (*s < '0' || *s > '9') {
        return 0;
    }
    long long v = 0;
    for (; *s >= '0' && *s <= '9'; ++s) {
        v = v * 10 + (*s - '0');
        if (v > INT_MAX) {
            v = INT_MAX;
        }
    }
    while (std::isspace(static_cast<unsigned char>(*s))) {
        ++s;
    }
    if (*s != '\0' && !(first_of_list && *s == ',')) {
        return 0;
    }
    return static_cast<int>(v);
}

int resolve_setting() noexcept {
    if (const int n = env_threads("PEREGRINE_NUM_THREADS", false); n > 0) {
        return n;
    }
    if (const int n = env_threads("OMP_NUM_THREADS", true); n > 0) {
        return n;
    }
    return physical_cores();
}

} // namespace

void set_num_threads(int n) noexcept {
    g_setting.store(n > 0 ? n : 0, std::memory_order_relaxed);
}

int get_num_threads() noexcept {
    const int n = g_setting.load(std::memory_order_relaxed);
    if (n > 0) {
        return n;
    }
    static const int resolved = resolve_setting();
    return resolved;
}

int call_threads() noexcept {
#if defined(_OPENMP)
    const int setting = get_num_threads();
    const int limit = omp_get_max_threads();
    const int n = setting < limit ? setting : limit;
    return n > 0 ? n : 1;
#else
    return 1;
#endif
}

int team_size(double work, double serial_below, double per_thread, int64_t max_units) noexcept {
    if (work < serial_below) {
        return 1;
    }
    // In double so a huge work / per_thread never overflows an int; a NaN
    // quotient (0 / 0) fails the comparison and leaves the cap alone.
    double nt = call_threads();
    const double by_work = std::ceil(work / per_thread);
    if (by_work < nt) {
        nt = by_work;
    }
    if (static_cast<double>(max_units) < nt) {
        nt = static_cast<double>(max_units);
    }
    return nt >= 1 ? static_cast<int>(nt) : 1;
}

unsigned mxcsr() noexcept {
    return _mm_getcsr();
}

void set_mxcsr(unsigned csr) noexcept {
    _mm_setcsr(csr);
}

// The caller's own thread (team member 0, or the whole serial region) already
// holds the value, so it writes nothing and keeps any exception flags it raised.
mxcsr_scope::mxcsr_scope(unsigned caller) noexcept : own_(_mm_getcsr()), caller_(caller) {
    if (own_ != caller_) {
        _mm_setcsr(caller_);
    }
}

mxcsr_scope::~mxcsr_scope() {
    if (own_ != caller_) {
        _mm_setcsr(own_);
    }
}

} // namespace pg::cpu
