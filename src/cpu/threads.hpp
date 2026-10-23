#pragma once
#include <cstdint>

namespace pg::cpu {

// The process-wide CPU thread setting every parallel region honors, shared by
// all calling threads. Until set_num_threads is called it is the value resolved
// on the first read (module import does that read): PEREGRINE_NUM_THREADS when
// it holds a positive int, else the first entry of OMP_NUM_THREADS when that is
// a positive int, else physical_cores(). Any other value is ignored, never an
// error. Values above INT_MAX clamp to it. Peregrine never calls
// omp_set_num_threads: under MSVC's vcomp it is process-wide, so it would
// change the team of every OpenMP user in the process, not just ours. The
// caller validates n; n < 1 returns to the resolved value.
void set_num_threads(int n) noexcept;
int get_num_threads() noexcept;

// Threads one call may use: the setting, capped by omp_get_max_threads() on the
// calling thread (OpenMP's own limit, which OMP_NUM_THREADS or threadpoolctl
// may have lowered), at least 1. Regions pass it as num_threads explicitly
// because vcomp's default team ignores the affinity mask.
int call_threads() noexcept;

// Team size for a region doing `work` units: 1 when work < serial_below (no
// OpenMP call is made then), else min(call_threads(), ceil(work / per_thread),
// max_units), at least 1. max_units is the number of independent pieces the
// region can split into, so no thread is left without one. Pure apart from
// call_threads().
int team_size(double work, double serial_below, double per_thread, int64_t max_units) noexcept;

// MXCSR holds FTZ/DAZ and the rounding mode per thread, and OpenMP workers do
// not inherit the caller's, so a caller running with flush-to-zero would get
// different subnormal bits from the rows a worker computed. Read the caller's
// value before a region; at the top of the region each thread holds an
// mxcsr_scope, which adopts it and restores the thread's own on exit.
unsigned mxcsr() noexcept;
void set_mxcsr(unsigned csr) noexcept;

class mxcsr_scope {
public:
    explicit mxcsr_scope(unsigned caller) noexcept;
    ~mxcsr_scope();
    mxcsr_scope(const mxcsr_scope&) = delete;
    mxcsr_scope& operator=(const mxcsr_scope&) = delete;

private:
    unsigned own_;
    unsigned caller_;
};

} // namespace pg::cpu
