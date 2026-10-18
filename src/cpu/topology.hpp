#pragma once

namespace pg::cpu {

// Physical cores the process may run on: cores with at least one logical CPU in
// the affinity mask, at least 1. SMT siblings share one core's FMA ports, so
// FMA-bound kernels measured slower at 12 threads than at 6 on the 6-core
// reference machine; this is the default team size. Computed on the first call
// and reused, so a later affinity change does not move it.
int physical_cores() noexcept;

} // namespace pg::cpu
