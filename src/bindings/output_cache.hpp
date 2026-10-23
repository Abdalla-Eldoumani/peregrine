#pragma once
#include <cstddef>

// Reuse of large host output buffers. The Windows heap hands every request from
// about 1016 KiB up to VirtualAlloc and returns it to the system on free, so
// each fresh large output faults in zeroed pages on first touch (about 0.4-0.5
// us per 4 KiB page) and pays again to unmap them on free. Blocks freed here are
// kept, keyed by their exact size, and handed back to the next request of that
// size. Every block, cached or not, comes from std::malloc, so std::realloc and
// std::free are correct for any pointer this file ever returns. Thread-safe;
// callers need not hold the GIL.
namespace pg::bindings::output_cache {

// Requests below this go straight to std::malloc and std::free. 512 KiB is the
// smallest size the heap was seen to hand out as fresh pages: whether a request
// from 512 to 1016 KiB faults depends on the heap's history (0.9 us a page at
// 512 and 768 KiB in one state, none in another), and none faulted at 256 KiB.
// At 512 KiB an allocation and free through the cache cost 470 ns against 410
// for the heap's free list, beside at least about 10 us of work for an output
// that size.
inline constexpr std::size_t kThreshold = std::size_t{512} << 10;

// Reads PEREGRINE_OUTPUT_CACHE_MB once (cached bytes cap in MiB, default 256;
// 0 turns the cache off). False when the cache is off. 256 MiB holds a 4096^2
// float64 result twice over (its transpose ran 1.3x faster cached). The one
// larger output measured, a 5793^2 float64 fused result, ran within noise of
// uncached with a 512 MiB cap, so a larger default would hold more memory for
// no measured return.
bool init() noexcept;

void* allocate(std::size_t size) noexcept;
void* allocate_zeroed(std::size_t nelem, std::size_t elsize) noexcept;
void* reallocate(void* ptr, std::size_t new_size) noexcept;
void release(void* ptr, std::size_t size) noexcept;

struct stats {
    std::size_t cap;    // bytes; 0 when the cache is off
    std::size_t bytes;  // held in idle cached blocks
    std::size_t blocks;
    std::size_t hits;   // requests served from the cache since init
};
stats read() noexcept;

} // namespace pg::bindings::output_cache
