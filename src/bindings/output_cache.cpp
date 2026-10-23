// std::getenv is the standard portable read and the value is only parsed as a
// number, so MSVC's /W4 C4996 deprecation (which steers toward getenv_s) is
// noise here; silence it TU-locally.
#define _CRT_SECURE_NO_WARNINGS

#include "bindings/output_cache.hpp"

#include <cstdint>
#include <cstdlib>
#include <malloc.h>
#include <mutex>
#include <new>
#include <vector>

namespace pg::bindings::output_cache {
namespace {

struct block {
    void* ptr;
    std::size_t size;
};

struct cache {
    std::mutex lock;
    std::vector<block> blocks; // oldest first
    std::size_t bytes = 0;
    std::size_t hits = 0;
};

constexpr std::size_t kDefaultCapMb = 256;

std::size_t g_cap = 0;
// Never destroyed: an array holding a block may be freed during interpreter
// finalization, and a static destructor could already have run by then.
cache* g_cache = nullptr;

// A whole decimal number, else the default. Silently, like the thread
// variables: a bad value must not make the package fail to import.
std::size_t env_cap_mb() noexcept {
    const char* s = std::getenv("PEREGRINE_OUTPUT_CACHE_MB");
    if (s == nullptr || *s < '0' || *s > '9') {
        return kDefaultCapMb;
    }
    std::size_t mb = 0;
    for (; *s >= '0' && *s <= '9'; ++s) {
        mb = mb > (SIZE_MAX >> 20) / 10 ? SIZE_MAX >> 20 : mb * 10 + static_cast<std::size_t>(*s - '0');
    }
    return *s == '\0' ? mb : kDefaultCapMb;
}

// The allocator's own record of the block. A free size above it would mean the
// caller misreported the size, and caching the block under that size would hand
// a short buffer to the next request.
std::size_t usable_size(void* ptr) noexcept {
#if defined(_WIN32)
    return _msize(ptr);
#else
    return malloc_usable_size(ptr);
#endif
}

} // namespace

bool init() noexcept {
    const std::size_t mb = env_cap_mb();
    if (mb == 0) {
        return false;
    }
    g_cache = new (std::nothrow) cache;
    if (g_cache == nullptr) {
        return false;
    }
    g_cap = mb > (SIZE_MAX >> 20) ? SIZE_MAX : mb << 20;
    return true;
}

void* allocate(std::size_t size) noexcept {
    if (size >= kThreshold && size <= g_cap) {
        std::lock_guard<std::mutex> hold(g_cache->lock);
        std::vector<block>& v = g_cache->blocks;
        // Newest first: the most recently freed block is the likeliest to be
        // in cache still.
        for (std::size_t i = v.size(); i-- > 0;) {
            if (v[i].size == size) {
                void* ptr = v[i].ptr;
                v.erase(v.begin() + static_cast<std::ptrdiff_t>(i));
                g_cache->bytes -= size;
                ++g_cache->hits;
                return ptr;
            }
        }
    }
    return std::malloc(size);
}

// Fresh pages from the system already read as zero, while a cached block would
// need a full memset, so zeroed requests never use the cache.
void* allocate_zeroed(std::size_t nelem, std::size_t elsize) noexcept {
    return std::calloc(nelem, elsize);
}

void* reallocate(void* ptr, std::size_t new_size) noexcept {
    return std::realloc(ptr, new_size);
}

void release(void* ptr, std::size_t size) noexcept {
    if (ptr == nullptr || size < kThreshold || size > g_cap || usable_size(ptr) < size) {
        std::free(ptr);
        return;
    }
    // Evict oldest first until the block fits under the cap. The system free of
    // a victim runs outside the lock: unmapping a large block costs about 0.1 us
    // a page, too long to stall other threads' allocations.
    for (;;) {
        void* victim = nullptr;
        {
            std::lock_guard<std::mutex> hold(g_cache->lock);
            std::vector<block>& v = g_cache->blocks;
            if (g_cache->bytes + size <= g_cap) {
                try {
                    v.push_back({ptr, size});
                } catch (...) {
                    break;
                }
                g_cache->bytes += size;
                return;
            }
            victim = v.front().ptr;
            g_cache->bytes -= v.front().size;
            v.erase(v.begin());
        }
        std::free(victim);
    }
    std::free(ptr);
}

stats read() noexcept {
    if (g_cache == nullptr) {
        return {0, 0, 0, 0};
    }
    std::lock_guard<std::mutex> hold(g_cache->lock);
    return {g_cap, g_cache->bytes, g_cache->blocks.size(), g_cache->hits};
}

} // namespace pg::bindings::output_cache
