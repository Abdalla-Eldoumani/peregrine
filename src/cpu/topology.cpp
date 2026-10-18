#include "cpu/topology.hpp"

#include <cstdlib>
#include <thread>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sched.h>

#include <cstdio>
#endif

namespace pg::cpu {
namespace {

// hardware_concurrency ignores the affinity mask (it reads 12 under a 6-CPU
// mask on the reference machine), so it is only the answer when the topology
// cannot be read.
int logical_fallback() noexcept {
    const unsigned n = std::thread::hardware_concurrency();
    return n > 0 ? static_cast<int>(n) : 1;
}

#if defined(_WIN32)
int count_physical_cores() noexcept {
    DWORD_PTR proc = 0;
    DWORD_PTR sys = 0;
    GROUP_AFFINITY group{};
    if (!GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys) || proc == 0 ||
        !GetThreadGroupAffinity(GetCurrentThread(), &group)) {
        return logical_fallback();
    }
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0) {
        return logical_fallback();
    }
    auto* buf = static_cast<unsigned char*>(std::malloc(len));
    if (buf == nullptr ||
        !GetLogicalProcessorInformationEx(
            RelationProcessorCore, reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf), &len)) {
        std::free(buf);
        return logical_fallback();
    }
    // shortcut: counts cores in the calling thread's processor group only; upgrade past 64 logical CPUs
    int cores = 0;
    for (DWORD off = 0; off < len;) {
        const auto* e = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf + off);
        if (e->Relationship == RelationProcessorCore) {
            for (WORD g = 0; g < e->Processor.GroupCount; ++g) {
                const GROUP_AFFINITY& ga = e->Processor.GroupMask[g];
                if (ga.Group == group.Group && (ga.Mask & proc) != 0) {
                    ++cores;
                    break;
                }
            }
        }
        off += e->Size;
    }
    std::free(buf);
    return cores > 0 ? cores : logical_fallback();
}
#else
// The lowest CPU in a core's thread_siblings_list names that core, and CPU
// numbers are global, so it is unique across packages and dies.
int first_cpu_in_list(const char* path) noexcept {
    std::FILE* f = std::fopen(path, "r");
    if (f == nullptr) {
        return -1;
    }
    int first = -1;
    if (std::fscanf(f, "%d", &first) != 1) {
        first = -1;
    }
    std::fclose(f);
    return first;
}

int count_physical_cores() noexcept {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        return logical_fallback();
    }
    bool seen[CPU_SETSIZE] = {};
    int cores = 0;
    char path[96];
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &set)) {
            continue;
        }
        std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
        const int first = first_cpu_in_list(path);
        if (first < 0 || first >= CPU_SETSIZE) {
            return CPU_COUNT(&set);
        }
        if (!seen[first]) {
            seen[first] = true;
            ++cores;
        }
    }
    // shortcut: ignores cgroup cpu.max quotas; upgrade when quota-limited containers matter
    return cores > 0 ? cores : logical_fallback();
}
#endif

} // namespace

int physical_cores() noexcept {
    static const int cores = count_physical_cores();
    return cores;
}

} // namespace pg::cpu
