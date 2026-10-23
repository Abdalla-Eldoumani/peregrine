# Dev-only hardware ceiling probe. There is deliberately no install() rule, so
# the executable can never be packaged into the wheel.

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # Standalone configure: measuring the machine needs no Python or nanobind.
  cmake_minimum_required(VERSION 3.26)
  project(pg_ceilings LANGUAGES CXX)
  option(PG_CEILINGS_CUDA "Build the PCIe transfer probe (requires the CUDA Toolkit)" OFF)
  set(pg_ceilings_cuda ${PG_CEILINGS_CUDA})
  if (pg_ceilings_cuda)
    enable_language(CUDA)
  endif()
else()
  # Under the top-level project the CUDA language is already enabled by
  # PG_ENABLE_CUDA, so the probe simply follows it.
  set(pg_ceilings_cuda ${PG_ENABLE_CUDA})
endif()

add_executable(pg_ceilings ceilings.cpp)
target_compile_features(pg_ceilings PRIVATE cxx_std_20)
find_package(Threads REQUIRED)
target_link_libraries(pg_ceilings PRIVATE Threads::Threads)
if (WIN32)
  target_link_libraries(pg_ceilings PRIVATE pdh)
endif()

# AVX2 at target level is safe here, unlike in the Python module: this binary
# is only ever run on the machine being measured. No fast-math flag, so FMA and
# the bandwidth kernels keep strict IEEE semantics; no OpenMP, so the threads
# are plain pinned std::threads.
if (MSVC)
  set(pg_opts /O2 /arch:AVX2 /W4 /permissive-)
else()
  set(pg_opts -O3 -mavx2 -mfma -Wall -Wextra)
endif()
# CXX-fenced so nvcc never sees raw MSVC host flags on its own command line.
target_compile_options(pg_ceilings PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${pg_opts}>")

# The report records exactly how it was built. Per-config flags come through
# generator expressions because multi-config generators only know the config
# at build time. These are set on ceilings.cpp alone so the quoted, spaced
# strings never reach the nvcc command line.
set(pg_cfg_flags "")
foreach (cfg Debug Release RelWithDebInfo MinSizeRel)
  string(TOUPPER ${cfg} cfg_upper)
  string(APPEND pg_cfg_flags "$<$<CONFIG:${cfg}>:${CMAKE_CXX_FLAGS_${cfg_upper}}>")
endforeach()
list(JOIN pg_opts " " pg_opts_str)
set_property(SOURCE ceilings.cpp APPEND PROPERTY COMPILE_DEFINITIONS
  "PG_CEILINGS_COMPILER_ID=\"${CMAKE_CXX_COMPILER_ID}\""
  "PG_CEILINGS_COMPILER_VERSION=\"${CMAKE_CXX_COMPILER_VERSION}\""
  "PG_CEILINGS_CONFIG=\"$<CONFIG>\""
  "PG_CEILINGS_FLAGS=\"${CMAKE_CXX_FLAGS} ${pg_cfg_flags} ${pg_opts_str}\"")

if (pg_ceilings_cuda)
  target_sources(pg_ceilings PRIVATE pcie.cu)
  # sm_86 is the GA106 in the primary dev machine; the probe launches no
  # kernels, so the arch only has to be one the toolkit accepts. Shared cudart
  # because cudart_static.lib is built against the static CRT and collides
  # with the /MD objects (LNK4098); the Python module links shared cudart too.
  set_target_properties(pg_ceilings PROPERTIES
    CUDA_ARCHITECTURES 86
    CUDA_RUNTIME_LIBRARY Shared)
  set_property(SOURCE ceilings.cpp APPEND PROPERTY COMPILE_DEFINITIONS
    PG_CEILINGS_HAS_CUDA=1
    "PG_CEILINGS_NVCC_VERSION=\"${CMAKE_CUDA_COMPILER_VERSION}\"")
endif()
