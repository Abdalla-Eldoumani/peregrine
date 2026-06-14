cmake_minimum_required(VERSION 3.26)
project(peregrine LANGUAGES CXX)

option(PG_ENABLE_CUDA "Build the CUDA backend (requires CUDA Toolkit 12.8)" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)

# nanobind is resolved from the build venv that scikit-build-core creates
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

nanobind_add_module(_core STABLE_ABI NB_STATIC
  src/bindings/module.cpp
  src/cpu/feature_detect.cpp
  src/cpu/gemm_naive.cpp
  src/cpu/pack.cpp
  src/cpu/microkernel_avx2_f64.cpp
  src/cpu/microkernel_avx2_f32.cpp
  src/cpu/gemm_blis.cpp
  src/cpu/fused_avx2.cpp
  src/cpu/fused_naive.cpp
  src/cpu/transpose.cpp
  src/cpu/reduce.cpp
  src/dispatch/dispatch.cpp
)
target_include_directories(_core PRIVATE src)
target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)

# AVX2 codegen is confined to exactly these kernel translation units. The arch
# flag is deliberately absent at target level: AVX2 instructions in a TU that the
# no-AVX2 fallback path links produce a module that executes an illegal
# instruction at import on a CPU without AVX2 (the legacy import crash). The
# feature_detect, gemm_naive, fused_naive, dispatch, and module TUs must stay
# free of AVX2 codegen so the fallback path runs everywhere; fused_naive.cpp is
# the fused fallback and is kept OUT of this list on purpose.
set(PG_AVX2_SOURCES
  src/cpu/pack.cpp
  src/cpu/microkernel_avx2_f64.cpp
  src/cpu/microkernel_avx2_f32.cpp
  src/cpu/gemm_blis.cpp
  src/cpu/fused_avx2.cpp
)

if (MSVC)
  # /openmp:llvm because the legacy /openmp runtime is OpenMP 2.0 and rejects
  # modern pragmas. No /fp:fast: it changes NaN and Inf semantics and the
  # library promises NumPy-matching results.
  #
  # Fenced behind COMPILE_LANGUAGE:CXX so these MSVC host flags reach only the
  # .cpp TUs. nvcc rejects a raw /permissive- on its own command line ("a single
  # input file is required for a non-link phase"); host flags for .cu files must
  # go through -Xcompiler, which nvcc adds itself. On a CXX-only (CUDA OFF)
  # target the generator expression is true for every TU, so the OFF build sees
  # byte-identical flags.
  target_compile_options(_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/O2;/openmp:llvm;/permissive-;/W4>)
  set_source_files_properties(${PG_AVX2_SOURCES} PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
  # No -ffast-math for the same reason. CXX-fenced for the same nvcc reason as
  # the MSVC branch (and a no-op difference on the CUDA-OFF Linux/CI build).
  target_compile_options(_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>)
  set_source_files_properties(${PG_AVX2_SOURCES} PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")

  # libatomic on GCC/Clang. The MC/KC/NC sweep hook publishes its triple through
  # a std::atomic<blocking> (gemm_blis.cpp); blocking is three int64 (24 bytes),
  # which is NOT lock-free on x86. GCC lowers a non-lock-free std::atomic to
  # __atomic_load/__atomic_store libcalls that live in libatomic, but does so
  # NON-deterministically: depending on inlining it sometimes emits an inline
  # cmpxchg16b instead and needs no library. When it emits the libcalls and
  # libatomic is not linked, _core.so has undefined __atomic_store/__atomic_load
  # and fails to import (caught only by a from-clean GCC build -- a prior build
  # happened to inline and imported fine, which is exactly the flakiness). Link
  # libatomic when present so the libcall path always resolves; it is a no-op when
  # GCC inlines. MSVC handles the 16-byte atomic intrinsically, so this is the
  # GNU/Clang branch only. find_library keeps it portable (no hard -latomic on a
  # toolchain that lacks a separate libatomic).
  find_library(PG_LIBATOMIC NAMES atomic libatomic.so.1)
  if (PG_LIBATOMIC)
    target_link_libraries(_core PRIVATE ${PG_LIBATOMIC})
  endif()
endif()

if (PG_ENABLE_CUDA)
  enable_language(CUDA)
  # sm_86 is the GA106 in the primary dev machine. PTX for 8.6 keeps newer
  # consumer parts working through JIT.
  set(CMAKE_CUDA_ARCHITECTURES 86)
  # Keep the C++ standard identical across the .cu and .cpp halves: nvcc passes
  # host code to MSVC, and a standard mismatch between the two compilers is a
  # subtle ODR and ABI hazard.
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # No version argument. The installed toolkit is 12.8 (nvcc 12.8.61); a bare
  # find locates it and exposes CUDA::cudart/cublas/cublasLt. A "13" here
  # returns FALSE on 12.8 and FATALs the configure (verified on-machine).
  find_package(CUDAToolkit REQUIRED)
  # context.cu is the process-lifetime singleton (streams, mempool,
  # cublas/cublasLt handles, atexit teardown); gemm_cublas.cu is the f32/f64
  # cuBLAS GEMM; transfer.cu is the async H2D/D2H + device allocator with the OOM
  # pre-flight; fused.cu is the hand-written grid-stride/float4 fused elementwise
  # kernels (axpby/fma3/scaled_relu) + the time_fused_chain timer, device-resident
  # only (the one place the project ships custom CUDA kernels; GEMM is
  # cuBLAS-only). enable_language(CUDA) plus any one .cu here keeps the CUDA
  # toolchain active.
  target_sources(_core PRIVATE
    src/cuda/context.cu
    src/cuda/gemm_cublas.cu
    src/cuda/transfer.cu
    src/cuda/fused.cu
  )
  # _core is created by nanobind_add_module above, before CUDA is enabled, so
  # the global CMAKE_CUDA_ARCHITECTURES default never reached its property and
  # generate fails with "CUDA_ARCHITECTURES is empty for target _core". Set the
  # arch (and the matching C++20 standard) on the existing target explicitly.
  set_target_properties(_core PROPERTIES
    CUDA_ARCHITECTURES 86
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON)
  target_compile_definitions(_core PRIVATE PG_HAS_CUDA=1)
  target_link_libraries(_core PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt)
endif()

install(TARGETS _core LIBRARY DESTINATION peregrine)
