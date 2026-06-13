cmake_minimum_required(VERSION 3.26)
project(fastmathext LANGUAGES CXX)

option(FME_ENABLE_CUDA "Build the CUDA backend (requires CUDA Toolkit 13.x)" OFF)

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
  src/dispatch/dispatch.cpp
)
target_include_directories(_core PRIVATE src)
target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)

# AVX2 codegen is confined to exactly these four kernel translation units. The
# arch flag is deliberately absent at target level: AVX2 instructions in a TU
# that the no-AVX2 fallback path links produce a module that executes an illegal
# instruction at import on a CPU without AVX2 (the legacy import crash). The
# feature_detect, gemm_naive, dispatch, and module TUs must stay free of AVX2
# codegen so the fallback path runs everywhere.
set(FME_AVX2_SOURCES
  src/cpu/pack.cpp
  src/cpu/microkernel_avx2_f64.cpp
  src/cpu/microkernel_avx2_f32.cpp
  src/cpu/gemm_blis.cpp
)

if (MSVC)
  # /openmp:llvm because the legacy /openmp runtime is OpenMP 2.0 and rejects
  # modern pragmas. No /fp:fast: it changes NaN and Inf semantics and the
  # library promises NumPy-matching results.
  target_compile_options(_core PRIVATE /O2 /openmp:llvm /permissive- /W4)
  set_source_files_properties(${FME_AVX2_SOURCES} PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
  # No -ffast-math for the same reason.
  target_compile_options(_core PRIVATE -O3 -Wall -Wextra)
  set_source_files_properties(${FME_AVX2_SOURCES} PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
endif()

if (FME_ENABLE_CUDA)
  if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/context.cu")
    message(FATAL_ERROR
      "FME_ENABLE_CUDA=ON but src/cuda sources are not implemented yet. "
      "Build with the option OFF until the cuda backend lands.")
  endif()
  enable_language(CUDA)
  # sm_86 is the GA106 in the primary dev machine. PTX for 8.6 keeps newer
  # consumer parts working through JIT.
  set(CMAKE_CUDA_ARCHITECTURES 86)
  find_package(CUDAToolkit 13 REQUIRED)
  target_sources(_core PRIVATE
    src/cuda/context.cu
    src/cuda/gemm_cublas.cu
  )
  target_compile_definitions(_core PRIVATE FME_HAS_CUDA=1)
  target_link_libraries(_core PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt)
endif()

install(TARGETS _core LIBRARY DESTINATION fastmathext)
