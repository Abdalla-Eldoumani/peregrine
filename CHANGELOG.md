# Changelog

## 3.1.0

A performance release. The API is unchanged except for the additions listed
below; results keep the same numerics contract (see `docs/numerics.md`).
Measured speedups, with their sizes, dtypes and thread counts, are in the
README and `docs/benchmarks.md`.

### Added

- `out=` on `matmul`, `axpby`, `fma3` and `scaled_relu`, writing the result
  into an existing array of the right shape and dtype.
- `set_num_threads(n)` and `get_num_threads()`, and the
  `PEREGRINE_NUM_THREADS` environment variable. Without them Peregrine uses the
  first entry of `OMP_NUM_THREADS`, or else the physical cores available to the
  process.
- Device arrays support `__dlpack__`.
- Release wheels for CPython 3.12 and later (one abi3 wheel per platform) on
  Windows x86-64 and manylinux_2_28 x86-64, published on the GitHub Release with
  SHA256 checksums and build attestations.

### Changed

- Every operation now enters through a native fast path that handles the common
  case directly, which removes most of the fixed cost of a small call. Anything
  outside that case falls back to the Python implementation, which owns every
  error message, so errors and warnings are unchanged.
- matmul has four kernels chosen from the shape: a register-tiled kernel for
  products up to 96 in every dimension, a streaming kernel for thin outputs and
  tiny inner dimensions, a re-tuned blocked engine (6x16 float32 microkernel,
  new block sizes, no per-step barrier), and the reference loop for machines
  without AVX2.
- matmul reads F-contiguous operands and transposed views without copying them.
- New AVX2 kernels for `transpose`, `sum`, `mean` and the fused operations, with
  multithreading sized to each call's work.
- Results of 512 KiB and more reuse freed blocks of the same size, which removes
  page-fault time from repeated calls. `PEREGRINE_OUTPUT_CACHE_MB` sets the cap
  (256 MiB by default; 0 turns it off).
- The GPU path runs on one stream, overlaps uploads and downloads with compute
  for large float32 products, and returns host results in page-locked memory.
- Host float32 products are routed on their work and bytes moved, from a
  calibration that times the native paths; the calibration cache format is
  version 2, so older caches are recalibrated.
- The Windows module links Microsoft's redistributable OpenMP runtime
  (vcomp140), which the Windows wheel bundles.

### Fixed

- `sum` along `axis=0` now combines 128-row blocks pairwise, so its error grows
  with the logarithm of the row count instead of linearly; results for 130 rows
  or more differ in the last bits from 3.0.0.
- When NaNs meet in a fused operation, the result's NaN payload no longer
  depends on the array's alignment.
- Auto routing no longer re-reads the machine signature and the calibration
  file on every call; on CUDA builds every auto-routed host product used to pay
  for a subprocess and a file read.
- Feature detection takes the reference path when the operating system has not
  enabled AVX state, instead of faulting on the first AVX2 instruction.

## 3.0.0

The v3 rebuild of the 2024 project: nanobind bindings with zero-copy NumPy
input, packed AVX2 GEMM kernels for float32 and float64, `transpose`, `sum`,
`mean`, the fused `axpby`, `fma3` and `scaled_relu`, an optional cuBLAS
backend with device-resident arrays, and a per-machine calibration that
decides when a host float32 product runs on the GPU.
