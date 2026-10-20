# Peregrine

![python](https://img.shields.io/badge/python-3.12%2B-blue)
![C++](https://img.shields.io/badge/C%2B%2B-20-blue)
![CUDA](https://img.shields.io/badge/CUDA-optional-76B900)
![license](https://img.shields.io/badge/license-MIT-green)

Peregrine is a linear algebra library for Python. It covers a small set of NumPy
operations (the matrix product, transpose, sum, mean, and three fused
elementwise expressions) with its own AVX2 kernels, and adds an optional CUDA
backend for float32 matrix products. It reads NumPy arrays in place, returns
NumPy arrays, and holds every result to NumPy's within a written error bound,
bit for bit wherever the order of operations is the same.

Version 3.1.0. Windows (MSVC) is the primary platform; Linux builds with GCC.

## What it provides

- `matmul(a, b, *, out=None)` for float32 and float64, matching `numpy.matmul`
  including zero-sized dimensions and NaN and Inf, for operands in any memory
  layout (C or Fortran order, transposed or sliced views), read without a copy.
- `transpose(a)`, `sum(a, *, axis=None)` and `mean(a, *, axis=None)`.
- Fused elementwise operations that compute an expression in one pass:
  `axpby(x, y, *, a, b, out=None)` for `a*x + b*y`, `fma3(x, y, z, *, out=None)`
  for `x*y + z` with a single rounding, and `scaled_relu(x, *, scale, out=None)`
  for `maximum(scale*x, 0)`.
- Thread control: `set_num_threads(n)`, `get_num_threads()`, and the
  `PEREGRINE_NUM_THREADS` environment variable.
- An optional CUDA backend: device arrays (`to_device`, `from_device`,
  `__dlpack__`), device-resident matmul and fused operations, and an `"auto"`
  backend that sends a host float32 product to the GPU only when this machine's
  calibration says the GPU wins after paying for the transfers. float64 never
  goes to the GPU automatically.

The plain reference kernel stays in the library: it is the oracle the tests
check every kernel against, and the path taken on a CPU without AVX2.

## Install

Each GitHub Release carries wheels for CPython 3.12 and later (one abi3 wheel
per platform: Windows x86-64 and Linux x86-64, manylinux_2_28), a `SHA256SUMS`
file, and build attestations. The wheels are CPU-only and bundle the C++ and
OpenMP runtimes they load. Install one from its release URL:

```bash
pip install https://github.com/Abdalla-Eldoumani/peregrine/releases/download/v3.1.0/peregrine-3.1.0-cp312-abi3-win_amd64.whl
```

To check a downloaded file against its attestation:

```bash
gh attestation verify peregrine-3.1.0-cp312-abi3-win_amd64.whl --repo Abdalla-Eldoumani/peregrine
```

From source, with a C++20 compiler (MSVC from Visual Studio 2022 or later, or
GCC) and CMake 3.26 or later:

```bash
pip install .
```

The CUDA backend needs the CUDA Toolkit 12.8 and an NVIDIA driver new enough for
it; the kernels target compute capability 8.6:

```bash
pip install . --config-settings=cmake.define.PG_ENABLE_CUDA=ON
```

`pg.has_cuda()` reports whether a usable device is present at runtime. On a
CPU-only build it returns False and every operation runs on the CPU.

## Quickstart

```python
import numpy as np
import peregrine as pg

a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.eye(2)

# Matrix product (host arrays in, host array out).
pg.matmul(a, b)

# Transpose returns an owned copy, not a view.
pg.transpose(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

# Reductions: axis is keyword-only (None, 0, or 1).
pg.sum(a)            # scalar
pg.sum(a, axis=0)    # length-n vector
pg.mean(a, axis=1)   # length-m vector

# Fused elementwise ops. Array operands are positional; scalars are keyword-only.
x = np.array([[1.0, 2.0], [3.0, 4.0]])
y = np.array([[10.0, 20.0], [30.0, 40.0]])
z = np.array([[1.0, 1.0], [1.0, 1.0]])
pg.axpby(x, y, a=2.0, b=-1.0)   # a*x + b*y
pg.fma3(x, y, z)                # x*y + z, one rounding
pg.scaled_relu(x, scale=3.0)    # maximum(scale*x, 0)
```

With a CUDA build and a usable device:

```python
import numpy as np
import peregrine as pg

if pg.has_cuda():
    a = np.random.default_rng(0).standard_normal((1024, 1024)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((1024, 1024)).astype(np.float32)

    # Place operands on the device; matmul of two device arrays returns a device
    # array (no implicit transfer back).
    da = pg.to_device(a)
    db = pg.to_device(b)
    dc = pg.matmul(da, db)
    c = pg.from_device(dc)

    # Measure this machine and route host float32 products by the result.
    pg.calibrate(force=True)
    pg.set_backend("auto")
```

`set_backend` accepts `"auto"`, `"cpu"`, or `"cuda"`. `calibrate` measures a square
GEMM per size and dtype and caches the crossover the auto policy reads.

### Output memory

A large result is new memory from the operating system, and the first write to
each 4 KiB page of it costs a page fault, which for a
transpose or a fused op over a few MiB can take longer than the arithmetic.
Peregrine therefore keeps the buffers of large results you have freed (512 KiB
and up) and hands them to the next result of the same size. Results are still
ordinary NumPy-owned arrays; NumPy's own allocations and yours are untouched.
The price is memory: up to 256 MiB of freed buffers stay with the process
instead of going back to the system. Set `PEREGRINE_OUTPUT_CACHE_MB` before
importing Peregrine to change that cap, or to `0` to turn the cache off.

### Threads

Peregrine runs its kernels on a thread team sized to each call's work, so a
small call stays on one thread. The thread count comes from, in order,
`pg.set_num_threads(n)`, the `PEREGRINE_NUM_THREADS` environment variable, the
first entry of `OMP_NUM_THREADS`, and otherwise the number of physical cores
the process may run on. Results are bitwise the same for every thread count.

### Configuration

| Variable | Effect |
| --- | --- |
| `PEREGRINE_NUM_THREADS` | Thread count, unless `set_num_threads` is called. |
| `PEREGRINE_BACKEND` | Starting backend: `auto` (default), `cpu`, or `cuda`. |
| `PEREGRINE_CACHE_DIR` | Where `calibrate()` stores this machine's timings. |
| `PEREGRINE_OUTPUT_CACHE_MB` | Cap on kept result buffers (default 256; 0 turns it off). |
| `PEREGRINE_ALLOW_TF32` | Lets GPU float32 products use TF32 (off by default). |
| `PEREGRINE_DISABLE_FAST_ENTRIES` | Sends every call through the Python path (for testing). |
| `PEREGRINE_DISABLE_AVX2` | Uses the reference kernels (for testing). |

### Numerics

Peregrine is never built with fast-math options. NaN and Inf land where NumPy
puts them, results do not depend on the thread count or on operand layout, and
reductions add in a pairwise order. Where the order of operations differs from
NumPy's, results agree within a bound derived from the standard error analysis
of each operation. [docs/numerics.md](docs/numerics.md) states the contract
exactly.

## Results

Every table in this section is generated by `benchmarks/report.py` from the
result files in `benchmarks/results/`, and `python benchmarks/report.py --check`
fails when a table and its file disagree. A ratio is NumPy's median time divided
by Peregrine's on the same inputs, with both libraries limited to the same
number of threads, so a ratio above one means Peregrine is faster. Every cell
has a target, and cells that miss it are listed below with the reason. The full
matrix, with spreads and minimums for every cell, is in
[docs/benchmarks.md](docs/benchmarks.md).

### Headline

<!-- report:headline -->
Representative cells from the full matrix. Times are medians of the timed samples; both libraries ran at the thread count in the threads column. The ratio is the baseline's median time divided by Peregrine's, so above 1.00x Peregrine is faster; a percentage compares Peregrine with its own reference route. The auto rows come from the CUDA build, the rest from the CPU-only build.

| operation | dtype | shape | threads | baseline | baseline median | Peregrine median | ratio | target | status |
|---|---|---|---|---|---|---|---|---|---|
| matmul a @ b | float32 | 16x16x16 | 6 | - | - | - | - | - | not measured |
| matmul a @ b | float32 | 64x64x64 | 6 | - | - | - | - | - | not measured |
| matmul a @ b | float32 | 256x256x256 | 6 | - | - | - | - | - | not measured |
| matmul a @ b | float32 | 512x512x512 | 6 | - | - | - | - | - | not measured |
| matmul a @ b | float32 | 2048x2048x2048 | 6 | - | - | - | - | - | not measured |
| matmul a @ b | float64 | 16x16x16 | 6 | - | - | - | - | - | not measured |
| matmul a @ b | float64 | 64x64x64 | 6 | - | - | - | - | - | not measured |
| matmul a @ b | float64 | 256x256x256 | 6 | - | - | - | - | - | not measured |
| matmul a @ b | float64 | 512x512x512 | 6 | - | - | - | - | - | not measured |
| matmul a @ b | float64 | 2048x2048x2048 | 6 | - | - | - | - | - | not measured |
| matmul a @ b | float32 | 1x4096x4096 | 6 | - | - | - | - | - | not measured |
| matmul a @ b | float32 | 4096x64x4096 | 6 | - | - | - | - | - | not measured |
| matmul a @ b | float64 | 1000x333x750 | 6 | - | - | - | - | - | not measured |
| matmul a.T @ b | float32 | 512x512x512 | 6 | - | - | - | - | - | not measured |
| matmul a @ b.T | float32 | 512x512x512 | 6 | - | - | - | - | - | not measured |
| matmul a @ b, both F-order | float32 | 512x512x512 | 6 | - | - | - | - | - | not measured |
| transpose | float32 | 64x64 | 6 | - | - | - | - | - | not measured |
| transpose | float32 | 2048x2048 | 6 | - | - | - | - | - | not measured |
| transpose | float64 | 1000x3000 | 6 | - | - | - | - | - | not measured |
| sum, axis=None | float32 | 64x64 | 6 | - | - | - | - | - | not measured |
| sum, axis=None | float64 | 4096x4096 | 6 | - | - | - | - | - | not measured |
| mean, axis=0 | float32 | 1024x1024 | 6 | - | - | - | - | - | not measured |
| mean, axis=1 | float64 | 1000000x16 | 6 | - | - | - | - | - | not measured |
| axpby, new output | float32 | 1024x1024 | 6 | - | - | - | - | - | not measured |
| axpby, out= | float32 | 1024x1024 | 6 | - | - | - | - | - | not measured |
| fma3, new output | float32 | 1024x1024 | 6 | - | - | - | - | - | not measured |
| fma3, out= | float32 | 1024x1024 | 6 | - | - | - | - | - | not measured |
| scaled_relu, new output | float32 | 1024x1024 | 6 | - | - | - | - | - | not measured |
| scaled_relu, out= | float32 | 1024x1024 | 6 | - | - | - | - | - | not measured |
| matmul a @ b, auto route | float32 | 2048x2048x2048 | 6 | - | - | - | - | - | not measured |
| matmul a @ b, auto route | float32 | 4096x4096x4096 | 6 | - | - | - | - | - | not measured |

Not measured: `benchmarks/results/refbox_fused_cpu_v310.json` is absent.

Not measured: `benchmarks/results/refbox_gpu_cuda_v310.json` is absent.

Not measured: `benchmarks/results/refbox_matmul_layout_cpu_v310.json` is absent.

Not measured: `benchmarks/results/refbox_matmul_rect_cpu_v310.json` is absent.

Not measured: `benchmarks/results/refbox_matmul_square_cpu_v310.json` is absent.

Not measured: `benchmarks/results/refbox_reduce_cpu_v310.json` is absent.

Not measured: `benchmarks/results/refbox_transpose_cpu_v310.json` is absent.
<!-- /report:headline -->

### The whole matrix at a glance

Each square is the median ratio over the cells of one operation in one size
range, on a log color scale centered at parity. Gray squares are cells the
protocol closed as noisy, which carry no number.

![Ratio of NumPy time to Peregrine time by operation and size](benchmarks/results/ratio_heatmap.png)

### Matrix multiply throughput

Square matmul throughput against size for Peregrine and NumPy on the CPU, and
for Peregrine's device-resident GPU path, with the CPU's measured FMA peak drawn
as a ceiling.

![GFLOP/s against n for square matmul](benchmarks/results/gflops_vs_n.png)

### Losses and closures

<!-- report:losses -->
**Below 1.00x.** Every measured ratio below one: the baseline's median time divided by Peregrine's, both libraries at the thread count in the cell id, with the reason.

No measured cell is below 1.00x against any baseline.

**Closed below target.** Cells that miss their target and are closed at a measured ceiling or deferred; every number is computed from the results files.

None.

**Below target, not closed.**

None.

**Noisy.** These cells stayed above the CV limit on every attempt, including the fresh-process retry, so the protocol publishes no number for them.

None.

Not measured: `benchmarks/results/refbox_matmul_square_cpu_v310.json` is absent.

Not measured: `benchmarks/results/refbox_matmul_rect_cpu_v310.json` is absent.

Not measured: `benchmarks/results/refbox_matmul_layout_cpu_v310.json` is absent.

Not measured: `benchmarks/results/refbox_transpose_cpu_v310.json` is absent.

Not measured: `benchmarks/results/refbox_reduce_cpu_v310.json` is absent.

Not measured: `benchmarks/results/refbox_fused_cpu_v310.json` is absent.

Not measured: `benchmarks/results/refbox_overhead_cpu_v310.json` is absent.

Not measured: `benchmarks/results/refbox_gemm_1t_cpu_v310.json` is absent.

Not measured: `benchmarks/results/refbox_gpu_cuda_v310.json` is absent.

Not measured: `benchmarks/results/refbox_overhead_cuda_v310.json` is absent.
<!-- /report:losses -->

### v3.0.0 to v3.1.0

<!-- report:compare -->
Not measured: `benchmarks/results/refbox_quick_cpu_v300.json` is absent.

Not measured: `benchmarks/results/refbox_quick_cpu_v310.json` is absent.

Not measured: `benchmarks/results/refbox_quick_cuda_v300.json` is absent.

Not measured: `benchmarks/results/refbox_quick_cuda_v310.json` is absent.
<!-- /report:compare -->

## Method

Both libraries run in one process on the same ndarray inputs, built once per
seed outside the timed region. Samples interleave in short blocks of one library
then the other, mirrored so neither always goes first, and each sample is a
batch of calls through the same timing loop for both. After every switch the
incoming library runs untimed for a short rewarm. Every sample's output is
checked bit for bit against the first output, which is itself checked against
the test suite's oracle. A cell whose spread exceeds the coefficient-of-variation
limit is retried, then run once more in a fresh process, and if it is still
noisy it is reported as noisy with no number. The process runs at high priority
on one logical CPU per physical core, and samples taken while the processor
clock sits outside a band set from the ceiling tool are discarded. The exact
settings are below; the full method and the commands to reproduce every file
are in [docs/benchmarks.md](docs/benchmarks.md).

<!-- report:environment -->
Not measured: `benchmarks/results/refbox_matmul_square_cpu_v310.json` is absent.
Not measured: `benchmarks/results/refbox_matmul_rect_cpu_v310.json` is absent.
Not measured: `benchmarks/results/refbox_matmul_layout_cpu_v310.json` is absent.
Not measured: `benchmarks/results/refbox_transpose_cpu_v310.json` is absent.
Not measured: `benchmarks/results/refbox_reduce_cpu_v310.json` is absent.
Not measured: `benchmarks/results/refbox_fused_cpu_v310.json` is absent.
Not measured: `benchmarks/results/refbox_overhead_cpu_v310.json` is absent.
Not measured: `benchmarks/results/refbox_gemm_1t_cpu_v310.json` is absent.
Not measured: `benchmarks/results/refbox_gpu_cuda_v310.json` is absent.
Not measured: `benchmarks/results/refbox_overhead_cuda_v310.json` is absent.
Not measured: `benchmarks/results/refbox_quick_cpu_v300.json` is absent.
Not measured: `benchmarks/results/refbox_quick_cpu_v310.json` is absent.
Not measured: `benchmarks/results/refbox_quick_cuda_v300.json` is absent.
Not measured: `benchmarks/results/refbox_quick_cuda_v310.json` is absent.
<!-- /report:environment -->

## Hardware

The ceilings below come from a small measurement tool in `tools/ceilings`, run
on the same machine as the benchmarks.

<!-- report:hardware -->
Not measured: `benchmarks/results/refbox_ceilings.json` is absent.
<!-- /report:hardware -->

## Development

```bash
pip install -e ".[test]"
python -m pytest tests -q
```

`PEREGRINE_DISABLE_FAST_ENTRIES=1` runs the suite through the Python path and
`PEREGRINE_DISABLE_AVX2=1` through the reference kernels; CI runs both, plus GCC
on Linux and a build under AddressSanitizer and UndefinedBehaviorSanitizer.
[docs/architecture.md](docs/architecture.md) describes how the pieces fit, and
[docs/benchmarks.md](docs/benchmarks.md) how the numbers above are measured and
how to reproduce them.

## License

MIT; see LICENSE. The wheels include third-party code and runtimes listed in
THIRD_PARTY_NOTICES.md.
