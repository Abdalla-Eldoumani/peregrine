# FastMathExt

FastMathExt is a heterogeneous linear algebra library for Python with a
NumPy-compatible surface. It dispatches matrix and elementwise work across packed
AVX2 CPU kernels and an optional cuBLAS CUDA backend behind one zero-copy API. The
native core takes views of C-contiguous float32 and float64 arrays with no copy;
the Python layer adds dtype promotion, contiguity normalization, clear errors, and
a per-machine routing policy that decides CPU versus GPU from calibrated timings.

Version 3.0.0. Windows is the primary platform (MSVC); the CPU build also runs on
Linux.

## What it provides

- `matmul` for float32 and float64, matching `numpy.matmul` values including
  zero-sized dimensions and NaN/Inf propagation.
- `transpose`, `sum`, `mean` reductions with NumPy-matching results.
- Fused elementwise ops `axpby`, `fma3`, `scaled_relu` that compute a chain in one
  memory pass.
- An optional CUDA backend: device-resident `fme.Array` handles, `to_device` /
  `from_device` transfers, and an auto policy that routes a host float32 product to
  the GPU only when the calibrated crossover says the device wins after the
  transfer is paid. float64 never auto-routes to the GPU.

The naive reference kernel is permanent: it is the correctness oracle every result
is checked against, and the fallback when AVX2 is absent.

## Install

The default build is CPU-only and needs no CUDA toolkit:

```bash
pip install -e .
```

To build with the CUDA backend, enable it at configure time. This needs the
CUDA Toolkit 12.8 (nvcc and the cuBLAS/cublasLt libraries) and an NVIDIA driver
new enough for it; the kernels target compute capability 8.6:

```bash
pip install -e . --config-settings=cmake.define.FME_ENABLE_CUDA=ON
```

`fme.has_cuda()` reports whether a usable device is present at runtime. On a
CPU-only build it returns False and every operation runs on the CPU.

## Quickstart

```python
import numpy as np
import fastmathext as fme

a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.eye(2)

# Matrix product (host arrays in, host array out).
fme.matmul(a, b)

# Transpose returns an owned copy, not a view.
fme.transpose(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

# Reductions: axis is keyword-only (None, 0, or 1).
fme.sum(a)            # scalar
fme.sum(a, axis=0)    # length-n vector
fme.mean(a, axis=1)   # length-m vector

# Fused elementwise ops. Array operands are positional; scalars are keyword-only.
x = np.array([[1.0, 2.0], [3.0, 4.0]])
y = np.array([[10.0, 20.0], [30.0, 40.0]])
z = np.array([[1.0, 1.0], [1.0, 1.0]])
fme.axpby(x, y, a=2.0, b=-1.0)   # a*x + b*y
fme.fma3(x, y, z)                # x*y + z, one rounding
fme.scaled_relu(x, scale=3.0)    # maximum(scale*x, 0)
```

With a CUDA build and a usable device:

```python
import numpy as np
import fastmathext as fme

if fme.has_cuda():
    a = np.random.default_rng(0).standard_normal((1024, 1024)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((1024, 1024)).astype(np.float32)

    # Place operands on the device; matmul of two device arrays returns a device
    # array (no implicit transfer back).
    da = fme.to_device(a)
    db = fme.to_device(b)
    dc = fme.matmul(da, db)
    c = fme.from_device(dc)

    # Measure this machine and route host float32 products by the result.
    fme.calibrate(force=True)
    fme.set_backend("auto")
```

`set_backend` accepts `"auto"`, `"cpu"`, or `"cuda"`. `calibrate` measures a square
GEMM per size and dtype and caches the crossover the auto policy reads.

## Results

All numbers below are regenerated from the committed results JSON by
`benchmarks/report.py` and the charts by `benchmarks/plots.py`; none are typed by
hand. They include the regimes where NumPy or OpenBLAS wins, because float64
mid-size parity is the stated target, not a win. The headline statistic is the
floor (min over reps), for the reason given under Methodology.

The CPU tables come from a CPU-only build; the GPU tables from a CUDA build on the
same machine. The two are not directly comparable per operation, only per regime.

### float64 matmul (CPU)

| n | FastMathExt floor (GFLOP/s) | OpenBLAS floor (GFLOP/s) | floor ratio |
| --- | --- | --- | --- |
| 256 | 29.0 | 97.3 | 0.30 |
| 512 | 40.9 | 93.5 | 0.44 |
| 1024 | 65.4 | 104.2 | 0.63 |
| 2048 | 76.2 | 112.5 | 0.68 |

At n=2048 the fresh per-rep sweep measures 0.58 of OpenBLAS on the floor and 0.74
on the median. float64 mid-size parity is the target; FastMathExt does not beat
OpenBLAS here.

### Small-matrix float32 (CPU)

The small-matrix path loses to NumPy at every tiny size: NumPy dispatches to an
OpenBLAS small-GEMM, and the irreducible Python plus binding round-trip dominates
at this scale.

| n | speedup vs NumPy | outcome |
| --- | --- | --- |
| 8 | 0.52x | loses to NumPy |
| 16 | 0.39x | loses to NumPy |
| 32 | 0.18x | loses to NumPy |
| 64 | 0.08x | loses to NumPy |

### Thread scaling (CPU)

1 to 6 thread scaling at n=1024 (f64): 2.87x floor speedup, below the 4x the
four-core plateau would give and far below linear. More threads do not help past
four cores here.

### GPU matmul

Device-resident float32 (without transfer), warm, versus NumPy CPU f32. CuPy is
not installed on this machine, so the GPU baseline is NumPy CPU f32.

| n | GFLOP/s | speedup vs NumPy CPU f32 |
| --- | --- | --- |
| 256 | 890 | 6.83x |
| 333 | 2282 | 11.36x |
| 512 | 4416 | 22.76x |
| 750 | 5531 | 26.82x |
| 1024 | 6125 | 28.21x |
| 2048 | 4873 | 23.54x |

With the host round-trip (to_device + matmul + from_device), warm. The small sizes
lose to the transfer cost, which is why the auto policy keeps small host arrays on
the CPU:

| n | speedup vs NumPy CPU f32 | outcome |
| --- | --- | --- |
| 256 | 0.35x | loses to transfer cost |
| 333 | 0.35x | loses to transfer cost |
| 512 | 0.48x | loses to transfer cost |
| 750 | 0.44x | loses to transfer cost |
| 1024 | 2.93x | wins after transfer |
| 2048 | 5.29x | wins after transfer |

### Fused 3-op chain

Chain `scaled_relu(fma3(axpby(x,y,a,b),z))` at 8M float32 elements. The fused
kernel makes one memory pass where the unfused NumPy chain makes several:

| backend | speedup vs NumPy unfused chain |
| --- | --- |
| CPU (floor) | 3.24x |
| GPU device-resident | 71.37x |

### Autotuned blocking

Autotuned CPU blocking on this machine (from the saved sweep): mc=48, kc=192,
nc=2048 (selected on min_gflops).

### Charts

The three charts are regenerated from the same JSON:

- `benchmarks/results/gflops_vs_n.png`: GFLOP/s vs n per backend.
- `benchmarks/results/speedup_bars.png`: speedup per regime, wins and losses.
- `benchmarks/results/crossover.png`: the host float32 crossover, CPU compute vs
  GPU compute plus transfer, showing the size where the device wins for a host
  array.

## Methodology

Every contender in a series takes an ndarray in and returns an ndarray out, on
both sides, with the same dtype, layout, and data; inputs are built once with a
seeded generator outside the timed region, so no conversion time is counted. CPU
timing uses `perf_counter_ns`; GPU device work is timed with CUDA event pairs after
a stream sync, never a wall clock around an asynchronous launch. The host
round-trip series is the one wall-clocked GPU case, because the transfer
synchronizes at the boundary. Each series is checked against the NumPy oracle on
the same inputs before any time is recorded, so an unverified series never reaches
a table; the published `verified` flag is set only after that toleranced check
passes.

The statistic reported depends on the regime, because the timers differ:

- CPU regimes (float64 matmul, small-matrix, thread scaling): the floor (min over
  reps) of FastMathExt against the floor of NumPy. On this machine a background
  antivirus preempts an OpenMP worker at the fork/join barrier, which inflates the
  median and maximum of every multi-threaded CPU series; the floor is the
  noise-free best case, and the coefficient of variation, recorded alongside,
  documents the residual rather than hiding it.
- GPU device-resident (the without-transfer headline): the warm cudaEvent-timed
  mean over reps for the device, against the NumPy CPU f32 median. cudaEvent
  timing reports an averaged warm figure, so this regime quotes the device mean
  rather than a per-rep min.
- GPU with-transfer (the host round-trip): the warm wall-clock floor of the full
  to_device + matmul + from_device round-trip, against the NumPy CPU f32 floor.

Median, IQR, and coefficient of variation are recorded alongside every regime as a
noise readout. The size grid includes non-multiples of the vector width so a
kernel cannot hide its tail path.

## Hardware

| Component | Value |
| --- | --- |
| Platform | Windows-10-10.0.19045-SP0 |
| CPU | Intel64 Family 6 Model 165 Stepping 2, GenuineIntel |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| CUDA driver | 610.47 |
| Python | 3.12.4 |
| NumPy | 2.4.6 |
| BLAS | openblas 0.3.31.188.0 |

## License

MIT. See LICENSE.
