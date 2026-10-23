# Architecture

Peregrine has two layers. The Python package `peregrine` owns everything a
user sees: argument checking, dtype promotion, layout handling, error
messages, and the decision of where a product runs. The native module
`peregrine._core`, built with nanobind against the stable ABI, owns the
kernels and knows nothing about Python semantics. One abi3 wheel serves
CPython 3.12 and later.

## The call path

A small matmul on this machine costs well under a microsecond in arithmetic,
so the cost of getting from Python to the kernel matters as much as the kernel.
Each public operation therefore starts with a native fast entry, a C function
registered with `METH_FASTCALL` that reads its arguments straight from the
interpreter's argument array. A fast entry handles the common case (host NumPy
arrays of float32 or float64 in a layout the kernel reads directly, with valid
shapes) and runs the kernel. For anything else it returns `NotImplemented`,
and the Python function takes over: promotion, layout normalization, warnings,
and every error message live there and only there, so the fast entry can
decline anything unusual without risking a different error from the slow
path. A property test runs both paths on the same random inputs and requires
the same bits, types, errors, and warnings.

Setting `PEREGRINE_DISABLE_FAST_ENTRIES` to a non-empty value before import
sends every call through the Python path.

The fast entries release the GIL only above a work threshold (32768
multiply-adds for matmul, 4096 elements for the other operations), because
below it releasing and reacquiring the GIL costs a large fraction of the
call.

## Output memory

CPU results are ordinary NumPy arrays that own their memory. For results of
512 KiB and up, Peregrine allocates through its own NumPy memory handler,
installed only around its own allocations, which keeps freed blocks and hands
an exact-size block back to the next request. A large result is otherwise
fresh memory from the operating system, and faulting in its pages can take
longer than the arithmetic that fills them. The cache holds at most 256 MiB;
`PEREGRINE_OUTPUT_CACHE_MB` changes the cap, and 0 turns it off.

## CPU matmul

`choose_gemm_path` picks one of four kernels from the shape alone, so the
choice never depends on the thread count:

- **small**, when every dimension is at most 96: register-tiled code that
  reads the operands in place (or packs a 16 KiB panel when the layout would
  thrash L1). Products of at least 2^16 vector multiply-adds split their rows
  across two threads.
- **skinny**, when an output dimension or `k` is at most 4: these products are
  bound by reading the long operand, so the kernel streams it once instead of
  packing it. Above 12 MiB of operand it keeps only a few streams in flight,
  since this CPU's memory bandwidth peaks with three or four readers.
- **packed**, everything else: a five-loop blocked GEMM in the style of BLIS,
  with 6x8 (float64) and 6x16 (float32) AVX2 microkernels. Block sizes come
  from a sweep on this machine: float64 uses MC=48, KC=256, NC=2048; float32
  uses MC=48, KC=256, NC=4080. Threads share the work on a 2-D grid of row and
  column tiles. There is no barrier per block step; each thread waits only for
  the B panel it needs and for the buffer it is about to overwrite, and the
  kernel prefetches the next B panel while it computes the current one.
- **naive**, when the CPU or OS lacks AVX2 and FMA: a plain strided loop. It is
  also the correctness reference in the tests and is never vectorized by hand.

Feature detection checks CPUID for AVX2 and FMA and also checks, through
OSXSAVE and XGETBV, that the operating system saves the YMM registers, so a
machine with AVX2 disabled in the OS takes the naive path instead of faulting.
`PEREGRINE_DISABLE_AVX2` forces the naive path.

Every kernel reads operand strides, so F-contiguous operands and transposed
views are used without a copy.

## Threading

The thread count follows, in order: `set_num_threads`, then
`PEREGRINE_NUM_THREADS`, then the first entry of `OMP_NUM_THREADS`, then the
number of physical cores in the process's affinity mask. Each call sizes its
team from the work it has, so a small call does not wake six threads. Peregrine
never calls `omp_set_num_threads`, because on MSVC's runtime that changes the
setting for every library in the process.

Results do not depend on the team size: each output element is computed by
one thread, in a fixed order, and threads only divide the elements between
them. See [numerics.md](numerics.md).

## Reductions, transpose, fused operations

`sum` and `mean` use a pairwise order shared by an AVX2 kernel and a scalar
twin, so both give the same bits. `transpose` writes tiles through vector lane
loads and stores, with a separate path for outputs under 32 KiB. The fused
operations (`axpby`, `fma3`, `scaled_relu`) run one elementwise driver that
handles the head and tail of the array with masked vector loads and stores,
so no element is computed by scalar code and the bits never depend on
alignment. Above 12 MiB of traffic they use streaming stores, and above 24 MiB
they cap the team because more threads only contend for memory bandwidth.

## GPU path

The CUDA backend is optional (`PG_ENABLE_CUDA=ON` at build time) and targets
sm_86. It keeps one CUDA stream per process for allocation, compute, and
frees, so device operations queue and return without host waits, plus two side
streams that the staged host product uses to overlap uploads and downloads
with compute. Device memory comes from a stream-ordered pool. Host results of
GPU work are written into page-locked buffers lent to NumPy, so the copy goes
straight from the device into the result instead of through the driver's
staging buffer into freshly allocated, not yet faulted-in pageable memory. Requests of 256 MiB and
more are checked against free device memory first, so an oversized request
raises an error instead of spilling into shared system memory.

`peregrine.to_device` returns a device array that supports `__dlpack__`.
Device-resident matmul and fused operations never touch the host. Errors from
asynchronous work surface at the next point that waits for the stream.

## Routing and calibration

With the default backend `"auto"`, a host float32 matmul goes to the GPU only
when the predicted staged time (upload, compute, download, and a fixed
per-call cost) beats the predicted CPU time. Both predictions come from
`peregrine.calibrate()`, which times the CPU kernel, the device kernel, the
staged route, and the transfers on this machine and stores the result in a
per-machine cache file. Without a calibration file every host product stays
on the CPU. float64 host products never auto-route to the GPU.

`set_backend("cpu")` or `set_backend("cuda")` overrides the router, and
`PEREGRINE_BACKEND` sets the starting backend.
