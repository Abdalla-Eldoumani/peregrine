# Numerics

Peregrine is meant to be a drop-in for the NumPy calls it covers, so its
results are held to a written contract rather than to "close enough". This page
states that contract and the guarantees around it. Every clause below is
enforced by the test suite, and the tests use one shared comparison helper per
operation family so that no test can quietly loosen a bound.

## What is guaranteed

**No fast-math.** The library is compiled with `/fp:precise` on MSVC and GCC's
default floating-point model. It is never built with `/fp:fast`, `-ffast-math`,
or `-funsafe-math-optimizations`, so the compiler may not reorder sums, assume
values are finite, or drop signed zeros. The GEMM kernels use fused
multiply-add explicitly, which is the one contraction they rely on.

**NaN and Inf land where NumPy puts them.** A NaN or an infinity in the inputs
produces NaN or an infinity of the same sign at the same output positions as
the matching NumPy expression. `scaled_relu` propagates NaN the way
`np.maximum` does, so a NaN input gives NaN, not 0. For the fused ops, when
several NaNs meet in one element the result carries the payload of the first
NaN operand in the written expression, quieted; that choice does not depend on
array alignment or thread count. Which payload survives inside a matmul or a
reduction is not part of the contract.

**Results do not depend on the thread count.** Every operation returns the same
bits whether it runs on one thread or six. The work is split so that each
output element is always computed by one chain of operations in a fixed order;
threads only decide who computes which elements. The same holds across operand
layouts: a matmul with F-contiguous inputs, transposed views, or negative
strides returns exactly the bits of the C-contiguous product. Repeated calls
with the same inputs return the same bits.

**Peregrine never changes the floating-point mode.** It does not enable
flush-to-zero or denormals-are-zero. If the calling thread has enabled them,
the worker threads adopt the caller's mode for the duration of the call, so
the result is the one a single thread would produce under that mode.

**Reductions are pairwise.** `sum` and `mean` with `axis=None` add in a
pairwise tree with 128-element leaves, the order v3.0.0 used, so their bits
are unchanged from v3.0.0. Along `axis=1` each row is reduced the same way.
Along `axis=0` rows are summed in order within blocks of 128 rows and the
block sums are combined pairwise, which keeps the error growth logarithmic in
the number of blocks. `mean` divides by the count in the result dtype.

**float64 matmul never goes to the GPU automatically.** The RTX 3060 Laptop GPU
runs float64 at 1/64 of its float32 rate, slower than the CPU, so the router
keeps every float64 host product on the CPU. GPU float32 products use full
float32 arithmetic; TF32 is off unless `PEREGRINE_ALLOW_TF32` is set.

## Dtypes

| Inputs | Result |
|---|---|
| float32 and float32 | float32 |
| float64 and float64 | float64 |
| float32 and float64 | float64 |
| an integer and a float | NumPy's promotion (int8, int16, uint8 and uint16 with float32 give float32; other pairs give float64) |
| two integers | float64 (NumPy keeps an integer type; Peregrine is a floating-point library) |
| bool, float16, complex, object | `TypeError` naming the dtype |

## The tolerance contract

Peregrine's kernels add in a different order from NumPy's, so results agree to
within a bound rather than bitwise. The bounds come from the standard
forward-error analysis for each operation, with a margin constant of 4. They
apply to elements where both results are finite; special values are compared
exactly by position first.

**matmul, float64.** For inner dimension `k`:

    |got - ref| <= 1e-12 * |ref| + max(4 * k * eps64 * amax(a) * amax(b), 5e-324)

where `amax` is the largest finite absolute entry of an operand and `eps64` is
2^-52. The `k`-scaled term covers cancellation, where the relative error of a
correct result can be large because the result itself is near zero. The
subnormal floor keeps the bound meaningful for deep-denormal inputs.

**matmul, float32.** Both Peregrine and NumPy are compared against the float64
product of the same inputs. Peregrine must land within four times the larger of
NumPy's own error and `k * eps32 * amax(a) * amax(b)`. Measuring against the
float64 truth, rather than against NumPy, avoids penalizing a result for being
closer to the true value than NumPy's.

**sum and mean.** With `r` the number of elements reduced into each output and
`S` the largest sum of absolute addends feeding one output:

    sum:  |got - ref| <= 1e-12 * |ref| + max(4 * r * eps * S, smallest subnormal)
    mean: the sum bound divided by r

The bound is linear in `r` because NumPy itself sums sequentially along a
non-contiguous axis, and the difference between two results can only be as
small as the larger of their errors.

**axpby, fma3, scaled_relu.** These do a constant amount of arithmetic per
element, so the bound has no size term:

    float64: |got - ref| <= 1e-12 * |ref| + 5e-324
    float32: |got - ref| <= 8 * eps32 * |ref| + 1.4e-45

against the unfused NumPy expression in the same dtype (`a*x + b*y`,
`x*y + z`, `np.maximum(scale*x, 0)`). `fma3` rounds once, so it can be closer
to the true value than NumPy's two roundings; the bound is two-sided, so it
accepts that. `axpby` rounds the two products and then the sum, as NumPy does;
on MSVC builds it is bitwise equal to NumPy, and GCC builds can contract it to
a fused multiply-add and differ by one unit in the last place.

## Edge cases

- `(m, 0) @ (0, n)` returns zeros and `(0, k) @ (k, n)` returns an empty
  `(0, n)` array, as NumPy does.
- A matmul whose every product underflows to -0.0 returns +0.0, as NumPy does,
  because the accumulation starts from +0.0.
- `mean` over an empty axis returns NaN with NumPy's "Mean of empty slice"
  warning.
- The legacy failure sizes of the 2024 kernel (1, 3, 7, 50, 250, 333, 750) stay
  in the test grid for both dtypes.
