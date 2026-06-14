#include "cpu/reduce.hpp"

namespace pg::cpu {

namespace {

// Pairwise summation, NumPy's algorithm class. Error grows as O(log n) * eps
// instead of the O(n) * eps of a naive running sum, and the recursive shape
// here mirrors NumPy's so our result tracks theirs along the contiguous axis.
//
// Base case (n <= 128): eight independent accumulators consume the buffer
// eight-at-a-time, then a fixed-order tree combine. The eight lanes let the
// add pipeline stay full without waiting on a single accumulator's latency
// chain; the combine order is fixed so the result is deterministic regardless
// of build. Accumulators start at +0.0: summing all -0.0 must yield +0.0
// (+0.0 + -0.0 is +0.0 under round-to-nearest), which is NumPy's identity.
// NaN and Inf propagate through the additions exactly as NumPy's do.
//
// Recursive case: split into a left half whose length is a multiple of 8 (so
// the base-case 8-lane stride stays aligned to the split boundary) and a
// right remainder, recurse, add. The split rounds down via & ~7.
template <typename T>
T pairwise_sum(const T* x, int64_t n) {
    if (n <= 128) {
        T acc[8] = {T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0)};
        int64_t i = 0;
        const int64_t blocked = n & ~int64_t{7};
        for (; i < blocked; i += 8) {
            acc[0] += x[i + 0];
            acc[1] += x[i + 1];
            acc[2] += x[i + 2];
            acc[3] += x[i + 3];
            acc[4] += x[i + 4];
            acc[5] += x[i + 5];
            acc[6] += x[i + 6];
            acc[7] += x[i + 7];
        }
        // fixed-order pairwise combine of the eight lanes
        T s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) +
              ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        // scalar remainder for the tail below the 8-lane stride
        for (; i < n; ++i) {
            s += x[i];
        }
        return s;
    }
    const int64_t n2 = (n / 2) & ~int64_t{7};
    return pairwise_sum(x, n2) + pairwise_sum(x + n2, n - n2);
}

} // namespace

template <typename T>
T sum_all(const T* a, int64_t m, int64_t n) {
    // The wrapper guarantees a C-contiguous buffer, so the whole array is one
    // flat contiguous run and sums with a single pairwise pass.
    return pairwise_sum<T>(a, m * n);
}

template <typename T>
void sum_axis(const T* a, T* out, int64_t m, int64_t n, int axis) {
    if (axis == 1) {
        // Each row is contiguous: pairwise per row into out[m]. A zero-width
        // row (n == 0) sums to +0.0, NumPy's empty-sum identity.
        for (int64_t i = 0; i < m; ++i) {
            out[i] = pairwise_sum<T>(a + i * n, n);
        }
        return;
    }
    // axis == 0: reduce over rows into out[n]. NumPy accumulates this
    // non-contiguous axis sequentially (its column sums are sequential, not
    // pairwise), so a row-by-row running sum matches both NumPy's result and
    // its error class. One forward pass keeps each out[j] line warm. Start at
    // +0.0 so an all--0.0 column sums to +0.0; m == 0 leaves the zeros.
    for (int64_t j = 0; j < n; ++j) {
        out[j] = T(0);
    }
    for (int64_t i = 0; i < m; ++i) {
        const T* ai = a + i * n;
        for (int64_t j = 0; j < n; ++j) {
            out[j] += ai[j];
        }
    }
}

template float sum_all<float>(const float*, int64_t, int64_t);
template double sum_all<double>(const double*, int64_t, int64_t);
template void sum_axis<float>(const float*, float*, int64_t, int64_t, int);
template void sum_axis<double>(const double*, double*, int64_t, int64_t, int);

} // namespace pg::cpu
