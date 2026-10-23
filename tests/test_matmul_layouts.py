"""matmul reads its operands in place in any memory layout.

The kernels read A and B through element strides and pack them into the same
panels whatever the layout, so a transposed, F-ordered, sliced, or reversed
operand gives bitwise the result of the same values in C order, on the native
fast entry and on the Python path alike, at every team size, and neither path
copies it. Only what the kernels cannot read (a misaligned element, another
dtype or byte order, a byte stride that is not a whole number of elements) is
copied first.

A length-1 dim may carry any stride in NumPy, even one that is not a whole
number of elements, and still be flagged contiguous and aligned; those
operands must give the same bits too.

The shapes cover the direct small path (every dim at most 96), the packed
path inside one KC panel and across several, vector-like products, and the
team partition: n = 256, 333, 1000, and the long-k (64, 4096, 64).
"""

import hashlib
import tracemalloc
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import peregrine as pg

DTYPES = [np.float64, np.float32]


def _kc(dtype):
    return pg._core._get_gemm_blocking(f32=np.dtype(dtype) == np.float32)["kc"]


# name -> builder of a view holding exactly the values of the C-ordered x
def _c(x):
    return x


def _f(x):
    return np.asfortranarray(x)


def _t(x):
    # the transposed view of a C-ordered array, the a.T in a.T @ b
    return np.ascontiguousarray(x.T).T


def _neg(x):
    return np.ascontiguousarray(x[::-1, ::-1])[::-1, ::-1]


def _rows(x):
    big = np.zeros((2 * x.shape[0], x.shape[1]), x.dtype)
    big[::2] = x
    return big[::2]


def _cols(x):
    big = np.zeros((x.shape[0], 3 * x.shape[1]), x.dtype)
    big[:, ::3] = x
    return big[:, ::3]


def _odd1(x, step):
    strides = [step if x.shape[i] == 1 else x.strides[i] for i in (0, 1)]
    return np.lib.stride_tricks.as_strided(x.copy(), x.shape, strides)


LAYOUTS = {
    "C": _c,
    "F": _f,
    "T": _t,
    "neg": _neg,
    "rows": _rows,
    "cols": _cols,
    # a length-1 dim with an odd whole-element stride, and with 3 bytes
    "odd1": lambda x: _odd1(x, 1001 * x.itemsize),
    "odd1-bytes": lambda x: _odd1(x, 3),
}
PAIRS = (
    [(name, "C") for name in LAYOUTS]
    + [("C", name) for name in LAYOUTS if name != "C"]
    + [("T", "T"), ("F", "neg"), ("neg", "T"), ("rows", "cols"), ("odd1-bytes", "odd1-bytes")]
)


def _shapes(dtype):
    kc = _kc(dtype)
    return {
        "small": (7, 13, 5),
        "small-edge": (96, 96, 96),
        "packed": (97, 50, 33),
        "k-past-kc": (130, 2 * kc + 3, 70),
        "odd": (250, 333, 257),
        "row-vector": (1, 300, 200),
        "col-vector": (200, 300, 1),
        "outer": (150, 1, 120),
        "single": (1, 1, 1),
    }


SHAPE_NAMES = list(_shapes(np.float64))


def _operands(dtype, m, k, n, seed=0):
    rng = np.random.default_rng([m, k, n, seed])
    a = rng.standard_normal((m, k)).astype(dtype)
    b = rng.standard_normal((k, n)).astype(dtype)
    return a, b


@pytest.fixture(autouse=True)
def _cpu_backend():
    # A CUDA build may auto-route a float32 product to the device; these tests
    # are about the CPU kernels.
    prev = pg._backend
    pg.set_backend("cpu")
    yield
    pg.set_backend(prev)


@pytest.fixture(autouse=True)
def _restore_thread_setting():
    previous = pg.get_num_threads()
    yield
    pg.set_num_threads(previous)


@pytest.fixture(params=["native", "python"])
def entry(request, monkeypatch):
    if request.param == "python":
        monkeypatch.setattr(pg, "_fast_matmul", pg._no_fast_entry)
    return request.param


@pytest.mark.parametrize("shape", SHAPE_NAMES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_every_layout_gives_the_c_order_bits(dtype, shape, entry):
    m, k, n = _shapes(dtype)[shape]
    a, b = _operands(dtype, m, k, n)
    ref = pg.matmul(a, b)
    differ = []
    for la, lb in PAIRS:
        va, vb = LAYOUTS[la](a), LAYOUTS[lb](b)
        assert np.array_equal(va, a) and np.array_equal(vb, b)
        got = pg.matmul(va, vb)
        if got.dtype != ref.dtype or got.tobytes() != ref.tobytes():
            differ.append((la, lb))
    assert differ == []


def test_native_entry_takes_f_order_and_transposed_views():
    a, b = _operands(np.float64, 40, 30, 20)
    for la, lb in [("F", "C"), ("C", "F"), ("T", "T"), ("odd1", "C"), ("odd1-bytes", "C")]:
        va, vb = LAYOUTS[la](a), LAYOUTS[lb](b)
        assert pg._core._fast_matmul(va, vb, None) is not NotImplemented, (la, lb)
    for la, lb in [("neg", "C"), ("C", "rows"), ("cols", "C")]:
        va, vb = LAYOUTS[la](a), LAYOUTS[lb](b)
        assert pg._core._fast_matmul(va, vb, None) is NotImplemented, (la, lb)


@pytest.mark.parametrize("layout", [name for name in LAYOUTS if name != "odd1-bytes"])
def test_python_path_passes_kernel_readable_operands_through(layout):
    # the odd1 layouts differ from C order only on a length-1 dim
    a, _ = _operands(np.float64, 1 if layout.startswith("odd1") else 40, 30, 20)
    v = LAYOUTS[layout](a)
    assert pg._matmul_operand(v, v.dtype) is v


@pytest.mark.parametrize("dtype", DTYPES)
def test_python_path_copies_only_what_the_kernels_cannot_read(dtype):
    a, _ = _operands(dtype, 6, 4, 1)
    item = np.dtype(dtype).itemsize
    raw = np.frombuffer(bytearray(a.nbytes + 1), np.uint8)
    unaligned = raw[1:].view(dtype).reshape(a.shape)
    unaligned[...] = a
    one_col = _operands(dtype, 6, 1, 1)[0]
    cases = [
        unaligned,
        a.astype(np.dtype(dtype).newbyteorder(">")),
        a.astype(np.float16),
        LAYOUTS["odd1-bytes"](one_col),
    ]
    assert not cases[0].flags.aligned
    assert cases[3].flags.c_contiguous and cases[3].flags.aligned and cases[3].strides[1] % item
    for x in cases:
        got = pg._matmul_operand(x, np.dtype(dtype))
        assert got is not x
        assert got.dtype == np.dtype(dtype) and got.flags.aligned and got.flags.c_contiguous
        np.testing.assert_array_equal(got, x)


def _traced_peak(call):
    tracemalloc.start()
    try:
        call()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("n", [64, 256])
@pytest.mark.parametrize("layout", ["F", "T", "neg", "rows", "cols"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_no_operand_copy(dtype, layout, n, entry):
    # NumPy reports its buffer allocations to tracemalloc. Operands and result
    # are the same size, so a copy of either operand would at least double the
    # traced peak over the result alone.
    a, b = _operands(dtype, n, n, n)
    va, vb = LAYOUTS[layout](a), LAYOUTS[layout](b)
    pg.matmul(va, vb)
    assert _traced_peak(lambda: pg.matmul(va, vb)) < 1.5 * a.nbytes
    # the instrument sees a copy when there is one
    assert _traced_peak(lambda: np.ascontiguousarray(va)) >= a.nbytes


# The team partition: square sizes whose MR and NR panel counts split unevenly,
# and a long-k product with a small output that pays a barrier per KC panel.
THREAD_SHAPES = [(256, 256, 256), (333, 333, 333), (64, 4096, 64), (1000, 1000, 1000)]
THREAD_PAIRS = [("C", "C"), ("T", "C"), ("C", "T"), ("F", "F"), ("neg", "rows"), ("cols", "T")]
THREAD_CASES = [(shape, la, lb) for shape in THREAD_SHAPES for la, lb in THREAD_PAIRS]


def _ids(case):
    (m, k, n), la, lb = case
    return f"{m}x{k}x{n}-{la}-{lb}"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", THREAD_CASES, ids=_ids)
def test_one_and_six_threads_bitwise(dtype, case):
    (m, k, n), la, lb = case
    a, b = _operands(dtype, m, k, n)
    va, vb = LAYOUTS[la](a), LAYOUTS[lb](b)
    pg.set_num_threads(1)
    one = pg.matmul(va, vb)
    pg.set_num_threads(6)
    six = pg.matmul(va, vb)
    c_order = pg.matmul(a, b)
    assert one.tobytes() == six.tobytes() == c_order.tobytes()


# The small kernel's two-thread team (from 2^16 vector FMAs, m * k * n / W)
# splits C into runs of six-row tiles, or into runs of B panels (8 float64 or 16
# float32 columns) when an operand is column-major and the panels split evenly.
# The shapes straddle the threshold (float64 63/64 rows, float32 80/81 cubes)
# and take both splits: odd panel counts (float64 n = 72 and 81, float32 65 and
# 80) stay on rows, even ones split panels, some with a partial edge panel
# (float64 93, float32 81 and 93), and row counts sit off the tile grid.
SMALL_TEAM_SHAPES = {
    np.float64: [(63, 64, 64), (64, 64, 64), (65, 64, 64), (64, 64, 72), (64, 64, 81), (95, 96, 93), (96, 96, 96)],
    np.float32: [(80, 80, 80), (81, 81, 81), (90, 96, 65), (96, 96, 80), (95, 96, 93), (96, 96, 96)],
}
SMALL_TEAM_VIEWS = {
    **LAYOUTS,
    # column-major with its columns walked backwards: a negative column stride
    "F-rev": lambda x: np.asfortranarray(x[:, ::-1])[:, ::-1],
}
SMALL_TEAM_PAIRS = [("C", "C"), ("C", "F"), ("F", "F"), ("T", "F"), ("C", "F-rev"), ("T", "C"), ("F-rev", "C"),
                    ("C", "T")]


def _vector_fmas(m, k, n, dtype):
    return m * k * n // (4 if dtype == np.float64 else 8)


def test_small_team_shapes_straddle_the_team_threshold():
    assert _vector_fmas(63, 64, 64, np.float64) < 2**16 <= _vector_fmas(64, 64, 64, np.float64)
    assert _vector_fmas(80, 80, 80, np.float32) < 2**16 <= _vector_fmas(81, 81, 81, np.float32)
    for dtype, shapes in SMALL_TEAM_SHAPES.items():
        assert all(max(s) <= 96 for s in shapes)  # dispatch takes the small path
        assert sum(_vector_fmas(*s, dtype) >= 2**16 for s in shapes) == len(shapes) - 1


def _assert_views_give_the_c_order_bits(dtype, shapes, pairs, threads):
    for m, k, n in shapes:
        a, b = _operands(dtype, m, k, n)
        pg.set_num_threads(1)
        ref = pg.matmul(a, b)
        for la, lb in pairs:
            va, vb = SMALL_TEAM_VIEWS[la](a), SMALL_TEAM_VIEWS[lb](b)
            assert np.array_equal(va, a) and np.array_equal(vb, b)
            for t in threads:
                pg.set_num_threads(t)
                # NaN-filled, so an element no member wrote cannot pass
                out = np.full((m, n), np.nan, dtype)
                pg.matmul(va, vb, out=out)
                assert out.tobytes() == ref.tobytes(), (m, k, n, la, lb, t)


@pytest.mark.parametrize("dtype", DTYPES)
def test_small_team_splits_give_the_c_order_bits(dtype):
    assert SMALL_TEAM_VIEWS["F-rev"](np.zeros((3, 2), dtype)).strides[1] < 0
    _assert_views_give_the_c_order_bits(dtype, SMALL_TEAM_SHAPES[dtype], SMALL_TEAM_PAIRS, (2, 6, 1))


# A column-major A whose column pitch reaches few L1 sets (float64 32, 64, and
# 96 rows, float32 64) reads in place over packed B panels with its next
# columns prefetched, from one k step past the line-count bound (2 * k against
# 8 ways of the reached sets); one step short keeps the in-place B rows. Edge
# panels, a one-vector float32 panel (n = 7, 8), and row counts off the tile
# grid ride along.
CROWDED_A_SHAPES = {
    np.float64: [(64, 32, 24), (64, 33, 24), (96, 64, 20), (96, 65, 20), (32, 64, 40), (32, 65, 40),
                 (64, 64, 64), (64, 96, 93), (96, 96, 96)],
    np.float32: [(64, 64, 40), (64, 65, 40), (64, 96, 8), (64, 96, 7), (64, 96, 64)],
}
CROWDED_A_PAIRS = [("T", "C"), ("F", "F"), ("F-rev", "C"), ("T", "F-rev"), ("F", "T")]


@pytest.mark.parametrize("dtype", DTYPES)
def test_crowded_column_major_a_gives_the_c_order_bits(dtype):
    _assert_views_give_the_c_order_bits(dtype, CROWDED_A_SHAPES[dtype], CROWDED_A_PAIRS, (1, 6))


def _digest(r):
    return r.dtype.str, r.shape, hashlib.blake2b(r.tobytes()).hexdigest()


def test_concurrent_callers_bitwise():
    # Each submitting thread roots its own OpenMP team and owns its own packing
    # buffers; a shared buffer would show up as a wrong digest here.
    tasks = {}
    for dtype in DTYPES:
        for case in THREAD_CASES:
            (m, k, n), la, lb = case
            if m >= 1000:  # the smaller shapes already split unevenly; this one only adds time
                continue
            a, b = _operands(dtype, m, k, n, seed=1)
            va, vb = LAYOUTS[la](a), LAYOUTS[lb](b)
            tasks[(np.dtype(dtype).name, _ids(case))] = lambda va=va, vb=vb: _digest(pg.matmul(va, vb))
    pg.set_num_threads(1)
    refs = {name: fn() for name, fn in tasks.items()}
    pg.set_num_threads(6)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [(name, pool.submit(fn)) for _ in range(2) for name, fn in tasks.items()]
        differ = [name for name, f in futures if f.result(timeout=120) != refs[name]]
    assert differ == []
