import numpy as np
import pytest

import fastmathext as fme


def test_inner_dim_mismatch_is_value_error():
    a = np.zeros((3, 4))
    b = np.zeros((5, 2))
    with pytest.raises(ValueError, match="inner dimensions"):
        fme.matmul(a, b)


def test_1d_input_rejected():
    with pytest.raises(ValueError, match="2-dimensional"):
        fme.matmul(np.zeros(3), np.zeros((3, 2)))


def test_module_reports_capabilities():
    feats = fme.cpu_features()
    assert set(feats) == {"avx2", "fma", "avx512f"}
    assert isinstance(fme.has_cuda_build(), bool)


@pytest.mark.parametrize("name", ["bool", "float16", "complex64", "complex128", "object"])
def test_rejected_dtype_is_type_error(name):
    a = np.zeros((2, 2), dtype=name)
    b = np.zeros((2, 2))
    # the match includes the dtype name: the message must name its dtype
    with pytest.raises(TypeError, match=f"unsupported dtype {name}"):
        fme.matmul(a, b)


def test_rejected_dtype_on_b_operand_is_type_error():
    # rejection is per operand, not just on a
    a = np.zeros((2, 2))
    b = np.zeros((2, 2), dtype=np.float16)
    with pytest.raises(TypeError, match="unsupported dtype float16"):
        fme.matmul(a, b)


def test_out_keyword_raises_not_implemented():
    a = np.zeros((2, 2))
    b = np.zeros((2, 2))
    with pytest.raises(NotImplementedError, match="out= is not implemented"):
        fme.matmul(a, b, out=np.zeros((2, 2)))


def test_out_none_matches_omission():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 3))
    b = rng.standard_normal((3, 5))
    # same inputs, deterministic kernel: bitwise equality is the correct bar
    np.testing.assert_array_equal(fme.matmul(a, b, out=None), fme.matmul(a, b))


def test_out_positional_rejected():
    # out is keyword-only; the interpreter rejects a third positional
    # argument at the signature, so no message match
    a = np.zeros((2, 2))
    b = np.zeros((2, 2))
    with pytest.raises(TypeError):
        fme.matmul(a, b, np.zeros((2, 2)))
