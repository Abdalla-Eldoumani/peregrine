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
