# -*- coding: utf-8 -*-
import pytest
import hypothesis.strategies as st
from hypothesis import given, settings
import numpy as np
import operator

from auto_uncertainties import Uncertainty, NegativeStdDevError


@given(v=st.floats(), e=st.floats())
def test_scalar_creation(v, e):
    if e < 0:
        with pytest.raises(NegativeStdDevError):
            u = Uncertainty(v, e)
    else:
        u = Uncertainty(v, e)
        if np.isfinite(v) and np.isfinite(e):
            assert u.nominal_value == v
            assert u.error == e
            if v > 0:
                assert u.relative == e / v
            elif v == 0:
                assert np.isnan(u.relative)


@given(
    v1=st.floats(),
    v2=st.floats(),
    e1=st.floats(min_value=0, max_value=1e3),
    e2=st.floats(min_value=0, max_value=1e3),
    op=st.sampled_from([operator.add, operator.sub, operator.iadd, operator.isub]),
)
def test_scalar_add_sub(v1, e1, v2, e2, op):
    u1 = Uncertainty(v1, e1)
    u2 = Uncertainty(v2, e2)

    u = op(u1, u2)
    if np.isfinite(v1) and np.isfinite(v2):
        assert u.value == op(u1.value, u2.value)
    if np.isfinite(e1) and np.isfinite(e2):
        assert u.error == np.sqrt(u1.error ** 2 + u2.error ** 2)


@given(
    v1=st.floats(min_value=1, max_value=1e3),
    v2=st.floats(min_value=1, max_value=1e3),
    e1=st.floats(min_value=0, max_value=1e3),
    e2=st.floats(min_value=0, max_value=1e3),
    op=st.sampled_from([operator.mul, operator.truediv, operator.imul, operator.itruediv]),
)
def test_scalar_mul_div(v1, e1, v2, e2, op):
    u1 = Uncertainty(v1, e1)
    u2 = Uncertainty(v2, e2)

    u = op(u1, u2)
    if np.isfinite(v1) and np.isfinite(v2):
        assert u.value == op(u1.value, u2.value)
    if np.isfinite(e1) and np.isfinite(e2):
        np.testing.assert_almost_equal(u.error, u.value * np.sqrt(u1.rel ** 2 + u2.rel ** 2))
