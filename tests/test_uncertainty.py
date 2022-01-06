# -*- coding: utf-8 -*-
from typing import Type
import pytest
import hypothesis.strategies as st
from hypothesis import given, settings
from hypothesis.extra import numpy as hnp
import numpy as np
import operator

from auto_uncertainties import Uncertainty, NegativeStdDevError

BINARY_OPS = [operator.lt, operator.le, operator.eq, operator.ne, operator.gt, operator.ge]
UNARY_OPS = [operator.not_, operator.abs]


@given(v=st.floats(), e=st.floats())
def test_scalar_creation(v, e):
    if e < 0:
        if np.isfinite(e):
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


@given(v1=st.floats(), e1=st.floats(min_value=0, max_value=1e3), op=st.sampled_from(UNARY_OPS))
def test_scalar_unary(v1, e1, op):
    u1 = Uncertainty(v1, e1)

    u = op(u1)
    if np.isfinite(v1):
        if isinstance(u, Uncertainty):
            assert u.value == op(u1.value)
            if np.isfinite(e1):
                assert np.isfinite(u.error)
        else:
            assert u == op(u1.value)


@given(
    v1=st.floats(),
    v2=st.floats(),
    e1=st.floats(min_value=0, max_value=1e3),
    e2=st.floats(min_value=0, max_value=1e3),
    op=st.sampled_from(BINARY_OPS),
)
def test_scalar_binary(v1, e1, v2, e2, op):
    u1 = Uncertainty(v1, e1)
    u2 = Uncertainty(v2, e2)

    u = op(u1, u2)
    if isinstance(u, Uncertainty):
        assert u.value == op(u1.value, u2.value)
        if np.isfinite(e1) and np.isfinite(e2):
            assert np.isfinite(u.error)
    else:
        assert u == op(u1.value, u2.value)


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


@given(
    v1=st.floats(min_value=0, max_value=1e3),
    v2=hnp.arrays(
        dtype=st.sampled_from([np.float64]),
        shape=(31,),
        elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    ),
    e1=st.floats(min_value=0, max_value=1e3),
    e2=hnp.arrays(
        dtype=st.sampled_from([np.float64]),
        shape=(31,),
        elements=st.floats(min_value=0, max_value=10.0, allow_nan=False, allow_infinity=False),
    ),
    op=st.sampled_from(
        [
            operator.add,
            operator.sub,
            operator.iadd,
            operator.isub,
            operator.mul,
            operator.truediv,
            operator.imul,
            operator.itruediv,
            operator.mod,
            operator.pow,
        ]
    ),
)
def test_mixed_arithmetic(v1, e1, v2, e2, op):
    u1 = Uncertainty(v1, e1)
    u2 = Uncertainty(v2, e2)

    u = op(u1, u2)
    np.testing.assert_almost_equal(u.value, op(u1.value, u2.value))


@given(
    v1=st.floats(min_value=0, max_value=1e3),
    v2=hnp.arrays(
        dtype=st.sampled_from([np.float64]),
        shape=(31,),
        elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    ),
    e1=st.floats(min_value=0, max_value=1e3),
    e2=hnp.arrays(
        dtype=st.sampled_from([np.float64]),
        shape=(31,),
        elements=st.floats(min_value=0, max_value=10.0, allow_nan=False, allow_infinity=False),
    ),
    op=st.sampled_from(
        [
            operator.add,
            operator.sub,
            operator.iadd,
            operator.isub,
            operator.mul,
            operator.truediv,
            operator.imul,
            operator.itruediv,
            operator.mod,
            operator.pow,
        ]
    ),
)
def test_mixed_arithmetic2(v1, e1, v2, e2, op):
    u1 = Uncertainty(v1, e1)
    u2 = Uncertainty(v2, e2)

    u = op(u2, u1)
    np.testing.assert_almost_equal(u.value, op(u2.value, u1.value))
