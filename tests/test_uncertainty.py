# -*- coding: utf-8 -*-
from __future__ import annotations

import operator
import warnings
from typing import Type

import hypothesis.strategies as st
import numpy as np
import pint
import pytest
from hypothesis import given, settings
from hypothesis.extra import numpy as hnp

from auto_uncertainties import NegativeStdDevError, Uncertainty

BINARY_OPS = [
    operator.lt,
    operator.le,
    operator.eq,
    operator.ne,
    operator.gt,
    operator.ge,
]
UNARY_OPS = [operator.not_, operator.abs]
unit_registry = pint.UnitRegistry()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    unit_registry.Quantity([])
warnings.filterwarnings("ignore", category=pint.UnitStrippedWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
unit_registry.enable_contexts("boltzmann")
unit_registry.default_format = "0.8g~P"
unit_registry.default_system = "cgs"
UNITS = [None, unit_registry(""), unit_registry("m"), unit_registry("s")]


def check_units_and_mag(unc, units, mag, err):
    if units is not None and not hasattr(unc, "units"):
        raise ValueError
    if units is None:
        assert unc.value == mag
        assert unc.error == err
    else:
        assert unc.units.is_compatible_with(units)
        assert unc.value.units.is_compatible_with(units)
        assert unc.value.to(units).m == mag
        assert unc.error.units.is_compatible_with(units)
        assert unc.error.to(units).m == err


@given(
    v=st.floats(allow_subnormal=False),
    e=st.floats(allow_subnormal=False),
    units=st.sampled_from(UNITS),
)
def test_scalar_creation(v, e, units):
    if e < 0:
        if np.isfinite(e):
            with pytest.raises(NegativeStdDevError):
                u = Uncertainty(v, e)
            if units is not None:
                with pytest.raises(NegativeStdDevError):
                    u = Uncertainty(v * units, e * units)
    else:
        u = Uncertainty(v, e)
        if np.isfinite(v) and np.isfinite(e):
            assert u.nominal_value == v
            assert u.error == e
            if v > 0:
                assert u.relative == e / v
            elif v == 0:
                assert not np.isfinite(u.relative)

            if units is not None:
                u = Uncertainty(v * units, e * units)
                check_units_and_mag(u, units, v, e)

                u = Uncertainty(v, e) * units
                check_units_and_mag(u, units, v, e)

                with pytest.raises(pint.DimensionalityError):
                    u = Uncertainty(v * units, e)
                with pytest.raises(pint.DimensionalityError):
                    u = Uncertainty(v, e * units)


@given(
    v1=st.floats(allow_subnormal=False),
    e1=st.floats(min_value=0, max_value=1e3, allow_subnormal=False),
    op=st.sampled_from(UNARY_OPS),
)
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
    v1=st.floats(allow_subnormal=False),
    v2=st.floats(allow_subnormal=False),
    e1=st.floats(min_value=0, max_value=1e3, allow_subnormal=False),
    e2=st.floats(min_value=0, max_value=1e3, allow_subnormal=False),
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
    v1=st.floats(allow_subnormal=False),
    v2=st.floats(allow_subnormal=False),
    e1=st.floats(min_value=0, max_value=1e3, allow_subnormal=False),
    e2=st.floats(min_value=0, max_value=1e3, allow_subnormal=False),
    op=st.sampled_from([operator.add, operator.sub]),
)
def test_scalar_add_sub(v1, e1, v2, e2, op):
    u1 = Uncertainty(v1, e1)
    u2 = Uncertainty(v2, e2)

    u = op(u1, u2)
    if np.isfinite(v1) and np.isfinite(v2):
        assert u.value == op(u1.value, u2.value)
    if np.isfinite(e1) and np.isfinite(e2):
        assert u.error == np.sqrt(u1.error**2 + u2.error**2)


@given(
    v1=st.floats(min_value=1, max_value=1e3, allow_subnormal=False),
    v2=st.floats(min_value=1, max_value=1e3, allow_subnormal=False),
    e1=st.floats(min_value=0, max_value=1e3, allow_subnormal=False),
    e2=st.floats(min_value=0, max_value=1e3, allow_subnormal=False),
    op=st.sampled_from([operator.mul, operator.truediv]),
)
def test_scalar_mul_div(v1, e1, v2, e2, op):
    u1 = Uncertainty(v1, e1)
    u2 = Uncertainty(v2, e2)

    u = op(u1, u2)
    if np.isfinite(v1) and np.isfinite(v2):
        assert u.value == op(u1.value, u2.value)
    if np.isfinite(e1) and np.isfinite(e2):
        np.testing.assert_almost_equal(
            u.error, u.value * np.sqrt(u1.rel**2 + u2.rel**2)
        )


@given(
    v1=st.floats(min_value=0, max_value=1e3, allow_subnormal=False),
    v2=hnp.arrays(
        dtype=st.sampled_from([np.float64]),
        shape=(11,),
        elements=st.floats(
            min_value=-10.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    ),
    e1=st.floats(min_value=0, max_value=1e3, allow_subnormal=False),
    e2=hnp.arrays(
        dtype=st.sampled_from([np.float64]),
        shape=(11,),
        elements=st.floats(
            min_value=0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
    ),
    op=st.sampled_from(
        [
            operator.add,
            operator.sub,
            operator.mul,
            operator.truediv,
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

    u = op(u2, u1)
    np.testing.assert_almost_equal(u.value, op(u2.value, u1.value))
