from __future__ import annotations

import math
import operator
import warnings

from hypothesis import given
from hypothesis.extra import numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest

from auto_uncertainties import (
    NegativeStdDevError,
    ScalarUncertainty,
    Uncertainty,
)

try:
    from pint import DimensionalityError
except ImportError:

    class DimensionalityError(Exception):
        pass


BINARY_OPS = [
    operator.lt,
    operator.le,
    operator.eq,
    operator.ne,
    operator.gt,
    operator.ge,
]
UNARY_OPS = [operator.not_, operator.abs]
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
UNITS = [None]


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


general_float_strategy = dict(
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
    min_value=-1e3,
    max_value=1e3,
)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@given(
    v=st.floats(),
    e=st.floats(),
    units=st.sampled_from(UNITS),
    call_super=st.sampled_from([True, False]),
)
def test_scalar_creation(v, e, units, call_super):
    if call_super:
        const = Uncertainty
    else:
        const = ScalarUncertainty

    if not np.isfinite(v):
        u = const(v, e)
        assert isinstance(u, float)
        assert not np.isfinite(v)
    elif not np.isfinite(e):
        u = const(v, e)
        assert u.error == 0
    elif e < 0:
        with pytest.raises(NegativeStdDevError):
            u = const(v, e)
        if units is not None:
            with pytest.raises(NegativeStdDevError):
                u = const.from_quantities(v * units, e * units)
    else:
        u = const(v, e)
        if np.isfinite(v) and np.isfinite(e):
            assert math.isclose(u.value, v)
            assert math.isclose(u.error, e)
            if v > 0:
                assert math.isclose(u.relative, e / v)
            elif v == 0:
                assert not np.isfinite(u.relative)

            if units is not None:
                with pytest.raises(NotImplementedError):
                    u = const(v * units, e * units)

                u = const.from_quantities(v * units, e * units)
                check_units_and_mag(u, units, v, e)

                u = const(v, e) * units
                check_units_and_mag(u, units, v, e)

                with pytest.raises(DimensionalityError):
                    u = const.from_quantities(v * units, e)
                with pytest.raises(DimensionalityError):
                    u = const.from_quantities(v, e * units)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@given(
    v1=st.floats(**general_float_strategy),
    e1=st.floats(min_value=0, max_value=1e3),
    op=st.sampled_from(UNARY_OPS),
)
def test_scalar_unary(v1, e1, op):
    u1 = Uncertainty(v1, e1)

    u = op(u1)
    if np.isfinite(v1):
        if isinstance(u, Uncertainty):
            assert math.isclose(u.value, op(u1.value))
            if np.isfinite(e1):
                assert np.isfinite(u.error)
        else:
            assert u == op(u1.value)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@given(
    v1=st.floats(**general_float_strategy),
    v2=st.floats(**general_float_strategy),
    e1=st.floats(
        min_value=0,
        max_value=1e3,
    ),
    e2=st.floats(
        min_value=0,
        max_value=1e3,
    ),
    op=st.sampled_from(BINARY_OPS),
)
def test_scalar_binary(v1, e1, v2, e2, op):
    u1 = Uncertainty(v1, e1)
    u2 = Uncertainty(v2, e2)

    u = op(u1, u2)
    if isinstance(u, Uncertainty):
        assert math.isclose(u.value, op(u1.value, u2.value))
        if np.isfinite(e1) and np.isfinite(e2):
            assert np.isfinite(u.error)
    else:
        assert math.isclose(u, op(u1.value, u2.value))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@given(
    v1=st.floats(**general_float_strategy),
    v2=st.floats(**general_float_strategy),
    e1=st.floats(
        min_value=0,
        max_value=1e3,
    ),
    e2=st.floats(
        min_value=0,
        max_value=1e3,
    ),
    op=st.sampled_from([operator.add, operator.sub]),
)
def test_scalar_add_sub(v1, e1, v2, e2, op):
    u1 = Uncertainty(v1, e1)
    u2 = Uncertainty(v2, e2)

    u = op(u1, u2)
    if np.isfinite(v1) and np.isfinite(v2):
        assert math.isclose(u.value, op(u1.value, u2.value))
    if np.isfinite(e1) and np.isfinite(e2):
        assert math.isclose(u.error, np.sqrt(u1.error**2 + u2.error**2))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@given(
    v1=st.floats(
        min_value=1,
        max_value=1e3,
    ),
    v2=st.floats(
        min_value=1,
        max_value=1e3,
    ),
    e1=st.floats(
        min_value=0,
        max_value=1e3,
    ),
    e2=st.floats(
        min_value=0,
        max_value=1e3,
    ),
    op=st.sampled_from([operator.mul, operator.truediv]),
)
def test_scalar_mul_div(v1, e1, v2, e2, op):
    u1 = Uncertainty(v1, e1)
    u2 = Uncertainty(v2, e2)

    u = op(u1, u2)
    if np.isfinite(v1) and np.isfinite(v2):
        assert math.isclose(u.value, op(u1.value, u2.value))
    if np.isfinite(e1) and np.isfinite(e2):
        np.testing.assert_almost_equal(
            u.error, u.value * np.sqrt(u1.rel**2 + u2.rel**2)
        )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@given(
    v1=st.floats(
        min_value=0,
        max_value=1e3,
    ),
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
    e1=st.floats(
        min_value=0,
        max_value=1e3,
    ),
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
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_mixed_arithmetic(v1, e1, v2, e2, op):
    u1 = Uncertainty(v1, e1)
    u2 = Uncertainty(v2, e2)

    u = op(u1, u2)
    np.testing.assert_almost_equal(u.value, op(u1.value, u2.value))

    u = op(u2, u1)
    np.testing.assert_almost_equal(u.value, op(u2.value, u1.value))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@given(
    v1=st.floats(
        min_value=0,
        max_value=1e3,
    ),
    e1=st.floats(
        min_value=0,
        max_value=1e3,
    ),
    op=st.sampled_from(
        [
            np.exp,
            np.abs,
            np.log,
        ]
    ),
)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_numpy_math_ops(v1, e1, op):
    u1 = Uncertainty(v1, e1)

    u = op(u1)
    if np.isfinite(v1) and np.isfinite(u):
        math.isclose(u.value, op(u1.value))
