from __future__ import annotations

import math
import operator
import warnings

from hypothesis import given
from hypothesis.extra import numpy as hnp
import hypothesis.strategies as st
import numpy as np
from pint import (
    DimensionalityError,
    Quantity,
)
import pytest

# requires installation with CI dependencies
from auto_uncertainties import NegativeStdDevError
from auto_uncertainties.numpy import HANDLED_FUNCTIONS, HANDLED_UFUNCS
from auto_uncertainties.uncertainty.uncertainty_containers import (
    ScalarUncertainty,
    Uncertainty,
    VectorUncertainty,
    _check_units,
    nominal_values,
    std_devs,
)

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
        assert math.isclose(u, op(u1.value, u2.value), rel_tol=1e-09)


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


# -----------------------------------------------------------------
# --------------------------  NEW TESTS  --------------------------
# -----------------------------------------------------------------


def test_check_units():
    assert _check_units


def test_nominal_values():
    x = Uncertainty(2, 3)
    assert nominal_values(x) == x.value

    x = np.array([1, 2, 3])
    assert nominal_values(x).all() == Uncertainty.from_sequence(x).value.all()

    x = "not an Uncertainty"
    assert nominal_values(x) == x

    x = np.nan
    assert np.isnan(nominal_values(x))

    # TODO: could be improved


def test_std_devs():
    x = Uncertainty(2, 3)
    assert std_devs(x) == x.error

    x = np.array([1, 2, 3])
    assert std_devs(x).all() == Uncertainty.from_sequence(x).error.all()

    x = "not an Uncertainty"
    assert std_devs(x) == 0

    x = np.nan
    assert std_devs(x) == 0

    # TODO: could be improved


class TestUncertainty:
    """Tests that expand the coverage of the previous tests."""

    @staticmethod
    def test_getstate():
        u = Uncertainty(2, 3)
        assert u.__getstate__() == {"nominal_value": u._nom, "std_devs": u._err}

    @staticmethod
    def test_setstate():
        u = Uncertainty(2, 3)
        u.__setstate__({"nominal_value": 100, "std_devs": 200})
        assert u.__getstate__()["nominal_value"] == 100
        assert u.__getstate__()["std_devs"] == 200

    @staticmethod
    def test_getnewargs():
        u = Uncertainty(2, 3)
        assert u.__getnewargs__()[0] == 2
        assert u.__getnewargs__()[1] == 3

    @staticmethod
    def test_init():
        scalar = Uncertainty(2, 3)
        vector = Uncertainty(np.array([1, 2, 3]), np.array([4, 5, 6]))
        assert isinstance(scalar, ScalarUncertainty)  # verify scalar type was chosen
        assert isinstance(vector, VectorUncertainty)  # verify vector type was chosen

        # Verify inheritance
        assert isinstance(scalar, Uncertainty)
        assert isinstance(vector, Uncertainty)

    @staticmethod
    def test_properties():
        u = Uncertainty(2, 3)
        assert isinstance(u, ScalarUncertainty)

        assert u.value == u._nom
        assert u.error == u._err
        assert u.relative == 1.5
        assert u.rel == u.relative
        assert u.rel2 == 2.25

    @staticmethod
    @given(
        st.floats(0, 10),
        st.floats(0, 10),
        st.floats(0, 10),
    )
    def test_plus_minus(val, err, pm):
        u = Uncertainty(val, err)
        u = u.plus_minus(pm)

        assert u.value == val
        assert u.error == np.sqrt(err**2 + pm**2)

    @staticmethod
    @pytest.mark.parametrize(
        "val, expected",
        [
            ("2.0 +/- 3.5", Uncertainty(2.0, 3.5)),
            ("2.0 +- 3.5", Uncertainty(2.0, 3.5)),
            ("2.6", Uncertainty(2.6)),
        ],
    )
    def test_from_string(val, expected):
        assert Uncertainty.from_string(val) == expected

        # TODO: uncertainty_containers:154  ->  should err param be default to 0.0? Seems to break wrappers if so

    @staticmethod
    @pytest.mark.parametrize(
        "val, err", [(Quantity(2, "radian"), Quantity(3, "radian")), (2, 3), (8.5, 9.5)]
    )
    def test_from_quantities(val, err):
        if isinstance(val, Quantity) and isinstance(err, Quantity):
            with pytest.raises(NotImplementedError):
                _ = Uncertainty.from_quantities(val, err)
            return

        assert Uncertainty.from_quantities(val, err).value == val
        assert Uncertainty.from_quantities(val, err).error == err

        # TODO: uncertainty_containers:290 & 63-69  -->  should this be tested if it doesn't work?

    @staticmethod
    def test_from_sequence():
        seq = [Uncertainty(2, 3), Uncertainty(3, 4), Uncertainty(4, 5)]
        result = Uncertainty.from_sequence(seq)
        assert isinstance(result, VectorUncertainty)
        assert result[0] == Uncertainty(2, 3)
        assert result[1] == Uncertainty(3, 4)
        assert result[2] == Uncertainty(4, 5)

        seq = [None, Uncertainty(2, 3)]
        with pytest.raises(TypeError):
            _ = Uncertainty.from_sequence(seq)

        # TODO: uncertainty_containers:327 & 63-69  -->  should this be tested if it doesn't work?

        # seq = [Uncertainty(2, 3) * Unit('radian'), Uncertainty(5, 8)]
        # result = Uncertainty.from_sequence(seq)
        # assert result.units == Unit('radian')

    # TODO: ------------------------------------------
    # TODO: Add tests for operator dunder methods here
    # TODO: ------------------------------------------


class TestVectorUncertainty:
    """Tests that expand the coverage of the previous VectorUncertainty tests."""

    @staticmethod
    def test_getattr():
        v = VectorUncertainty(np.array([1, 2, 3]), np.array([4, 5, 6]))

        with pytest.raises(AttributeError):
            _ = v.__array_something

        for item1 in v.__apply_to_both_ndarray__:
            try:
                assert callable(getattr(v, item1)) or isinstance(
                    getattr(v, item1), VectorUncertainty
                )
            except ValueError:
                continue

        for item2 in HANDLED_UFUNCS:
            assert callable(getattr(v, item2))

        for item3 in HANDLED_FUNCTIONS:
            assert getattr(v, item3) is not None

        for item4 in v.__ndarray_attributes__:
            assert getattr(v, item4) == getattr(v._nom, item4)

        with pytest.raises(AttributeError):
            _ = v.ATTRIBUTE_THAT_DOES_NOT_EXIST

    # WORK IN PROGRESS...
