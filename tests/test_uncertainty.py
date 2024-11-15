from __future__ import annotations

import copy
import locale
import math
import operator
import warnings

from hypothesis import assume, given
from hypothesis.extra import numpy as hnp
import hypothesis.strategies as st
import joblib
import numpy as np
from pint import (
    DimensionalityError,
    Quantity,
    Unit,
)
import pytest

# requires installation with CI dependencies
from auto_uncertainties import (
    DowncastError,
    DowncastWarning,
    NegativeStdDevError,
    set_downcast_error,
)
from auto_uncertainties.numpy import HANDLED_FUNCTIONS, HANDLED_UFUNCS
from auto_uncertainties.pint import UncertaintyQuantity
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

    # Compare Uncertainty with Uncertainty, and with float
    results = [op(u1, u2), op(u1, v2)]

    for result in results:
        if isinstance(result, Uncertainty):
            assert math.isclose(result.value, op(u1.value, u2.value), rel_tol=1e-9)
            if np.isfinite(e1) and np.isfinite(e2):
                assert np.isfinite(result.error)
        else:
            assert math.isclose(result, op(u1.value, u2.value), rel_tol=1e-9)


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
    val, err = Quantity(2, "radian"), Quantity(3, "radian")
    assert _check_units(val, err) == (2, 3, Unit("radian"))

    val, err = 2, Quantity(3, "radian")
    assert _check_units(val, err) == (2, 3, Unit("radian"))

    val, err = Quantity(2, "radian"), 3
    assert _check_units(val, err) == (2, 3, Unit("radian"))

    val, err = 2, 3
    assert _check_units(val, err) == (2, 3, None)

    with pytest.raises(DimensionalityError):
        val, err = Quantity(2, "m"), Quantity(3, "deg")
        _ = _check_units(val, err)


def test_nominal_values():
    x = Uncertainty(2, 3)
    assert nominal_values(x) == x.value

    # Check creating from sequence
    x = np.array([1, 2, 3])
    assert np.array_equal(nominal_values(x), Uncertainty.from_sequence(x).value)

    # Check when unable to create from sequence
    x = [None, Uncertainty(2, 3)]
    assert nominal_values(x) == x

    # Check wrong type
    x = "not an Uncertainty"
    assert nominal_values(x) == x

    # Check NaN / when float is returned instead of Uncertainty
    assert np.isnan(nominal_values(np.nan))

    # Check non-vector
    assert nominal_values(5) == 5


def test_std_devs():
    x = Uncertainty(2, 3)
    assert std_devs(x) == x.error

    # Check creating from sequence
    x = np.array([1, 2, 3])
    assert np.array_equal(std_devs(x), Uncertainty.from_sequence(x).error)

    # Check when unable to create from sequence
    x = [None, Uncertainty(2, 3)]
    assert np.array_equal(std_devs(x), np.array([0, 0]))

    # Check wrong type
    x = "not an Uncertainty"
    assert std_devs(x) == 0

    # Check NaN / when float is returned instead of Uncertainty
    assert std_devs(np.nan) == 0

    # Check non-vector
    assert std_devs(5) == 0


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

        # Check creating from other Uncertainty objects
        from_subclass = Uncertainty(vector)
        assert isinstance(from_subclass, Uncertainty)
        assert np.array_equal(from_subclass.value, vector.value)
        assert np.array_equal(from_subclass.error, vector.error)

        # Check creating from a Sequence
        from_list = Uncertainty(
            [Uncertainty(1, 2), Uncertainty(4, 5), Uncertainty(1.5, 9.25)]
        )
        assert isinstance(from_list, Uncertainty)

        # Check vector value, constant error edge case
        v = Uncertainty(np.array([1, 2, 3]), 0)
        assert isinstance(v, Uncertainty)

        # Check proper handling of Quantity inputs
        # (further tests for from_quantities are handled in a separate function)
        from_quant = Uncertainty(Quantity(2, "radian"), Quantity(1, "radian"))
        assert isinstance(from_quant, UncertaintyQuantity)
        assert isinstance(from_quant.m, Uncertainty)
        assert from_quant.units == Unit("radian")

        # Check error is raised when a negative value is found in the err array
        with pytest.raises(NegativeStdDevError):
            _ = Uncertainty(np.array([1, 2, 3]), np.array([-1, 2, 3]))

    @staticmethod
    def test_copy():
        u = Uncertainty(2, 3)
        dup = copy.copy(u)
        assert dup.value == u.value
        assert dup.error == u.error
        assert id(u) != id(dup)

    @staticmethod
    def test_deepcopy():
        u = Uncertainty(2, 3)
        dup = copy.deepcopy(u)
        assert id(u) != id(dup)

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

    @staticmethod
    @pytest.mark.parametrize(
        "val, err",
        [
            (Quantity(2, "radian"), Quantity(3, "radian")),
            (Quantity(2, "radian"), 3),
            (2, Quantity(3, "radian")),
            (2, 3),
            (8.5, 9.5),
        ],
    )
    def test_from_quantities(val, err):
        assert Uncertainty.from_quantities(val, err).value == val
        assert Uncertainty.from_quantities(val, err).error == err

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

        # Test with Quantity objects
        seq = [Uncertainty.from_quantities(Quantity(2, "radian"), 1), Uncertainty(5, 8)]
        result = Uncertainty.from_sequence(seq)
        assert result.units == Unit("radian")

    @staticmethod
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
    def test_addsub(v1, v2, e1, e2, op):
        u1 = Uncertainty(v1, e1)
        u2 = Uncertainty(v2, e2)

        result = op(u1, u2)
        assert isinstance(result, Uncertainty)
        assert result.value == op(u1.value, u2.value)
        assert result.error == np.sqrt(u1.error**2 + u2.error**2)

        result = op(u1, v2)
        assert isinstance(result, Uncertainty)
        assert result.value == op(u1.value, v2)
        assert result.error == u1.error

        # Reverse case
        result = op(v2, u1)
        assert isinstance(result, Uncertainty)
        assert result.value == op(v2, u1.value)
        assert result.error == u1.error

        u1 = Uncertainty(np.array([v1, v2]), np.array([e1, e2]))
        result = op(u1, np.array([1, 2]))
        assert isinstance(result, Uncertainty)

    @staticmethod
    @given(
        v1=st.floats(**general_float_strategy),
        v2=st.floats(**general_float_strategy),
        e1=st.floats(
            min_value=1,
            max_value=1e3,
        ),
        e2=st.floats(
            min_value=1,
            max_value=1e3,
        ),
        op=st.sampled_from([operator.mul, operator.truediv]),
    )
    def test_muldiv(v1, v2, e1, e2, op):
        assume(not (op == operator.truediv and (v2 == 0 or e2 == 0)))

        u1 = Uncertainty(v1, e1)
        u2 = Uncertainty(v2, e2)

        assume(not np.isnan(u1.rel2 + u2.rel2))
        assume(not np.isinf(u1.rel2 + u2.rel2))

        result = op(u1, u2)
        assert isinstance(result, Uncertainty)
        assert result.value == op(u1.value, u2.value)
        assert result.error == np.abs(result.value) * np.sqrt(u1.rel2 + u2.rel2)

        result = op(u1, v2)
        assert isinstance(result, Uncertainty)
        assert result.value == op(u1.value, v2)
        assert result.error == np.abs(op(u1.error, v2))

        # Reverse case
        result = op(v2, u1)
        assert isinstance(result, Uncertainty)
        assert result.value == op(v2, u1.value)
        assert (
            result.error == np.abs(result.value) * np.abs(u1.rel)
            if op == operator.truediv
            else np.abs(op(u1.error, v2))
        )

        u1 = Uncertainty(np.array([v1, v2]), np.array([e1, e2]))
        result = op(u1, np.array([1, 2]))
        assert isinstance(result, Uncertainty)

    @staticmethod
    @given(
        v1=st.floats(**general_float_strategy),
        v2=st.floats(**general_float_strategy),
        e1=st.floats(
            min_value=1,
            max_value=1e3,
        ),
        e2=st.floats(
            min_value=1,
            max_value=1e3,
        ),
    )
    def test_floordiv(v1, v2, e1, e2):
        assume(v1 != 0 and v2 != 0)

        u1 = Uncertainty(v1, e1)
        u2 = Uncertainty(v2, e2)

        assume(not np.isnan(u1.rel2 + u2.rel2))
        assume(not np.isinf(u1.rel2 + u2.rel2))

        result = u1 // u2
        assert isinstance(result, Uncertainty)
        assert result.value == u1.value // u2.value
        assert result.error == (u1 / u2).error

        result = u1 // v2
        assert isinstance(result, Uncertainty)
        assert result.value == u1.value // v2
        assert result.error == (u1 / v2).error

        # Reverse case
        result = v2 // u1
        assert isinstance(result, Uncertainty)
        assert result.value == v2 // u1.value
        assert result.error == (v2 / u1).error

        u1 = Uncertainty(np.array([v1, v2]), np.array([e1, e2]))
        result = u1 // np.array([1, 2])
        assert isinstance(result, Uncertainty)

    @staticmethod
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
    )
    def test_mod(v1, v2, e1, e2):
        assume(v1 != 0 and e1 != 0)
        assume(v2 != 0 and e2 != 0)

        u1 = Uncertainty(v1, e1)
        u2 = Uncertainty(v2, e2)

        result = u1 % u2
        assert isinstance(result, Uncertainty)
        assert result.value == u1.value % u2.value
        assert result.error == 0.0

        result = u1 % v2
        assert isinstance(result, Uncertainty)
        assert result.value == u1.value % v2
        assert result.error == 0.0

        # Reverse case
        result = v2 % u1
        assert isinstance(result, Uncertainty)
        assert result.value == v2 % u1.value
        assert result.error == 0.0

        # Reverse case w/ vector edge case
        result = v2 % Uncertainty(np.array([v1, v1, v1]), np.array([e1, e1, e1]))
        assert isinstance(result, Uncertainty)
        assert np.array_equal(result.error, np.array([0.0, 0.0, 0.0]))

        u1 = Uncertainty(np.array([v1, v2]), np.array([e1, e2]))
        result = u1 % np.array([1, 2])
        assert isinstance(result, Uncertainty)

    @staticmethod
    @given(
        v1=st.floats(
            min_value=0.1,
            max_value=5,
        ),
        v2=st.floats(
            min_value=0.1,
            max_value=5,
        ),
        e1=st.floats(
            min_value=0.1,
            max_value=5,
        ),
        e2=st.floats(
            min_value=0.1,
            max_value=5,
        ),
    )
    def test_pow(v1, v2, e1, e2):
        u1 = Uncertainty(v1, e1)
        u2 = Uncertainty(v2, e2)

        result = u1**u2
        assert isinstance(result, Uncertainty)
        assert result.value == u1.value**u2.value
        assert result.error == np.abs(result.value) * np.sqrt(
            (u2.value / u1.value * u1.error) ** 2
            + (np.log(np.abs(u1.value)) * u2.error) ** 2
        )

        result = u1**v2
        assert isinstance(result, Uncertainty)
        assert result.value == u1.value**v2
        assert result.error == np.abs(result.value) * np.sqrt(
            (v2 / u1.value * u1.error) ** 2 + (np.log(np.abs(u1.value)) * 0) ** 2
        )

        # Reverse case
        result = v2**u1
        assert isinstance(result, Uncertainty)
        assert result.value == v2**u1.value
        assert result.error == np.abs(result.value) * np.sqrt(
            (u1.value / v2 * 0) ** 2 + (np.log(np.abs(v2)) * u1.error) ** 2
        )

    @staticmethod
    @given(
        v=st.floats(**general_float_strategy),
        e=st.floats(
            min_value=0,
            max_value=1e3,
        ),
    )
    def test_abs(v, e):
        u = Uncertainty(v, e)
        assert abs(u).value == abs(v)
        assert abs(u).error == e

    @staticmethod
    @given(
        v=st.floats(**general_float_strategy),
        e=st.floats(
            min_value=0,
            max_value=1e3,
        ),
    )
    def test_pos(v, e):
        u = Uncertainty(v, e)
        assert (+u).value == v
        assert (+u).error == e

    @staticmethod
    @given(
        v=st.floats(**general_float_strategy),
        e=st.floats(
            min_value=0,
            max_value=1e3,
        ),
    )
    def test_neg(v, e):
        u = Uncertainty(v, e)
        assert (-u).value == -v
        assert (-u).error == e

    @staticmethod
    def test_not_implemented_ops():
        u = Uncertainty(2, 3)
        assert u.__add__("not a handled type") == NotImplemented
        assert u.__sub__("not a handled type") == NotImplemented
        assert u.__mul__("not a handled type") == NotImplemented
        assert u.__truediv__("not a handled type") == NotImplemented
        assert u.__rtruediv__("not a handled type") == NotImplemented
        assert u.__floordiv__("not a handled type") == NotImplemented
        assert u.__rfloordiv__("not a handled type") == NotImplemented
        assert u.__mod__("not a handled type") == NotImplemented
        assert u.__rmod__("not a handled type") == NotImplemented
        assert u.__pow__("not a handled type") == NotImplemented
        assert u.__rpow__("not a handled type") == NotImplemented

    @staticmethod
    def test_getattr():
        u = Uncertainty(2, 3)

        with pytest.raises(AttributeError):
            _ = u.__array_something

        for item1 in HANDLED_UFUNCS:
            assert callable(getattr(u, item1))

        for item2 in HANDLED_FUNCTIONS:
            assert callable(getattr(u, item2))

        with pytest.raises(AttributeError):
            _ = u.ATTRIBUTE_THAT_DOES_NOT_EXIST


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

    @staticmethod
    @given(
        arr1=hnp.arrays(np.float64, (3,), elements=st.floats(**general_float_strategy)),
        arr2=hnp.arrays(
            np.float64, (3,), elements=st.floats(min_value=0, max_value=1e3)
        ),
        arr3=hnp.arrays(np.float64, (3,), elements=st.floats(**general_float_strategy)),
        arr4=hnp.arrays(
            np.float64, (3,), elements=st.floats(min_value=0, max_value=1e3)
        ),
    )
    def test_ne_eq(arr1, arr2, arr3, arr4):
        assume(not np.array_equal(arr1, arr3) and not np.array_equal(arr2, arr4))

        v1 = VectorUncertainty(arr1, arr2)
        v2 = VectorUncertainty(arr3, arr4)
        v_same = VectorUncertainty(arr1, arr2)

        result = v1 == v_same
        assert np.all(result)

        result = v1 == v2
        assert not np.all(result)

        result = v1 != v2
        assert np.array_equal(result, np.logical_not(v1 == v2))

        # Test with bare array
        result = v1 == arr1
        assert np.all(result)

    @staticmethod
    @given(
        arr1=hnp.arrays(np.float64, (3,), elements=st.floats(**general_float_strategy)),
        arr2=hnp.arrays(
            np.float64, (3,), elements=st.floats(min_value=0, max_value=1e3)
        ),
    )
    def test_bytes(arr1, arr2):
        v = VectorUncertainty(arr1, arr2)
        assert v.__bytes__() == str(v).encode(locale.getpreferredencoding())

    @staticmethod
    @given(
        arr1=hnp.arrays(np.float64, (3,), elements=st.floats(**general_float_strategy)),
        arr2=hnp.arrays(
            np.float64, (3,), elements=st.floats(min_value=0, max_value=1e3)
        ),
    )
    def test_iter(arr1, arr2):
        v = VectorUncertainty(arr1, arr2)

        for idx, item in enumerate(v):
            assert item == Uncertainty(arr1[idx], arr2[idx])

    @staticmethod
    def test_properties():
        v = VectorUncertainty(np.array([1, 2, 3]), np.array([4, 5, 6]))

        assert np.array_equal(v.relative, np.array([4, 2, 2]))
        assert np.array_equal(v.rel2, v.relative**2)
        assert v.shape == (3,)
        assert v.nbytes == v._nom.nbytes + v._err.nbytes
        assert v.ndim == 1

    @staticmethod
    def test_round():
        v = VectorUncertainty(
            np.array([1.1111, 2.2222, 3.3333]), np.array([4.4444, 5.5555, 6.6666])
        )

        rounded = round(v, 2)
        assert np.array_equal(
            rounded.value, np.array([1.11, 2.22, 3.33])
        )  # should round value
        assert np.array_equal(
            rounded.error, np.array([4.4444, 5.5555, 6.6666])
        )  # should not round error

    @staticmethod
    @pytest.mark.filterwarnings(
        "ignore: __array__ implementation doesn't accept a copy keyword"
    )
    @given(
        arr1=hnp.arrays(np.float64, (3,), elements=st.floats(**general_float_strategy)),
        arr2=hnp.arrays(
            np.float64, (3,), elements=st.floats(min_value=0, max_value=1e3)
        ),
    )
    def test_array(arr1, arr2):
        v = VectorUncertainty(arr1, arr2)

        with pytest.warns(DowncastWarning):
            result = np.array(v)
            assert np.array_equal(result, np.asarray(v._nom))

        set_downcast_error(True)
        with pytest.raises(DowncastError):
            _ = np.array(v)

        set_downcast_error(False)

    @staticmethod
    def test_clip():
        v = VectorUncertainty(np.array([1, 2, 3]), np.array([4, 5, 6]))

        clipped = v.clip(min=1, max=2)
        assert np.array_equal(
            clipped.value, np.array([1, 2, 2])
        )  # value should be clipped
        assert np.array_equal(
            clipped.error, np.array([4, 5, 6])
        )  # error should not be clipped

    @staticmethod
    def test_fill():
        v = VectorUncertainty(np.empty(3), np.array([4, 5, 6]))
        v.fill(1)

        assert np.array_equal(v.value, np.array([1, 1, 1]))

    @staticmethod
    def test_put():
        v = VectorUncertainty(np.array([1, 2, 3]), np.array([4, 5, 6]))

        with pytest.raises(TypeError):
            v.put(1, 98)

        v.put(1, VectorUncertainty(np.array([98])))

        assert np.array_equal(v.value, np.array([1, 98, 3]))

    @staticmethod
    def test_copy():
        v = VectorUncertainty(np.array([1, 2, 3]), np.array([4, 5, 6]))
        dup = v.copy()

        assert np.array_equal(dup.value, v.value)
        assert np.array_equal(dup.error, v.error)
        assert id(dup) != id(v)

    @staticmethod
    def test_flat():
        v = VectorUncertainty(np.array([1, 2, 3]), np.array([4, 5, 6]))

        for i, item in enumerate(v.flat):
            assert v[i] == item

    @staticmethod
    def test_shape_reshape():
        v = VectorUncertainty(
            np.array([[1, 2, 3], [4, 5, 6]]), np.array([[4, 5, 6], [7, 8, 9]])
        )
        assert v.shape == (2, 3)

        # Return reshaped version
        reshaped = v.reshape(6)
        assert np.array_equal(reshaped.value, np.array([1, 2, 3, 4, 5, 6]))
        assert np.array_equal(reshaped.error, np.array([4, 5, 6, 7, 8, 9]))

        # In-place reshape
        v.shape = 6
        assert np.array_equal(v.value, np.array([1, 2, 3, 4, 5, 6]))
        assert np.array_equal(v.error, np.array([4, 5, 6, 7, 8, 9]))

    @staticmethod
    def test_nbytes():
        v = VectorUncertainty(np.array([1, 2, 3]), np.array([4, 5, 6]))
        assert v.nbytes == v.value.nbytes + v.error.nbytes

    @staticmethod
    def test_searchsorted():
        v = VectorUncertainty(
            np.array([1, 2, 3, 8, 9, 10]), np.array([1, 2, 3, 8, 9, 10])
        )
        assert np.array_equal(v.searchsorted([5, 6]), np.array([3, 3]))

    @staticmethod
    def test_len():
        v = VectorUncertainty(np.array([1, 2, 3]), np.array([4, 5, 6]))
        assert len(v) == len(v.value)

    @staticmethod
    @given(
        arr1=hnp.arrays(np.float64, (3,), elements=st.floats(**general_float_strategy)),
        arr2=hnp.arrays(
            np.float64, (3,), elements=st.floats(min_value=0, max_value=1e3)
        ),
    )
    def test_getitem(arr1, arr2):
        v = VectorUncertainty(arr1, arr2)

        for i in range(3):
            assert v[i] == Uncertainty(arr1[i], arr2[i])

        with pytest.raises(IndexError):
            _ = v[4]

    @staticmethod
    def test_setitem():
        v = VectorUncertainty(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))

        v[0] = Uncertainty(400.0, 100.0)
        assert v[0] == Uncertainty(400.0, 100.0)

        # Use VectorUncertainty with only one item
        v[0] = Uncertainty(np.array([400.0]), np.array([100.0]))
        assert v[0] == Uncertainty(400.0, 100.0)

        # Check NaN case
        v[0] = np.nan
        assert np.isnan(v[0])

        with pytest.raises(ValueError):
            v[0] = 400.0

        v._nom = {
            1,
            2,
            3,
            4,
            5,
        }  # Contrived set-based example to test non-indexable object
        with pytest.raises(ValueError):
            v[0] = Uncertainty(2, 3)

    @staticmethod
    @given(
        arr1=hnp.arrays(np.float64, (3,), elements=st.floats(**general_float_strategy)),
        arr2=hnp.arrays(
            np.float64, (3,), elements=st.floats(min_value=0, max_value=1e3)
        ),
    )
    def test_tolist(arr1, arr2):
        v = VectorUncertainty(arr1, arr2)
        result = v.tolist()

        assert isinstance(result, list)
        assert len(result) == 3

        for item in result:
            assert isinstance(item, Uncertainty)

    @staticmethod
    def test_tolist_edgecase():
        """Contrived test for when tolist is not available."""
        v = VectorUncertainty(np.array([1, 2, 3]), np.array([4, 5, 6]))
        v._nom = {1, 2, 3, 4}

        with pytest.raises(AttributeError):
            _ = v.tolist()

    @staticmethod
    @given(dims=st.integers(1, 64))
    def test_ndim(dims):
        args = np.ones(dims, dtype=int)
        arr = np.ones(shape=tuple([*args]))
        v = VectorUncertainty(arr, arr)
        assert v.ndim == dims

    @staticmethod
    def test_view():
        v = VectorUncertainty(np.array([1, 2, 3]), np.array([4, 5, 6]))
        view = v.view()

        assert np.array_equal(view.value, v.value.view())
        assert np.array_equal(view.error, v.error.view())

    @staticmethod
    def test_hash():
        v = VectorUncertainty(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))

        digest = joblib.hash((v._nom, v._err), hash_name="sha1")
        assert v.__hash__() == int.from_bytes(bytes(digest, encoding="utf-8"), "big")


class TestScalarUncertainty:
    """Tests that expand the coverage of the previous ScalarUncertainty tests."""

    @staticmethod
    def test_properties():
        s = ScalarUncertainty(0, 1)
        assert np.isnan(s.relative)

        s = ScalarUncertainty(5, 6)
        assert s.relative == s._err / s._nom
        assert s.rel2 == s.relative**2

        # Test overflow error catching + warning
        with pytest.warns(RuntimeWarning):
            s = ScalarUncertainty(np.float16(0.01), np.float16(6.55e4))
            assert np.isinf(s.relative)

    @staticmethod
    @given(
        v=st.floats(**general_float_strategy),
        e=st.floats(
            min_value=0,
            max_value=1e3,
        ),
    )
    def test_float(v, e):
        s = ScalarUncertainty(v, e)

        with pytest.warns(DowncastWarning):
            f = float(s)
            assert f == float(s._nom)

        set_downcast_error(True)

        with pytest.raises(DowncastError):
            _ = float(s)

        set_downcast_error(False)

    @staticmethod
    @given(
        v=st.floats(**general_float_strategy),
        e=st.floats(
            min_value=0,
            max_value=1e3,
        ),
    )
    def test_int(v, e):
        s = ScalarUncertainty(v, e)

        with pytest.warns(DowncastWarning):
            i = int(s)
            assert i == int(s._nom)

        set_downcast_error(True)

        with pytest.raises(DowncastError):
            _ = int(s)

        set_downcast_error(False)

    @staticmethod
    @given(
        v=st.floats(**general_float_strategy),
        e=st.floats(
            min_value=0,
            max_value=1e3,
        ),
    )
    def test_complex(v, e):
        s = ScalarUncertainty(v, e)

        with pytest.warns(DowncastWarning):
            f = complex(s)
            assert f == complex(s._nom)

        set_downcast_error(True)

        with pytest.raises(DowncastError):
            _ = complex(s)

        set_downcast_error(False)

    @staticmethod
    def test_round():
        s = ScalarUncertainty(2.2222, 3.3333)

        rounded = round(s, 2)
        assert rounded.value == 2.22
        assert rounded.error == 3.3333

    @staticmethod
    @given(
        v1=st.floats(**general_float_strategy),
        e1=st.floats(min_value=0, max_value=1e3),
        v2=st.floats(**general_float_strategy),
        e2=st.floats(min_value=0, max_value=1e3),
    )
    def test_ne_eq(v1, e1, v2, e2):
        assume(e1 != e2)
        assume(not np.isclose(v1, v2))

        s1 = ScalarUncertainty(v1, e1)
        s2 = ScalarUncertainty(v2, e2)
        s_same = ScalarUncertainty(v1, e1)

        result = s1 == s_same
        assert result is True

        result = s1 == s2
        assert result is False

        result = s1 != s2
        assert result == (s1 != s2)

        # Test with bare number
        result = s1 == v1
        assert result is True

        # Test invalid inputs
        result = s1 == "something"
        assert result is False

        s2._nom = "bad value"
        result = s1 == s2
        assert result is False

    @staticmethod
    def test_hash():
        s = ScalarUncertainty(1, 2)
        assert s.__hash__() == hash((s._nom, s._err))
