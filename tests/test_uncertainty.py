from __future__ import annotations

import copy
import itertools
import locale
import math
import operator
import warnings

from hypothesis import assume, given
from hypothesis.extra import numpy as hnp
import hypothesis.strategies as st
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
    EqualityError,
    EqualityWarning,
    NegativeStdDevError,
    Uncertainty,
    configure,
    nominal_values,
    std_devs,
)
from auto_uncertainties.units import split_units

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


general_float_strategy = {
    "allow_nan": False,
    "allow_infinity": False,
    "allow_subnormal": False,
    "min_value": -1e3,
    "max_value": 1e3,
}


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@given(
    v=st.floats(),
    e=st.floats(),
    units=st.sampled_from(UNITS),
)
def test_scalar_creation(v, e, units):
    const = Uncertainty

    if not np.isfinite(e):
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
        assert math.isclose(u.error, np.hypot(u1.error, u2.error))


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
        np.testing.assert_almost_equal(u.error, u.value * np.sqrt(u1.rel**2 + u2.rel**2))


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
        elements=st.floats(min_value=0, max_value=10.0, allow_nan=False, allow_infinity=False),
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


def testsplit_units():
    val, err = Quantity(2, "radian"), Quantity(3, "radian")
    assert split_units(val, err) == (2, 3, Unit("radian"))

    val, err = 2, Quantity(3, "radian")
    assert split_units(val, err) == (2, 3, Unit("radian"))

    val, err = Quantity(2, "radian"), 3
    assert split_units(val, err) == (2, 3, Unit("radian"))

    val, err = 2, 3
    assert split_units(val, err) == (2, 3, None)

    val, err = Quantity(2, "m"), Quantity(3, "deg")
    with pytest.raises(DimensionalityError):
        _ = split_units(val, err)


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
        assert u.__getstate__() == {"nominal_value": u.nom, "std_devs": u.err}

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
        fake_vector = Uncertainty(np.array(3), np.array(4))
        assert isinstance(scalar, Uncertainty)  # verify scalar type was chosen (DEPRECATED)
        assert isinstance(vector, Uncertainty)  # verify vector type was chosen (DEPRECATED)
        assert isinstance(fake_vector, Uncertainty)  # verify zero-D vectors become scalars

        # Verify inheritance
        assert isinstance(scalar, Uncertainty)
        assert isinstance(vector, Uncertainty)

        # Check creating from other Uncertainty objects
        from_subclass = Uncertainty(vector)
        assert isinstance(from_subclass, Uncertainty)
        assert np.array_equal(from_subclass.value, vector.value)
        assert np.array_equal(from_subclass.error, vector.error)

        # Check creating from a Sequence
        from_list = Uncertainty([Uncertainty(1, 2), Uncertainty(4, 5), Uncertainty(1, 9)])
        assert isinstance(from_list, Uncertainty)

        # Check vector value, constant error edge case
        v = Uncertainty(np.array([1, 2, 3]), 0)
        assert isinstance(v, Uncertainty)

        # Check proper handling of Quantity inputs
        # (further tests for from_quantities are handled in a separate function)
        from_quant = Uncertainty(Quantity(2, "radian"), Quantity(1, "radian"))
        assert isinstance(from_quant, Quantity)
        assert isinstance(from_quant.m, Uncertainty)
        assert from_quant.units == Unit("radian")

    def test_init_param_combinations(self):
        values_scalars = [
            2,
            2.5,
            Uncertainty(0.5, 0.01),
            Quantity(2, "radian"),
            Quantity(2.5, "radian"),
        ]
        values_sequences = [
            [1, 2, 3],
            [1.5, 2.5, 4.5],
            [Uncertainty(1, 2), Uncertainty(3, 4), Uncertainty(4, 5)],
            [Quantity(1, "radian"), Quantity(2, "radian"), Quantity(3, "radian")],
        ]
        values_arrays = [
            np.array([1, 2, 3]),
            np.array([1.5, 2.5, 3.25]),
        ]
        errors_scalars = [
            2,
            2.5,
            Quantity(2, "radian"),
            Quantity(2.5, "radian"),
            None,
        ]
        errors_sequences = [
            [1, 2, 3],
            [1.5, 2.5, 3.4],
            [Quantity(1, "radian"), Quantity(2, "radian"), Quantity(3, "radian")],
        ]
        errors_arrays = [
            np.array([1, 2, 3]),
            np.array([1.5, 2.5, 3.25]),
        ]

        # Scalar and scalar case.
        for val, err in itertools.product(values_scalars, errors_scalars):
            _ = Uncertainty(val, err)

        # Sequence and scalar case.
        for val, err in itertools.product(values_sequences, errors_scalars):
            _ = Uncertainty(val, err)

        # Array and scalar case.
        for val, err in itertools.product(values_arrays, errors_scalars):
            _ = Uncertainty(val, err)

        # Sequence and sequence case.
        for val, err in itertools.product(values_sequences, errors_sequences):
            _ = Uncertainty(val, err)

        # Array and array case.
        for val, err in itertools.product(values_arrays, errors_arrays):
            _ = Uncertainty(val, err)

    @pytest.mark.parametrize(
        ("val", "err", "exception"),
        (
            ([1, 2, 3], np.array([1, 2, 3]), TypeError),
            (np.array([1, 2, 3]), [1, 2, 3], TypeError),
            (2.5, [1, 2, 3], TypeError),
            (({"bad_type": True}), None, TypeError),
            (np.array([1, 2, 3]), np.array([1, 2]), ValueError),
            ([1, 2, 3], [1, 2], ValueError),
            (np.array([1, 2, 3]), np.array([-1, 2, 3]), NegativeStdDevError),
            ([Uncertainty(1, 0.5), {"bad_data": True}], None, TypeError),
            ([1, 2, 3], [1, 2, Uncertainty(0.5, 0.25)], TypeError),
        ),
    )
    def test_init_exceptions(self, val, err, exception):
        with pytest.raises(exception):
            _ = Uncertainty(val, err)

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
        assert isinstance(u, Uncertainty)

        assert u.value == u.nom
        assert u.error == u.err
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

    def test_plus_minus_dtype(self):
        u = Uncertainty(np.float32(1.5), np.float32(0.5))
        result = u.plus_minus(25)
        assert np.isclose(result.error, 25.005)
        assert type(result.error) is np.float32

        u = Uncertainty(1.5, 0.5)
        result = u.plus_minus(25)
        assert np.isclose(result.error, 25.005)
        assert type(result.error) is float

    @staticmethod
    @pytest.mark.parametrize(
        ("val", "expected"),
        (
            ("2.0 +/- 3.5", Uncertainty(2.0, 3.5)),
            ("2.0 +- 3.5", Uncertainty(2.0, 3.5)),
            ("2.6", Uncertainty(2.6)),
        ),
    )
    def test_from_string(val, expected):
        assert Uncertainty.from_string(val) == expected

    @staticmethod
    @pytest.mark.parametrize(
        ("val", "err"),
        (
            (Quantity(2, "radian"), Quantity(3, "radian")),
            (Quantity(2, "radian"), 3),
            (2, Quantity(3, "radian")),
            (2, 3),
            (8.5, 9.5),
        ),
    )
    def test_from_quantities(val, err):
        assert Uncertainty.from_quantities(val, err).value == val
        assert Uncertainty.from_quantities(val, err).error == err

    @staticmethod
    def test_from_sequence():
        seq = [Uncertainty(2, 3), Uncertainty(3, 4), Uncertainty(4, 5)]
        result = Uncertainty.from_sequence(seq)
        assert isinstance(result, Uncertainty)
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

        # Test with sequence of errors
        seq_v = [Quantity(2, "degree"), Quantity(4, "radian")]
        seq_e = [Quantity(2, "radian"), Quantity(6, "degree")]
        result = Uncertainty(seq_v, seq_e)
        assert result.units == Unit("degree")

        # Case where value has no units, error has units.
        seq_v = [10.25, 40.5]
        seq_e = [Quantity(2, "radian"), Quantity(6, "degree")]
        result = Uncertainty(seq_v, seq_e)
        assert result.units == Unit("radian")

        # Mixed sequence of Uncertainties and scalars.
        u = Uncertainty([10.25, Uncertainty(1, 0.4), 0.5])
        assert u.value.all() == np.array([10.25, 1.0, 0.5]).all()
        assert u.error.all() == np.array([0.0, 0.4, 0.0]).all()

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
        assert result.error == np.hypot(u1.error, u2.error)

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

        try:
            result = op(u1, u2)
            assert isinstance(result, Uncertainty)
            assert result.value == op(u1.value, u2.value)
            assert np.isclose(result.error, np.abs(result.value) * np.sqrt(u1.rel2 + u2.rel2))
        except OverflowError:
            pass

        result = op(u1, v2)
        assert isinstance(result, Uncertainty)
        assert result.value == op(u1.value, v2)
        assert math.isclose(result.error, np.abs(op(u1.error, v2)))

        # Reverse case
        result = op(v2, u1)
        assert isinstance(result, Uncertainty)
        assert result.value == op(v2, u1.value)
        expected_error = (
            np.abs(result.value) * np.abs(u1.rel)
            if op == operator.truediv
            else np.abs(op(u1.error, v2))
        )
        assert math.isclose(result.error, expected_error)

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

        try:
            result = u1 // u2
            assert isinstance(result, Uncertainty)
            assert result.value == u1.value // u2.value
            assert result.error == (u1 / u2).error
        except OverflowError:
            pass

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
        expected_error = np.hypot(u1.error, (u1.value // u2.value) * u2.error)
        assert result.error == (expected_error if np.isfinite(expected_error) else 0)

        result = u1 % v2
        assert isinstance(result, Uncertainty)
        assert result.value == u1.value % v2
        assert result.error == u1.error

        # Reverse case
        result = v2 % u1
        assert isinstance(result, Uncertainty)
        assert result.value == v2 % u1.value
        expected_error = np.abs((v2 // u1.value) * u1.error)
        assert result.error == (expected_error if np.isfinite(expected_error) else 0)

        # Reverse case w/ vector edge case
        result = v2 % Uncertainty(np.array([v1, v1, v1]), np.array([e1, e1, e1]))
        assert isinstance(result, Uncertainty)
        np.testing.assert_allclose(
            result.error, expected_error if np.isfinite(expected_error) else 0
        )

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
        assert result.value == pytest.approx(u1.value**u2.value)
        assert result.error == pytest.approx(
            np.abs(result.value)
            * np.sqrt(
                (u2.value / u1.value * u1.error) ** 2 + (np.log(np.abs(u1.value)) * u2.error) ** 2
            )
        )

        result = u1**v2
        assert isinstance(result, Uncertainty)
        assert result.value == pytest.approx(u1.value**v2)
        assert result.error == pytest.approx(
            np.abs(result.value)
            * np.sqrt((v2 / u1.value * u1.error) ** 2 + (np.log(np.abs(u1.value)) * 0) ** 2)
        )

        # Reverse case
        result = v2**u1
        assert isinstance(result, Uncertainty)
        assert result.value == pytest.approx(v2**u1.value)
        assert result.error == pytest.approx(
            np.abs(result.value)
            * np.sqrt((u1.value / v2 * 0) ** 2 + (np.log(np.abs(v2)) * u1.error) ** 2)
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
    @given(
        arr1=hnp.arrays(np.float64, (3,), elements=st.floats(**general_float_strategy)),
        arr2=hnp.arrays(np.float64, (3,), elements=st.floats(min_value=0, max_value=1e3)),
    )
    def test_bytes(arr1, arr2):
        v = Uncertainty(arr1, arr2)
        assert v.__bytes__() == str(v).encode(locale.getpreferredencoding())

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
    def test_explicit_scalar_methods():
        u = Uncertainty(2, 3)
        assert callable(u.sum)
        assert callable(u.reshape)
        for name in ("sin", "add", "concatenate", "median", "__array_something", "unknown"):
            with pytest.raises(AttributeError):
                getattr(u, name)


class TestVectorUncertainty:
    """Tests that expand the coverage of the previous Uncertainty tests."""

    @staticmethod
    def test_explicit_array_methods():
        v = Uncertainty(np.array([1, 2, 3]), np.array([4, 5, 6]))
        for name in (
            "reshape",
            "flatten",
            "astype",
            "sum",
            "mean",
            "std",
            "var",
            "sort",
            "compress",
            "conjugate",
        ):
            assert callable(getattr(v, name))
        for name in ("dtype", "ndim", "size"):
            assert getattr(v, name) == getattr(v.value, name)
        for name in ("T", "real", "imag"):
            result = getattr(v, name)
            np.testing.assert_array_equal(result.value, getattr(v.value, name))
            np.testing.assert_array_equal(result.error, getattr(v.error, name))
        for name in ("sin", "add", "concatenate", "__array_something", "unknown"):
            with pytest.raises(AttributeError):
                getattr(v, name)

    @staticmethod
    @given(
        arr1=hnp.arrays(np.float64, (3,), elements=st.floats(**general_float_strategy)),
        arr2=hnp.arrays(np.float64, (3,), elements=st.floats(min_value=0, max_value=1e3)),
        arr3=hnp.arrays(np.float64, (3,), elements=st.floats(**general_float_strategy)),
        arr4=hnp.arrays(np.float64, (3,), elements=st.floats(min_value=0, max_value=1e3)),
    )
    def test_ne_eq(arr1, arr2, arr3, arr4):
        assume(not np.array_equal(arr1, arr3) and not np.array_equal(arr2, arr4))

        v1 = Uncertainty(arr1, arr2)
        v2 = Uncertainty(arr3, arr4)
        v_same = Uncertainty(arr1, arr2)

        result = v1 == v_same
        assert np.all(result)

        result = v1 == v2
        assert not np.all(result)

        result = v1 != v2
        assert np.array_equal(result, np.logical_not(v1 == v2))

        # Test with bare array
        result = v1 == arr1
        assert np.all(result)

    def test_eq_exceptions(self):
        s1 = Uncertainty([1, 2, 3], [0, 0, 1])
        s2 = Uncertainty([1, 2, 3], [0, 0, 0])

        with pytest.warns(EqualityWarning):
            _ = s1 == s2
        with pytest.warns(EqualityWarning):
            _ = s1 == np.array([1.0, 2.0, 3.0])

        configure(error_on_equality=True)

        with pytest.raises(EqualityError):
            _ = s1 == s2
        with pytest.raises(EqualityError):
            _ = s1 == np.array([1.0, 2.0, 3.0])

        configure(error_on_equality=False)

    @staticmethod
    @given(
        arr1=hnp.arrays(np.float64, (3,), elements=st.floats(**general_float_strategy)),
        arr2=hnp.arrays(np.float64, (3,), elements=st.floats(min_value=0, max_value=1e3)),
    )
    def test_iter(arr1, arr2):
        v = Uncertainty(arr1, arr2)

        for idx, item in enumerate(v):
            assert item == Uncertainty(arr1[idx], arr2[idx])

    @staticmethod
    def test_properties():
        v = Uncertainty(np.array([1, 2, 3]), np.array([4, 5, 6]))
        v2 = Uncertainty(np.array([1, 0, np.inf]), np.array([4, 5, 6]))  # edge case

        assert np.array_equal(v.relative, np.array([4.0, 2.5, 2.0]))
        assert np.isnan(v2.relative[1])
        assert np.isinf(v2.relative[2])
        assert np.array_equal(v.rel2, v.relative**2)
        assert v.shape == (3,)
        assert v.nbytes == v.nom.nbytes + v.err.nbytes
        assert v.ndim == 1

    @staticmethod
    def test_round():
        v = Uncertainty(np.array([1.1111, 2.2222, 3.3333]), np.array([4.4444, 5.5555, 6.6666]))

        rounded = round(v, 2)
        assert np.array_equal(rounded.value, np.array([1.11, 2.22, 3.33]))  # should round value
        assert np.array_equal(
            rounded.error, np.array([4.4444, 5.5555, 6.6666])
        )  # should not round error

    @staticmethod
    @pytest.mark.filterwarnings("ignore: __array__ implementation doesn't accept a copy keyword")
    @given(
        arr1=hnp.arrays(np.float64, (3,), elements=st.floats(**general_float_strategy)),
        arr2=hnp.arrays(np.float64, (3,), elements=st.floats(min_value=0, max_value=1e3)),
    )
    def test_array(arr1, arr2):
        v = Uncertainty(arr1, arr2)

        with pytest.warns(DowncastWarning):
            result = np.array(v)
        assert np.array_equal(result, np.asarray(v.nom))

        configure(error_on_downcast=True)
        with pytest.raises(DowncastError):
            _ = np.array(v)

        configure(error_on_downcast=False)

    @staticmethod
    def test_clip():
        v = Uncertainty(np.array([1, 2, 3]), np.array([4, 5, 6]))

        clipped = v.clip(min=1, max=2)
        assert np.array_equal(clipped.value, np.array([1, 2, 2]))  # value should be clipped
        assert np.array_equal(clipped.error, np.array([4, 5, 6]))  # error should not be clipped

    @staticmethod
    def test_fill():
        v = Uncertainty(np.empty(3), np.array([4, 5, 6]))
        v.fill(1)

        assert np.array_equal(v.value, np.array([1, 1, 1]))

    @staticmethod
    def test_put():
        v = Uncertainty(np.array([1, 2, 3]), np.array([4, 5, 6]))

        with pytest.raises(TypeError):
            v.put(1, 98)

        v.put(1, Uncertainty(np.array([98])))

        assert np.array_equal(v.value, np.array([1, 98, 3]))

    @staticmethod
    def test_copy():
        v = Uncertainty(np.array([1, 2, 3]), np.array([4, 5, 6]))
        dup = v.copy()

        assert np.array_equal(dup.value, v.value)
        assert np.array_equal(dup.error, v.error)
        assert id(dup) != id(v)

    @staticmethod
    def test_flat():
        v = Uncertainty(np.array([1, 2, 3]), np.array([4, 5, 6]))

        for i, item in enumerate(v.flat):
            assert v[i] == item

    @staticmethod
    def test_shape_reshape():
        v = Uncertainty(np.array([[1, 2, 3], [4, 5, 6]]), np.array([[4, 5, 6], [7, 8, 9]]))
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
        v = Uncertainty(np.array([1, 2, 3]), np.array([4, 5, 6]))
        assert v.nbytes == v.value.nbytes + v.error.nbytes

    @staticmethod
    def test_searchsorted():
        v = Uncertainty(np.array([1, 2, 3, 8, 9, 10]), np.array([1, 2, 3, 8, 9, 10]))
        assert np.array_equal(v.searchsorted([5, 6]), np.array([3, 3]))

    @staticmethod
    def test_len():
        v = Uncertainty(np.array([1, 2, 3]), np.array([4, 5, 6]))
        assert len(v) == len(v.value)

    @staticmethod
    @given(
        arr1=hnp.arrays(np.float64, (3,), elements=st.floats(**general_float_strategy)),
        arr2=hnp.arrays(np.float64, (3,), elements=st.floats(min_value=0, max_value=1e3)),
    )
    def test_getitem(arr1, arr2):
        v = Uncertainty(arr1, arr2)

        for i in range(3):
            assert v[i] == Uncertainty(arr1[i], arr2[i])

        with pytest.raises(IndexError):
            _ = v[4]

    @staticmethod
    def test_setitem():
        v = Uncertainty(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))

        v[0] = Uncertainty(400.0, 100.0)
        assert v[0] == Uncertainty(400.0, 100.0)

        # Use Uncertainty with only one item
        v[0] = Uncertainty(np.array([400.0]), np.array([100.0]))
        assert v[0] == Uncertainty(400.0, 100.0)

        # Check NaN case
        v[0] = np.nan
        assert np.isnan(v[0])

        with pytest.raises(TypeError):
            v[0] = 400.0

    @staticmethod
    @given(
        arr1=hnp.arrays(np.float64, (3,), elements=st.floats(**general_float_strategy)),
        arr2=hnp.arrays(np.float64, (3,), elements=st.floats(min_value=0, max_value=1e3)),
    )
    def test_tolist(arr1, arr2):
        v = Uncertainty(arr1, arr2)
        result = v.tolist()

        assert isinstance(result, list)
        assert len(result) == 3

        for item in result:
            assert isinstance(item, Uncertainty)

    @staticmethod
    def test_tolist_edgecase():
        """Contrived test for when tolist is not available."""
        v = Uncertainty(np.array([1, 2, 3]), np.array([4, 5, 6]))
        v.nom = {1, 2, 3, 4}

        with pytest.raises(TypeError):
            _ = v.tolist()

    @staticmethod
    @given(dims=st.integers(1, 64))
    def test_ndim(dims):
        args = np.ones(dims, dtype=int)
        arr = np.ones(shape=tuple([*args]))
        v = Uncertainty(arr, arr)
        assert v.ndim == dims

    @staticmethod
    def test_view():
        v = Uncertainty(np.array([1, 2, 3]), np.array([4, 5, 6]))
        view = v.view()

        assert np.array_equal(view.value, v.value.view())
        assert np.array_equal(view.error, v.error.view())

    @staticmethod
    def test_hash():
        v = Uncertainty(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))

        with pytest.raises(TypeError, match="unhashable"):
            hash(v)

    def test_unimplemented_methods(self):
        v = Uncertainty(np.array([1, 2, 3]))

        with pytest.raises(TypeError):
            _ = float(v)
        with pytest.raises(TypeError):
            _ = int(v)
        with pytest.raises(TypeError):
            _ = complex(v)


class TestScalarUncertainty:
    """Tests that expand the coverage of the previous Uncertainty tests."""

    @staticmethod
    def test_properties():
        s = Uncertainty(0, 1)
        assert np.isnan(s.relative)

        s = Uncertainty(5, 6)
        assert s.relative == s.err / s.nom
        assert s.rel2 == s.relative**2

        # Test overflow error catching + warning
        s = Uncertainty(np.float16(0.01), np.float16(6.55e4))
        with pytest.warns(RuntimeWarning):
            relative = s.relative
        assert np.isinf(relative)

    @staticmethod
    @given(
        v=st.floats(**general_float_strategy),
        e=st.floats(
            min_value=0,
            max_value=1e3,
        ),
    )
    def test_float(v, e):
        s = Uncertainty(v, e)

        with pytest.warns(DowncastWarning):
            f = float(s)
        assert f == float(s.nom)

        configure(error_on_downcast=True)

        with pytest.raises(DowncastError):
            _ = float(s)

        configure(error_on_downcast=False)

    @staticmethod
    @given(
        v=st.floats(**general_float_strategy),
        e=st.floats(
            min_value=0,
            max_value=1e3,
        ),
    )
    def test_int(v, e):
        s = Uncertainty(v, e)

        with pytest.warns(DowncastWarning):
            i = int(s)
        assert i == int(s.nom)

        configure(error_on_downcast=True)

        with pytest.raises(DowncastError):
            _ = int(s)

        configure(error_on_downcast=False)

    @staticmethod
    @given(
        v=st.floats(**general_float_strategy),
        e=st.floats(
            min_value=0,
            max_value=1e3,
        ),
    )
    def test_complex(v, e):
        s = Uncertainty(v, e)

        with pytest.warns(DowncastWarning):
            f = complex(s)
        assert f == complex(s.nom)

        configure(error_on_downcast=True)

        with pytest.raises(DowncastError):
            _ = complex(s)

        configure(error_on_downcast=False)

    @staticmethod
    def test_round():
        s = Uncertainty(2.2222, 3.3333)

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

        s1 = Uncertainty(v1, e1)
        s2 = Uncertainty(v2, e2)
        s_same = Uncertainty(v1, e1)

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

        s2.nom = "bad value"
        result = s1 == s2
        assert result is False

    def test_eq_exceptions(self):
        s1 = Uncertainty(1, 2)
        s2 = Uncertainty(1, 3)
        with pytest.warns(EqualityWarning):
            _ = s1 == s2
        with pytest.warns(EqualityWarning):
            _ = s1 == 1.0

        configure(error_on_equality=True)

        with pytest.raises(EqualityError):
            _ = s1 == s2
        with pytest.raises(EqualityError):
            _ = s1 == 1.0

        configure(error_on_equality=False)

    @staticmethod
    def test_hash():
        s = Uncertainty(1, 2)
        with pytest.raises(TypeError, match="unhashable"):
            hash(s)

    def test_unimplemented_methods(self):
        s = Uncertainty(1, 2)

        with pytest.raises(TypeError):
            _ = s[0]
        with pytest.raises(TypeError):
            s[0] = 1.0
        with pytest.raises(TypeError):
            _ = next(s.__iter__())
        with pytest.raises(TypeError):
            _ = s.clip(min=1, max=2)
        with pytest.raises(TypeError):
            s.fill(1)
        with pytest.raises(TypeError):
            s.put(1, Uncertainty(2, 3))
        with pytest.raises(TypeError):
            _ = s.copy()
        with pytest.raises(TypeError):
            _ = next(s.flat)
        with pytest.raises(TypeError):
            _ = s.shape
        with pytest.raises(TypeError):
            _ = s.shape = (2, 3)
        with pytest.raises(TypeError):
            _ = s.nbytes
        with pytest.raises(TypeError):
            _ = s.searchsorted(1)
        with pytest.raises(TypeError):
            _ = len(s)
        with pytest.raises(TypeError):
            _ = s.view()
