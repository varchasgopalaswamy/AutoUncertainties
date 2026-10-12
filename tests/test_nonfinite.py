from __future__ import annotations

import copy

import numpy as np
from pint import Quantity
import pytest

from auto_uncertainties import NegativeStdDevError, Uncertainty, configure


@pytest.mark.parametrize("dtype", [float, np.float32, np.float64])
@pytest.mark.parametrize("value,error", [(1, np.nan), (np.nan, 2), (np.nan, np.nan)])
def test_scalar_nan_propagation(dtype, value, error):
    configure(nonfinite_error_policy="propagate")
    u = Uncertainty(dtype(value), dtype(error))
    assert np.isnan(u.value) and np.isnan(u.error)
    assert type(u.value) is dtype and type(u.error) is dtype


@pytest.mark.parametrize("policy", ["zero", "propagate"])
@pytest.mark.parametrize("vector", [False, True])
def test_infinite_errors(policy, vector):
    configure(nonfinite_error_policy=policy)
    value = np.array([1.0]) if vector else 1.0
    error = np.array([np.inf]) if vector else np.inf
    u = Uncertainty(value, error)
    np.testing.assert_equal(u.error, np.inf if policy == "propagate" else 0)
    if policy == "propagate":
        with pytest.raises(NegativeStdDevError):
            Uncertainty(value, -error)
    else:
        np.testing.assert_equal(Uncertainty(value, -error).error, 0)


@pytest.mark.parametrize("policy", ["zero", "propagate"])
def test_array_normalization_does_not_mutate_inputs(policy):
    configure(nonfinite_error_policy=policy)
    value = np.array([[1, np.nan], [3, 4]])
    error = np.array([[np.nan, 2], [np.inf, 0]])
    original_value, original_error = value.copy(), error.copy()
    value.flags.writeable = error.flags.writeable = False
    u = Uncertainty(value, error)
    np.testing.assert_equal(value, original_value)
    np.testing.assert_equal(error, original_error)
    if policy == "propagate":
        np.testing.assert_equal(u.value, [[np.nan, np.nan], [3, 4]])
        np.testing.assert_equal(u.error, [[np.nan, np.nan], [np.inf, 0]])
    else:
        np.testing.assert_equal(u.value, original_value)
        np.testing.assert_equal(u.error, [[0, 2], [0, 0]])


def test_broadcasting_sequences_and_missing_errors():
    configure(nonfinite_error_policy="propagate")
    for value in [np.array([1, 2]), [1, 2], (1, 2)]:
        u = Uncertainty(value, np.nan)
        assert np.isnan(u.value).all() and np.isnan(u.error).all()
    u = Uncertainty(np.array([np.nan, 2]))
    np.testing.assert_equal(u.error, [np.nan, 0])
    assert np.isnan(Uncertainty(np.nan).error)
    assert Uncertainty(2).error == 0


def test_zero_dimensional_inputs_and_integer_errors():
    configure(nonfinite_error_policy="propagate")
    u = Uncertainty(np.array(1), np.array(np.nan))
    assert not u.is_vector
    assert np.isnan(u.value) and np.isnan(u.error)
    v = Uncertainty(np.array([np.nan, 2]), np.array([1, 2]))
    np.testing.assert_equal(v.error, [np.nan, 2])


def test_copy_and_quantity_construction():
    configure(nonfinite_error_policy="propagate")
    u = Uncertainty(1, np.nan)
    for result in [Uncertainty(u), copy.copy(u), copy.deepcopy(u), Uncertainty([u])[0]]:
        assert np.isnan(result.value) and np.isnan(result.error)
    q = Uncertainty(Quantity(1, "m"), Quantity(np.nan, "m"))
    assert np.isnan(q.magnitude.value) and np.isnan(q.magnitude.error)
    assert q.units == Quantity(1, "m").units


@pytest.mark.parametrize("policy", ["zero", "propagate"])
def test_assignment(policy):
    configure(nonfinite_error_policy=policy)
    u = Uncertainty([1, 2, 3], [0.1, 0.2, 0.3])
    u[0] = np.nan
    u[1:] = np.nan
    assert np.isnan(u.value).all()
    np.testing.assert_equal(u.error, np.nan if policy == "propagate" else 0)


def test_configuration_changes_only_affect_subsequent_operations():
    u = Uncertainty(np.nan, 2)
    configure(nonfinite_error_policy="propagate")
    assert u.error == 2
    v = Uncertainty([1, 2])
    v[0] = u
    assert np.isnan(v.error[0])


def test_arithmetic_and_numpy_propagation():
    configure(nonfinite_error_policy="propagate")
    u = Uncertainty(1, np.nan)
    for result in [u + 2, u * 2, np.sin(u), np.sum(Uncertainty([1, 2], [0.1, np.nan]))]:
        assert np.isnan(result.value) and np.isnan(result.error)
    v = Uncertainty(1, np.inf) * 2
    assert v.value == 2 and np.isposinf(v.error)


@pytest.mark.parametrize("rounding", [False, True])
@pytest.mark.parametrize("value,error", [(np.nan, np.nan), (1, np.inf), (np.inf, 1)])
def test_nonfinite_display(rounding, value, error):
    configure(nonfinite_error_policy="propagate", round_on_display=rounding)
    expected = f"{value:g} +/- {error:g}"
    scalar = Uncertainty(value, error)
    vector = Uncertainty([value], [error])
    assert str(scalar) == expected
    assert format(scalar, "g") == expected
    assert str(vector) == f"[{expected}]"
    assert scalar._repr_html_() == expected.replace("+/-", "±")
    assert (
        "nan" in vector._repr_latex_()
        if np.isnan(value)
        else "inf" in vector._repr_latex_()
    )
