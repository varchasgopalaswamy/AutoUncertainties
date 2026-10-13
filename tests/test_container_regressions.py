"""Analytic regressions for arithmetic and container protocol edge cases."""

from __future__ import annotations

import operator

import numpy as np
import pytest

from auto_uncertainties import DowncastWarning, Uncertainty, configure


@pytest.mark.parametrize("policy", ("zero", "propagate"))
@pytest.mark.parametrize(("exponent", "expected_error"), ((0, 0), (1, 2), (2, 0)))
def test_zero_base_power(policy, exponent, expected_error):
    configure(nonfinite_error_policy=policy)
    scalar = Uncertainty(0, 2) ** exponent
    vector = Uncertainty([0, 0], [2, 2]) ** exponent
    assert scalar.value == 0**exponent
    assert scalar.error == expected_error
    np.testing.assert_equal(vector.value, [0**exponent, 0**exponent])
    np.testing.assert_equal(vector.error, [expected_error, expected_error])


@pytest.mark.parametrize("policy", ("zero", "propagate"))
def test_zero_base_with_uncertain_positive_exponent(policy):
    configure(nonfinite_error_policy=policy)
    result = Uncertainty(0, 2) ** Uncertainty(1, 0.1)
    assert result.value == 0
    assert result.error == 2
    result = 0 ** Uncertainty(2, 0.1)
    assert result.value == 0
    assert result.error == 0


def test_power_avoids_spurious_overflow_at_small_base():
    result = Uncertainty(1e-300, 1e-5) ** 1
    assert result.value == 1e-300
    assert result.error == 1e-5


@pytest.mark.parametrize("dtype", (np.float16, np.float32, np.float64))
def test_power_preserves_numpy_dtype_with_exact_scalar_exponent(dtype):
    value = np.array([1, 2], dtype=dtype)
    result = Uncertainty(value, np.full_like(value, 0.1)) ** 2
    assert result.value.dtype == result.error.dtype == dtype
    np.testing.assert_allclose(result.value, [1, 4])
    np.testing.assert_allclose(result.error, [0.2, 0.4], rtol=0.001)


def test_zero_base_fractional_power_preserves_infinite_derivative():
    configure(nonfinite_error_policy="propagate")
    result = Uncertainty(0, 2) ** 0.5
    assert result.value == 0
    assert np.isposinf(result.error)


@pytest.mark.parametrize(("base", "exponent", "expected"), ((2, 0, 1), (0, 2, 0)))
def test_zero_power_derivative_ignores_infinite_base_error(base, exponent, expected):
    configure(nonfinite_error_policy="propagate")
    scalar = Uncertainty(base, np.inf) ** exponent
    vector = Uncertainty([base, base], np.inf) ** exponent
    assert scalar.value == expected
    assert scalar.error == 0
    np.testing.assert_equal(vector.value, [expected, expected])
    np.testing.assert_equal(vector.error, [0, 0])


def test_zero_power_derivative_ignores_infinite_exponent_error():
    configure(nonfinite_error_policy="propagate")
    scalar = 1 ** Uncertainty(2, np.inf)
    vector = np.array([1.0, 1.0]) ** Uncertainty([2, 3], np.inf)
    assert scalar.value == 1
    assert scalar.error == 0
    np.testing.assert_equal(vector.value, [1, 1])
    np.testing.assert_equal(vector.error, [0, 0])


@pytest.mark.parametrize(("divisor", "coefficient"), ((2, 2), (-2, -3), (6, 0)))
def test_modulo_propagates_divisor_error(divisor, coefficient):
    result = Uncertainty(5, 0.1) % Uncertainty(divisor, 0.2)
    assert result.value == 5 % divisor
    assert result.error == pytest.approx(np.hypot(0.1, coefficient * 0.2))
    reverse = 5 % Uncertainty(divisor, 0.2)
    assert reverse.value == 5 % divisor
    assert reverse.error == pytest.approx(abs(coefficient) * 0.2)


@pytest.mark.parametrize("operation", (operator.add, operator.sub, operator.mod))
def test_exact_array_broadcasts_errors_and_result_is_mutable(operation):
    value = Uncertainty(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
    exact = np.array([[3.0], [4.0]])
    result = operation(value, exact)
    np.testing.assert_equal(result.value, operation(value.value, exact))
    np.testing.assert_equal(result.error, [[0.1, 0.2], [0.1, 0.2]])
    result[0, 0] = Uncertainty(9, 0.5)
    assert result.error[0, 0] == 0.5
    np.testing.assert_equal(value.error, [0.1, 0.2])


def test_searchsorted_respects_sorter():
    result = Uncertainty([3, 1, 2]).searchsorted([1, 2, 3], side="right", sorter=[1, 2, 0])
    np.testing.assert_equal(result, [1, 2, 3])


@pytest.mark.parametrize("value", (1.2, np.array([1.2, 2.8])))
@pytest.mark.parametrize("explicit_none", (False, True))
def test_round_accepts_default_precision(value, explicit_none):
    uncertain = Uncertainty(value, 0.1)
    result = round(uncertain, None) if explicit_none else round(uncertain)
    np.testing.assert_equal(result.value, np.round(value))
    np.testing.assert_equal(result.error, uncertain.error)


def test_failed_shape_assignment_preserves_both_array_shapes():
    nominal = np.arange(6.0).reshape(2, 3)
    error = np.arange(6.0).reshape(3, 2).T
    uncertain = Uncertainty(nominal, error)
    with pytest.raises(AttributeError, match="shape"):
        uncertain.shape = (6,)
    assert uncertain.value.shape == uncertain.error.shape == (2, 3)
    assert nominal.shape == error.shape == (2, 3)


def test_shape_assignment_updates_both_arrays():
    uncertain = Uncertainty(np.arange(6.0).reshape(2, 3), 0.1)
    uncertain.shape = (6,)
    assert uncertain.value.shape == uncertain.error.shape == (6,)


def test_array_conversion_copy_contract():
    uncertain = Uncertainty(np.array([1.0, 2.0]), 0.1)
    with pytest.warns(DowncastWarning):
        shared = uncertain.__array__(copy=False)
    assert np.shares_memory(shared, uncertain.value)
    with pytest.warns(DowncastWarning):
        copied = uncertain.__array__(copy=True)
    assert not np.shares_memory(copied, uncertain.value)
    with pytest.warns(DowncastWarning):
        converted = uncertain.__array__(dtype=np.float32)
    assert converted.dtype == np.float32
    with pytest.warns(DowncastWarning), pytest.raises(ValueError, match="copy"):
        uncertain.__array__(dtype=np.float32, copy=False)
    with pytest.warns(DowncastWarning), pytest.raises(ValueError, match="copy"):
        Uncertainty(1).__array__(copy=False)


@pytest.mark.parametrize("policy", ("zero", "propagate"))
def test_fill_normalizes_missing_values(policy):
    configure(nonfinite_error_policy=policy)
    uncertain = Uncertainty([1, 2], [0.1, 0.2])
    uncertain.fill(np.nan)
    np.testing.assert_equal(uncertain.value, [np.nan, np.nan])
    expected_error = [np.nan, np.nan] if policy == "propagate" else [0.1, 0.2]
    np.testing.assert_equal(uncertain.error, expected_error)


def test_fill_uncertain_scalar_updates_both_components():
    uncertain = Uncertainty([1, 2], [0.1, 0.2])
    uncertain.fill(Uncertainty(3, 0.5))
    np.testing.assert_equal(uncertain.value, [3, 3])
    np.testing.assert_equal(uncertain.error, [0.5, 0.5])


def test_put_normalizes_values_created_under_previous_policy():
    missing = Uncertainty(np.nan, 0.1)
    configure(nonfinite_error_policy="propagate")
    uncertain = Uncertainty([1, 2], [0.1, 0.2])
    uncertain.put(0, missing)
    np.testing.assert_equal(uncertain.value, [np.nan, 2])
    np.testing.assert_equal(uncertain.error, [np.nan, 0.2])


def test_failed_put_preserves_both_components():
    uncertain = Uncertainty([1, 2], [0.1, 0.2])
    with pytest.raises(IndexError):
        uncertain.put([0, 3], Uncertainty(9, 0.5))
    np.testing.assert_equal(uncertain.value, [1, 2])
    np.testing.assert_equal(uncertain.error, [0.1, 0.2])
