from __future__ import annotations

import operator

import jax
import numpy as np
import pytest

from auto_uncertainties import Uncertainty, configure


@pytest.mark.parametrize(
    ("operation", "ufunc"),
    (
        (operator.add, np.add),
        (operator.sub, np.subtract),
        (operator.mul, np.multiply),
        (operator.truediv, np.divide),
        (operator.pow, np.power),
    ),
)
@pytest.mark.parametrize("dtype", (np.float16, np.float32, np.float64))
@pytest.mark.parametrize("uncertain_right", (False, True))
def test_operator_ufunc_parity(operation, ufunc, dtype, uncertain_right):
    left = Uncertainty(np.array([[2], [3]], dtype=dtype), np.array([[0.2], [0.3]], dtype=dtype))
    right = (
        Uncertainty(np.array([1, 2], dtype=dtype), np.array([0.1, 0.2], dtype=dtype))
        if uncertain_right
        else 2
    )
    for first, second in ((left, right), (right, left)):
        actual, expected = operation(first, second), ufunc(first, second)
        np.testing.assert_array_equal(actual.value, expected.value)
        np.testing.assert_array_equal(actual.error, expected.error)
        assert actual.value.dtype == expected.value.dtype == dtype
        assert actual.error.dtype == expected.error.dtype == dtype


@pytest.mark.parametrize("error", (1e200, 1e-200))
@pytest.mark.parametrize("operation", (operator.add, operator.sub))
def test_quadrature_preserves_large_and_small_errors(error, operation):
    result = operation(Uncertainty(2, error), Uncertainty(3, error))
    np.testing.assert_allclose(result.error, np.sqrt(2) * error, rtol=1e-15, atol=0)


@pytest.mark.parametrize(
    ("operation", "value", "exact", "expected"),
    ((operator.mul, 2, 0, 0), (operator.pow, 2, 0, 1), (operator.pow, 0, 2, 0)),
)
def test_zero_derivatives_ignore_infinite_errors(operation, value, exact, expected):
    configure(nonfinite_error_policy="propagate")
    result = operation(Uncertainty(value, np.inf), exact)
    assert result.value == expected
    assert result.error == 0
    result = 0 / Uncertainty(2, np.inf)
    assert result.value == result.error == 0


def test_division_uses_numpy_zero_behavior():
    configure(nonfinite_error_policy="propagate")
    result = Uncertainty(2, 0.1) / 0
    assert np.isposinf(result.value) and np.isposinf(result.error)


def test_operator_promotion_respects_jax_policy_and_restores_precision():
    source = Uncertainty(np.array([1], dtype=np.float32), np.array([0.1], dtype=np.float32))
    with jax.enable_x64(new_val=False), jax.numpy_dtype_promotion("strict"):
        assert (source + 2).value.dtype == np.float32
        with pytest.raises(jax.dtypes.TypePromotionError):
            source + np.array([2], dtype=np.int32)
        assert not jax.config.x64_enabled


def test_unknown_reflected_operand_returns_not_implemented():
    value = Uncertainty(1, 0.1)
    for method in (value.__rsub__, value.__rtruediv__, value.__rpow__):
        assert method(object()) is NotImplemented
