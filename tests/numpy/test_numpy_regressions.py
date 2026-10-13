"""Analytic checks for uncertainty propagation and NumPy calling conventions."""

from __future__ import annotations

import numpy as np
import pytest

from auto_uncertainties import Uncertainty


@pytest.fixture
def matrix():
    values = np.arange(1.0, 7.0).reshape(2, 3)
    return Uncertainty(values, values / 10)


def assert_uncertainty(actual, value, error):
    assert isinstance(actual, Uncertainty)
    assert np.shape(actual.value) == np.shape(value)
    assert np.shape(actual.error) == np.shape(error)
    np.testing.assert_allclose(actual.value, value)
    np.testing.assert_allclose(actual.error, error)


@pytest.mark.parametrize("axis", (None, 0, 1, -1, (0,), (0, 1), ()))
@pytest.mark.parametrize("keepdims", (False, True))
def test_sum_axes_and_keepdims(matrix, axis, keepdims):
    actual = np.sum(matrix, axis=axis, keepdims=keepdims)
    expected_value = np.sum(matrix.value, axis=axis, keepdims=keepdims)
    expected_error = np.sqrt(np.sum(matrix.error**2, axis=axis, keepdims=keepdims))
    assert_uncertainty(actual, expected_value, expected_error)


def test_reduction_positional_metadata_and_uncertain_keyword(matrix):
    expected = np.sum(matrix.value, axis=0)
    error = np.sqrt(np.sum(matrix.error**2, axis=0))
    assert_uncertainty(np.sum(matrix, 0, np.float64), expected, error)
    assert_uncertainty(np.sum(a=matrix, axis=0), expected, error)


def test_mean_where_and_initial_sum(matrix):
    mask = np.array([[True, False, True], [False, True, True]])
    error = np.sqrt(np.sum(np.where(mask, matrix.error**2, 0), axis=1))
    mean = np.mean(matrix, axis=1, where=mask)
    assert_uncertainty(mean, [2.0, 5.5], error / 2)
    total = np.sum(matrix, axis=1, initial=10, where=mask)
    assert_uncertainty(total, [14.0, 21.0], error)


@pytest.mark.parametrize("operation", (np.var, np.std))
def test_variance_and_standard_deviation_errors(matrix, operation):
    centered = matrix.value - matrix.value.mean(axis=1, keepdims=True)
    gradient = 2 * centered / 2  # Three samples with ddof=1.
    if operation is np.std:
        gradient /= 2 * matrix.value.std(axis=1, ddof=1, keepdims=True)
    expected_error = np.sqrt(np.sum((gradient * matrix.error) ** 2, axis=1))
    assert_uncertainty(
        operation(matrix, axis=1, ddof=1),
        operation(matrix.value, axis=1, ddof=1),
        expected_error,
    )


@pytest.mark.parametrize("constant", (2, np.array([2, 3, 4]), [2, 3, 4]))
def test_integer_operands_and_broadcasting(matrix, constant):
    assert_uncertainty(np.add(matrix, constant), matrix.value + constant, matrix.error)
    assert_uncertainty(
        np.multiply(matrix, constant), matrix.value * constant, matrix.error * constant
    )


def test_broadcast_two_uncertain_operands():
    left = Uncertainty(np.array([[2.0], [3.0]]), np.array([[0.2], [0.3]]))
    right = Uncertainty(np.array([4.0, 5.0, 6.0]), np.array([0.4, 0.5, 0.6]))
    value = left.value * right.value
    error = np.hypot(left.error * right.value, left.value * right.error)
    assert_uncertainty(np.multiply(left, right), value, error)


@pytest.mark.parametrize(
    ("operation", "args", "kwargs"),
    (
        (np.reshape, ((3, 2),), {}),
        (np.repeat, (2,), {"axis": 0}),
        (np.take, (np.array([2, 0]),), {"axis": 1}),
        (np.transpose, ((1, 0),), {}),
        (np.swapaxes, (0, 1), {}),
        (np.diagonal, (1,), {}),
        (np.ravel, ("F",), {}),
    ),
)
def test_structural_positional_arguments(matrix, operation, args, kwargs):
    assert_uncertainty(
        operation(matrix, *args, **kwargs),
        operation(matrix.value, *args, **kwargs),
        operation(matrix.error, *args, **kwargs),
    )


def test_compress_preserves_condition_and_keyword_data(matrix):
    mask = [False, True, True]
    for result in [np.compress(mask, matrix, 1), np.compress(mask, a=matrix, axis=1)]:
        assert_uncertainty(result, matrix.value[:, 1:], matrix.error[:, 1:])


@pytest.mark.parametrize("operation", (np.max, np.min))
@pytest.mark.parametrize("axis", (None, 0, 1, (0, 1), ()))
@pytest.mark.parametrize("keepdims", (False, True))
def test_extreme_shapes_and_selected_errors(matrix, operation, axis, keepdims):
    value = operation(matrix.value, axis=axis, keepdims=keepdims)
    assert_uncertainty(operation(matrix, axis=axis, keepdims=keepdims), value, value / 10)


def test_extreme_where_initial_and_empty_axes(matrix):
    mask = np.array([[True, False, True], [False, True, False]])
    assert_uncertainty(np.max(matrix, 0, where=mask, initial=4), [4, 5, 4], [0, 0.5, 0])
    empty = Uncertainty(np.empty((0, 3)), np.empty((0, 3)))
    assert_uncertainty(np.max(empty, axis=0, initial=4), [4, 4, 4], [0, 0, 0])
    assert_uncertainty(np.max(matrix, initial=None), 6.0, 0.6)


def test_unique_metadata_and_multidimensional_values(matrix):
    source = Uncertainty(np.array([[2.0, 1.0], [2.0, 3.0]]), np.array([[0.2, 0.1], [0.4, 0.3]]))
    result, indices, inverse, counts = np.unique(
        source, return_index=True, return_inverse=True, return_counts=True
    )
    expected = np.unique(source.value, return_index=True, return_inverse=True, return_counts=True)
    assert_uncertainty(result, expected[0], source.error.ravel()[expected[1]])
    for actual, reference in zip((indices, inverse, counts), expected[1:], strict=True):
        np.testing.assert_array_equal(actual, reference)
    axis_result, axis_counts = np.unique(matrix, axis=1, return_counts=True)
    assert_uncertainty(axis_result, matrix.value, matrix.error)
    np.testing.assert_array_equal(axis_counts, [1, 1, 1])


def test_sort_and_take_along_axis_preserve_singleton_axes():
    source = Uncertainty(np.array([[3.0, 1.0, 2.0]]), np.array([[0.3, 0.1, 0.2]]))
    assert_uncertainty(np.sort(source, kind="stable"), [[1.0, 2.0, 3.0]], [[0.1, 0.2, 0.3]])
    assert_uncertainty(np.sort(source, axis=None), [1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    assert_uncertainty(np.take_along_axis(source, np.array([[1]]), axis=1), [[1.0]], [[0.1]])


def test_concatenate_positional_axis_and_exact_arrays(matrix):
    exact = np.full((2, 1), 10)
    expected_value = np.concatenate([matrix.value, exact], 1)
    expected_error = np.concatenate([matrix.error, np.zeros((2, 1))], 1)
    assert_uncertainty(np.concatenate([matrix, exact], 1), expected_value, expected_error)
    assert_uncertainty(
        np.concatenate([matrix, exact], axis=1, dtype=np.float32),
        expected_value.astype(np.float32),
        expected_error,
    )
    out = Uncertainty(np.zeros((2, 4)), np.zeros((2, 4)))
    with pytest.raises(TypeError, match="both out and dtype"):
        np.concatenate([matrix, exact], axis=1, out=out, dtype=np.float32)
    assert_uncertainty(out, np.zeros((2, 4)), np.zeros((2, 4)))


@pytest.mark.parametrize("decimals", (-2, 0, 2, 5))
def test_round_keeps_uncertainty_and_dimensions(decimals):
    source = Uncertainty(np.array([[1.2345, 7.8912]]), np.array([[0.001, 0.03]]))
    assert_uncertainty(np.round(source, decimals), np.round(source.value, decimals), source.error)


def test_ufunc_out_updates_both_buffers_and_honors_where():
    source = Uncertainty([1.0, 2.0], [0.1, 0.2])
    out = Uncertainty([10.0, 20.0], [1.0, 2.0])
    returned = np.multiply(source, 3, out=out, where=[True, False], dtype=np.float64)
    assert returned is out
    assert_uncertainty(out, [3.0, 20.0], [0.3, 2.0])
    assert np.add(source, 1, source) is source
    assert_uncertainty(source, [2.0, 3.0], [0.1, 0.2])


def test_function_out_is_positional_and_updates_both_buffers(matrix):
    out = Uncertainty(np.zeros(3), np.zeros(3))
    returned = np.sum(matrix, 0, None, out)
    assert returned is out
    assert_uncertainty(out, [5, 7, 9], np.sqrt(np.sum(matrix.error**2, axis=0)))


def test_invalid_out_does_not_partially_mutate():
    source = Uncertainty([1.0, 2.0], [0.1, 0.2])
    exact_out = np.array([10.0, 20.0])
    with pytest.raises(TypeError, match="Uncertainty out"):
        np.add(source, 1, out=exact_out)
    np.testing.assert_array_equal(exact_out, [10.0, 20.0])
    bad_shape = Uncertainty([10.0], [1.0])
    with pytest.raises(ValueError, match="same shape"):
        np.add(source, 1, out=bad_shape)
    assert_uncertainty(bad_shape, [10.0], [1.0])
    with pytest.raises(ValueError, match="where requires"):
        np.add(source, 1, where=[True, False])


def test_readonly_error_out_does_not_mutate_nominal():
    source = Uncertainty([1.0, 2.0], [0.1, 0.2])
    out = Uncertainty([10.0, 20.0], [1.0, 2.0])
    out.error.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        np.add(source, 1, out=out)
    assert_uncertainty(out, [10.0, 20.0], [1.0, 2.0])


@pytest.mark.parametrize(
    "options",
    (
        {"casting": "invalid"},
        {"order": "invalid"},
        {"dtype": np.int64, "casting": "unsafe"},
    ),
)
def test_invalid_ufunc_options_leave_out_unchanged(options):
    source = Uncertainty([1.0, 2.0], [0.1, 0.2])
    out = Uncertainty([10.0, 20.0], [1.0, 2.0])
    with pytest.raises((TypeError, ValueError)):
        np.multiply(source, 3, out=out, **options)
    assert_uncertainty(out, [10.0, 20.0], [1.0, 2.0])


def test_trapezoid_propagates_both_data_arguments():
    operation = np.trapezoid
    values = Uncertainty([2.0, 4.0, 8.0], [0.2, 0.4, 0.8])
    positions = Uncertainty([0.0, 1.0, 3.0], [0.1, 0.2, 0.3])
    # dI/dy = [0.5, 1.5, 1]; dI/dx = [-3, -3, 6].
    expected_error = np.linalg.norm([0.1, 0.6, 0.8, -0.3, -0.6, 1.8])
    assert_uncertainty(operation(values, positions), 15.0, expected_error)
    assert_uncertainty(operation(y=values, x=positions), 15.0, expected_error)


def test_median_overwrite_input_preserves_value_error_pairing():
    source = Uncertainty([3.0, 1.0, 2.0], [0.3, 0.1, 0.2])
    assert_uncertainty(np.median(source, overwrite_input=True), 2.0, 0.2)
    assert_uncertainty(source, [3.0, 1.0, 2.0], [0.3, 0.1, 0.2])


def test_power_requested_dtype_controls_derivatives():
    source = Uncertainty(np.float32(1e20), np.float32(0.1))
    result = np.power(source, 3, dtype=np.float64)
    assert_uncertainty(
        result,
        np.float64(source.value) ** 3,
        3 * np.float64(source.value) ** 2 * np.float64(source.error),
    )
    rounded = np.power(Uncertainty(1.00000003, 1.0), 10000000.0, dtype=np.float32)
    assert_uncertainty(rounded, np.float32(1.0), np.float32(1e7))


@pytest.mark.parametrize(
    ("operation", "value", "error", "expected_error"),
    (
        (np.exp, 100.0, 0.01, np.exp(100.0) * 0.01),
        (np.sqrt, 1e40, 1e20, 0.5),
        (np.log, 1e30, 1.0, 1e-30),
    ),
)
def test_finite_float64_derivatives_keep_finite_errors(operation, value, error, expected_error):
    result = operation(Uncertainty(value, error))
    assert result.error > 0
    np.testing.assert_allclose(result.error, expected_error, rtol=1e-12, atol=0)
