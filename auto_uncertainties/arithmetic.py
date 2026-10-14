"""Analytic propagation shared by operators and NumPy arithmetic."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from auto_uncertainties.propagation import precision_context, weighted_derivative

__all__ = ()


def binary(
    operation: np.ufunc,
    left: npt.ArrayLike,
    left_error: npt.ArrayLike,
    right: npt.ArrayLike,
    right_error: npt.ArrayLike,
    **options: Any,
) -> tuple[Any, Any]:
    """
    Evaluate smooth binary arithmetic and combine independent errors.

    :param operation: NumPy addition, subtraction, multiplication, division, or power.
    :param left: Nominal left operand.
    :param left_error: Left operand standard deviation.
    :param right: Nominal right operand.
    :param right_error: Right operand standard deviation.
    :param options: NumPy ufunc options, after output and mask handling.
    :return: The nominal result and propagated standard deviation.
    :raises TypeError: Promotion is not floating point or a signature override is used.
    :raises ValueError: The operation is unsupported.
    """
    if "signature" in options or "sig" in options:
        msg = "Uncertainty arithmetic does not support ufunc signature overrides"
        raise TypeError(msg)
    left = np.asarray(left) if isinstance(left, list | tuple) else left
    right = np.asarray(right) if isinstance(right, list | tuple) else right
    with precision_context(left, left_error, right, right_error, options):
        dtype = jnp.result_type(left, right)
        if options.get("dtype") is not None:
            dtype = jnp.result_type(np.dtype(options["dtype"]))
        if not jnp.issubdtype(dtype, jnp.floating):
            msg = f"Uncertainty propagation requires floating-point promotion, got {dtype}"
            raise TypeError(msg)
        nominal = operation(left, right, **options)
    # Derivatives must use the operands rounded to the actual NumPy computation dtype.
    left, right = np.broadcast_arrays(
        np.asarray(left, dtype=nominal.dtype), np.asarray(right, dtype=nominal.dtype)
    )
    error_dtype = np.result_type(nominal, left_error, right_error)
    with np.errstate(all="ignore"):
        if operation is np.add or operation is np.subtract:
            left_derivative, right_derivative = 1, 1
        elif operation is np.multiply:
            left_derivative, right_derivative = right, left
        elif operation is np.divide:
            left_derivative, right_derivative = 1 / right, -nominal / right
        elif operation is np.power:
            left_derivative = np.where(right == 0, 0, right * np.power(left, right - 1))
            right_derivative = np.where(
                (left == 0) & (right > 0), 0, nominal * np.log(np.abs(left))
            )
        else:
            msg = f"Unsupported arithmetic operation: {operation.__name__}"
            raise ValueError(msg)
        error = np.hypot(
            weighted_derivative(
                np.asarray(left_derivative, dtype=error_dtype),
                np.asarray(left_error, dtype=error_dtype),
            ),
            weighted_derivative(
                np.asarray(right_derivative, dtype=error_dtype),
                np.asarray(right_error, dtype=error_dtype),
            ),
        )
    error = np.broadcast_to(error, np.shape(nominal)).copy()
    return nominal, error[()] if error.ndim == 0 else error
