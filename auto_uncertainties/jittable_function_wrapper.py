from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from auto_uncertainties import Uncertainty
from auto_uncertainties.numpy.numpy_wrappers import (
    classify_and_split_args_and_kwargs,
    ndarray_to_scalar,
)

P = ParamSpec("P")
R = TypeVar("R")


def elementwise_value_and_grad(g):
    def wrapped(*args, **kwargs):
        y, g_vjp = jax.vjp(lambda *a: g(*a, **kwargs), *args)
        return np.array(y), g_vjp(jnp.ones_like(y))

    return wrapped


def propagate_uncertainties(
    func: Callable[P, R], implement_mode: str = "same_shape"
) -> Callable[P, Uncertainty]:
    """
    A decorator to propagate uncertainties through a given function using
    first-order Taylor expansion.

    Parameters
    ----------
    func : callable
        The function through which to propagate uncertainties. This function should
        be compatible with JAX for automatic differentiation.

    Returns
    -------
    callable
        A new function that propagates uncertainties when called with `Uncertainty`
        objects.

    Notes
    -----
    This decorator uses JAX's automatic differentiation to compute gradients,
    which are then used to propagate uncertainties. The function being decorated
    should be compatible with JAX, meaning it should use JAX-compatible operations.

    Examples
    --------
    >>> import numpy as np
    >>> import jax.numpy as jnp
    >>> from auto_uncertainties import Uncertainty, propagate_uncertainties
    >>> @propagate_uncertainties
    ... def my_function(x, y):
    ...     return x * jnp.sin(y)
    >>> x = Uncertainty(2.0, 0.1)  # Value 2.0 with uncertainty 0.1
    >>> y = Uncertainty(np.pi / 4, 0.01)  # Value pi/4 with uncertainty 0.01
    >>> result = my_function(x, y)
    """

    grad_and_val = elementwise_value_and_grad(func)

    @wraps(func)
    def wrapper(*args, **kwargs) -> Uncertainty:
        (
            uncert_argnums,
            uncert_arg_nom,
            uncert_arg_err,
            uncert_kwarg_nom,
        ) = classify_and_split_args_and_kwargs(*args, **kwargs)
        # Determine result through base numpy function on stripped arguments
        if implement_mode == "same_shape":
            bcast_args_nom = np.broadcast_arrays(*uncert_arg_nom)
            bcast_args_err = np.broadcast_arrays(*uncert_arg_err)
            value, grads = grad_and_val(*bcast_args_nom, **uncert_kwarg_nom)
            error_dot_grad_sqr = [
                (e * g) ** 2 for e, g in zip(bcast_args_err, grads, strict=False)
            ]
            error = np.sum(error_dot_grad_sqr, axis=0) ** 0.5
        elif (
            implement_mode == "same_shape_bool"
            or implement_mode == "nograd"
            or implement_mode == "selection_operator"
        ):
            value = func(*uncert_arg_nom, **uncert_kwarg_nom)
            error = 0
        elif implement_mode in ["reduction_binary", "reduction_unary"]:
            axis = uncert_kwarg_nom.get("axis", None)

            bcast_args_nom = np.broadcast_arrays(*uncert_arg_nom)
            bcast_args_err = np.broadcast_arrays(*uncert_arg_err)
            value, grads = grad_and_val(*bcast_args_nom, **uncert_kwarg_nom)
            if axis is not None:
                axis = tuple(axis)
                error_dot_grad_sqr = [
                    np.sum((e * g) ** 2, axis=axis)
                    for e, g in zip(bcast_args_err, grads, strict=False)
                ]
            else:
                error_dot_grad_sqr = [
                    np.sum((e * g) ** 2)
                    for e, g in zip(bcast_args_err, grads, strict=False)
                ]
            value = ndarray_to_scalar(value)
            error = ndarray_to_scalar(np.sum(error_dot_grad_sqr, axis=0) ** 0.5)
        else:
            msg = f"Invalid implement_mode {implement_mode}"
            raise ValueError(msg)

        return Uncertainty((value), (error))

    return wrapper
