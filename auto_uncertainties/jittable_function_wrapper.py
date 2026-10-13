from __future__ import annotations

from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast, overload
lazy from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from auto_uncertainties._propagation import precision_context, propagate

P = ParamSpec("P")
R = TypeVar("R")

__all__ = ["P", "R", "elementwise_value_and_grad", "propagate_uncertainties"]


def elementwise_value_and_grad(
    g: Callable[P, Any],
) -> Callable[P, tuple[npt.NDArray[Any], tuple[npt.NDArray[Any], ...]]]:
    """
    Wrap an elementwise function to return its values and derivatives.

    This compatibility helper differentiates the sum of outputs; use
    :func:`propagate_uncertainties` for functions that mix array elements.

    :param g: JAX-compatible elementwise function whose positional operands are
        differentiated.
    :return: A wrapper returning the nominal result and one gradient per positional
        operand.
    """

    @wraps(g)
    def wrapped(
        *args: P.args, **kwargs: P.kwargs
    ) -> tuple[npt.NDArray[Any], tuple[npt.NDArray[Any], ...]]:
        with precision_context(args, kwargs):
            dtype = jnp.result_type(*args)
            promoted = tuple(jnp.asarray(arg, dtype=dtype) for arg in args)
            value, pullback = jax.vjp(
                lambda *a: cast("Callable[..., Any]", g)(*a, **kwargs), *promoted
            )
            gradients = pullback(jnp.ones_like(value))
            return np.asarray(value), tuple(np.asarray(gradient) for gradient in gradients)

    return wrapped


@overload
def propagate_uncertainties(
    func: Callable[P, R], implement_mode: str | None = None
) -> Callable[..., Any]: ...


@overload
def propagate_uncertainties(
    func: None = None, implement_mode: str | None = None
) -> Callable[[Callable[P, R]], Callable[..., Any]]: ...


def propagate_uncertainties(
    func: Callable[P, R] | None = None, implement_mode: str | None = None
) -> Callable[..., Any] | Callable[[Callable[P, R]], Callable[..., Any]]:
    """
    Decorate a real-valued JAX function with first-order error propagation.

    Inputs may contain ``Uncertainty`` objects in positional arguments, keyword
    arguments, or nested lists, tuples and dictionaries. Exact arguments retain
    their original types and positions. Outputs may be scalars, arrays of any
    shape, or pytrees of these; each numeric output becomes an ``Uncertainty``.
    Boolean outputs remain exact booleans.

    Independent input errors are contracted with the full Jacobian, so array
    permutations, reductions, broadcasting and coupled outputs are supported.
    Each uncertainty occurrence in the input tree is an independent input;
    reuse of a parameter inside the function retains its dependence. Correlations
    are not retained between separately returned uncertainty objects.
    The wrapped function must use JAX
    operations when computing with uncertain inputs. The returned containers
    are NumPy-backed; the wrapper itself is not intended for ``jax.jit``.

    Both ``@propagate_uncertainties`` and ``@propagate_uncertainties()`` work.
    ``implement_mode`` is accepted for compatibility with older callers; all
    modes now use the same general propagation rule.

    :param func: JAX-compatible function to wrap, or ``None`` when configuring the
        decorator first.
    :param implement_mode: Legacy propagation mode. Accepted modes share the same
        general propagation rule.
    :return: The wrapped function, or a decorator when ``func`` is omitted.
    :raises ValueError: The legacy propagation mode is not recognized.

    .. rubric:: Examples

    >>> import jax.numpy as jnp
    >>> from auto_uncertainties import Uncertainty, propagate_uncertainties
    >>> @propagate_uncertainties
    ... def centered(x):
    ...     return x - jnp.mean(x)
    >>> result = centered(Uncertainty([1.0, 2.0], [0.1, 0.2]))
    """
    modes = {
        None,
        "same_shape",
        "same_shape_bool",
        "nograd",
        "selection_operator",
        "reduction_binary",
        "reduction_unary",
    }
    if implement_mode not in modes:
        msg = f"Invalid implement_mode {implement_mode}"
        raise ValueError(msg)

    # Uncertainty leaves change the accepted argument types and output tree.
    # Python typing cannot map those replacements through a ParamSpec.
    def decorate(function: Callable[P, R]) -> Callable[..., Any]:
        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return propagate(function, args, kwargs)

        return wrapper

    return decorate if func is None else decorate(func)
