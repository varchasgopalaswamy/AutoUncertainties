from __future__ import annotations

from functools import wraps
from typing import Any
lazy from collections.abc import Callable

from auto_uncertainties.propagation import propagate

__all__ = ["propagate_uncertainties"]


def propagate_uncertainties(func: Callable[..., Any]) -> Callable[..., Any]:
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

    Use ``@propagate_uncertainties`` or call ``propagate_uncertainties(func)``
    to wrap an existing function.

    :param func: JAX-compatible function to wrap.
    :return: The wrapped function.

    .. rubric:: Examples

    >>> import jax.numpy as jnp
    >>> from auto_uncertainties import Uncertainty, propagate_uncertainties
    >>> @propagate_uncertainties
    ... def centered(x):
    ...     return x - jnp.mean(x)
    >>> result = centered(Uncertainty([1.0, 2.0], [0.1, 0.2]))
    """

    # Uncertainty leaves change the accepted argument types and output tree.
    # Python typing cannot map those replacements through a ParamSpec.
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return propagate(func, args, kwargs)

    return wrapper
