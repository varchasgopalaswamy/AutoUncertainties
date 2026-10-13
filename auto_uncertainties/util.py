from __future__ import annotations

from functools import wraps
from typing import Any, ParamSpec, TypeGuard, TypeVar
import warnings
lazy from collections.abc import Callable, Iterable

from jax import Array
import numpy as np
from numpy.typing import NDArray

from .exceptions import DowncastWarning

P = ParamSpec("P")
R = TypeVar("R")


def ignore_runtime_warnings(f: Callable[P, R]) -> Callable[P, R]:
    """
    Suppress runtime warnings while calling a decorated function.

    :param f: Function to wrap.
    :return: A wrapper that preserves the function signature and suppresses runtime
        warnings.
    """

    @wraps(f)
    def runtime_warn_inner(*args: P.args, **kwargs: P.kwargs) -> R:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return f(*args, **kwargs)

    return runtime_warn_inner


def ignore_numpy_downcast_warnings(f: Callable[P, R]) -> Callable[P, R]:
    """
    Suppress :class:`~auto_uncertainties.exceptions.DowncastWarning` during a call.

    :param f: Function to wrap.
    :return: A wrapper that preserves the function signature and suppresses downcast
        warnings.
    """

    @wraps(f)
    def user_warn_inner(*args: P.args, **kwargs: P.kwargs) -> R:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DowncastWarning)
            return f(*args, **kwargs)

    return user_warn_inner


def deprecated(
    reason: str = "", category: type[Warning] = DeprecationWarning
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Warn when a decorated function is called.

    :param reason: Explanation appended to the deprecation warning.
    :param category: Warning category to emit.
    :return: A decorator that warns before forwarding each call.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        msg = f"{func.__name__} is deprecated" + (f" ({reason})." if reason else ".")

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            warnings.warn(msg, category, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def is_iterable(y: Any) -> TypeGuard[Iterable[Any]]:
    """
    Check whether an object supports iteration.

    :param y: Object to inspect.
    :return: Whether calling ``iter(y)`` succeeds without a ``TypeError``.
    """
    try:
        iter(y)
    except TypeError:
        return False
    return True


def has_length(y: Any) -> bool:
    """
    Check whether an object has a length.

    :param y: Object to inspect.
    :return: Whether calling ``len(y)`` succeeds without a ``TypeError``.
    """
    try:
        len(y)
    except TypeError:
        return False
    return True


def ndarray_to_scalar(value: NDArray[Any]) -> Any:
    """
    Extract the only element of an array.

    :param value: Array containing exactly one element.
    :return: The array element converted to a scalar.
    :raises ValueError: The array does not contain exactly one element.
    """
    return np.ndarray.item(strip_device_array(value))


def strip_device_array(value: Array | NDArray[Any] | float) -> NDArray[Any]:
    """
    Copy values into a NumPy array on the host.

    :param value: Values to copy, potentially stored on a JAX device.
    :return: A NumPy array containing the values.
    """
    return np.array(value)
