from __future__ import annotations

from functools import wraps
from typing import Any, ParamSpec, TypeGuard, TypeVar
import warnings
lazy from collections.abc import Callable, Iterable

from .exceptions import DowncastWarning

__all__ = ()

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
