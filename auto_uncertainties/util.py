from __future__ import annotations

from functools import wraps
from typing import TypeVar
import warnings

from jax import Array
import numpy as np
from numpy.typing import NDArray

from . import DowncastWarning

T = TypeVar("T", bound=np.generic, covariant=True)


def ignore_runtime_warnings(f):
    """
    A decorator to ignore runtime warnings
    Parameters
    ----------
    f: function
        The wrapped function

    Returns
    -------
    wrapped_function: function
        The wrapped function
    """

    @wraps(f)
    def runtime_warn_inner(*args, **kwargs):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return f(*args, **kwargs)

    return runtime_warn_inner


def ignore_numpy_downcast_warnings(f):
    """
    A decorator to ignore NumpyDowncastWarning
    Parameters
    ----------
    f: function
        The wrapped function

    Returns
    -------
    wrapped_function: function
        The wrapped function
    """

    @wraps(f)
    def user_warn_inner(*args, **kwargs):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore", category=DowncastWarning)
            return f(*args, **kwargs)

    return user_warn_inner


def is_iterable(y):
    try:
        iter(y)
    except TypeError:
        return False
    return True


def has_length(y):
    try:
        len(y)
    except TypeError:
        return False
    return True


def ndarray_to_scalar(value: NDArray[T]) -> T:
    return np.ndarray.item(strip_device_array(value))


def strip_device_array(value: Array | NDArray | float) -> NDArray:
    return np.array(value)
