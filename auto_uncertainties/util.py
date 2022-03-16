# -*- coding: utf-8 -*-
import warnings
from functools import wraps
from . import NumpyDowncastWarning


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
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore", category=RuntimeWarning)
            response = f(*args, **kwargs)
        return response

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
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore", category=NumpyDowncastWarning)
            response = f(*args, **kwargs)
        return response

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


def is_np_duck_array(cls):
    """Check if object is a numpy array-like, but not a Uncertainty

    Parameters
    ----------
    cls : class

    Returns
    -------
    bool
    """
    try:
        import numpy as np
    except ImportError:
        return False

    return issubclass(cls, np.ndarray) or (
        not hasattr(cls, "_nom")
        and not hasattr(cls, "_err")
        and hasattr(cls, "__array_function__")
        and hasattr(cls, "ndim")
        and hasattr(cls, "dtype")
    )


class Display(object):
    default_format: str = ""

    def __str__(self) -> str:
        if self._nom is not None:
            if self._err is not None:
                return f"{self._nom} +/- {self._err}"
            else:
                return f"{self._nom}"

    def __format__(self, fmt):
        return f"{self.value:{fmt}} +/- {self.error:{fmt}}"

    def __repr__(self) -> str:
        return str(self)
