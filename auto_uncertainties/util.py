# -*- coding: utf-8 -*-


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
