# -*- coding: utf-8 -*-
import numpy as np


class NegativeStdDevError(Exception):
    """An exception for when the standard deviation is negative"""

    pass


class NumpyDowncastWarning(RuntimeWarning):
    """An exception for when an uncertainties array is downcast to a numpy array"""

    pass


from .uncertainty import Uncertainty

try:
    from .pandas_compat import UncertaintyArray
except ImportError:
    pass


def nominal_values(x):
    # Is an Uncertainty
    if hasattr(x, "_nom"):
        return x.value
    else:
        if np.ndim(x) > 0:
            try:
                x2 = Uncertainty.from_sequence(x)
            except Exception:
                return x
            else:
                return x2.value
        else:
            try:
                x2 = Uncertainty(x)
            except Exception:
                return x
            else:
                return x2.value


def std_devs(x):
    # Is an Uncertainty
    if hasattr(x, "_nom"):
        return x.error
    else:
        if np.ndim(x) > 0:
            try:
                x2 = Uncertainty.from_sequence(x)
            except Exception:
                return np.zeros_like(x)
            else:
                return x2.error
        else:
            try:
                x2 = Uncertainty(x)
            except Exception:
                return 0
            else:
                return x2.error
