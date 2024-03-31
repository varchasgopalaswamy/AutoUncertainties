# -*- coding: utf-8 -*-
from __future__ import annotations

__all__ = ["NegativeStdDevError", "NumpyDowncastWarning"]

__all__ = ["NegativeStdDevError", "NumpyDowncastError", "NumpyDowncastWarning"]


class NegativeStdDevError(Exception):
    """An exception for when the standard deviation is negative"""

    pass


class NumpyDowncastError(RuntimeError):
    """An exception for when an uncertainties array is downcast to a numpy array"""

    pass


class NumpyDowncastWarning(RuntimeWarning):
    """An exception for when an uncertainties array is downcast to a numpy array"""

    pass
