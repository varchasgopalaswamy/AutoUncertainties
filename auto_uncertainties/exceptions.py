from __future__ import annotations

__all__ = ["NegativeStdDevError", "DowncastError", "DowncastWarning"]


class NegativeStdDevError(Exception):
    """An exception for when the standard deviation is negative"""


class DowncastError(RuntimeError):
    """An exception for when an uncertainties array is downcast to a numpy array"""


class DowncastWarning(RuntimeWarning):
    """An exception for when an uncertainties array is downcast to a numpy array"""
