from __future__ import annotations

__all__ = ["NegativeStdDevError", "DowncastError", "DowncastWarning"]


class NegativeStdDevError(Exception):
    """An exception for when the standard deviation is negative."""


class DowncastError(RuntimeError):
    """
    An exception for when an array of `~auto_uncertainties.uncertainty.uncertainty_containers.Uncertainty`
    objects is downcast to a NumPy `~numpy.ndarray`.
    """


class DowncastWarning(RuntimeWarning):
    """
    An exception for when an array of `~auto_uncertainties.uncertainty.uncertainty_containers.Uncertainty`
    objects is downcast to a NumPy `~numpy.ndarray`.
    """
