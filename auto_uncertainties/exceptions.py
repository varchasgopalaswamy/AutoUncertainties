from __future__ import annotations

__all__ = [
    "DowncastError",
    "DowncastWarning",
    "EqualityError",
    "EqualityWarning",
    "NegativeStdDevError",
]


class NegativeStdDevError(Exception):
    """An exception for when the standard deviation is negative."""


class DowncastError(RuntimeError):
    """
    An exception for when an array of `~auto_uncertainties.uncertainty.uncertainty_containers.Uncertainty`
    objects is downcast to a NumPy `~numpy.ndarray`.
    """


class DowncastWarning(RuntimeWarning):
    """
    A warning for when an array of `~auto_uncertainties.uncertainty.uncertainty_containers.Uncertainty`
    objects is downcast to a NumPy `~numpy.ndarray`.
    """


class EqualityWarning(RuntimeWarning):
    """
    A warning that is raised when the equality check is performed on two
    `~auto_uncertainties.uncertainty.uncertainty_containers.Uncertainty` objects with identical
    central values, but different standard deviations.
    """


class EqualityError(RuntimeError):
    """
    An exception that is raised when the equality check is performed on two
    `~auto_uncertainties.uncertainty.uncertainty_containers.Uncertainty` objects with identical
    central values, but different standard deviations.
    """
