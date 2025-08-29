from __future__ import annotations

__all__ = [
    "DowncastError",
    "DowncastWarning",
    "EqualityError",
    "EqualityWarning",
    "NegativeStdDevError",
    "set_compare_rtol",
    "set_downcast_error",
    "set_equality_error",
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


def set_numpy_kwargs_propagate_error(val: bool) -> None:
    """
    Set whether `EqualityError` should be raised instead of a warning when performing
    an equality check between two `Uncertainty` objects with identical central values,
    but different standard deviations.
    """
    from auto_uncertainties.numpy import numpy_wrappers

    numpy_wrappers.ERROR_ON_KWARGS_PROP = val


def set_equality_error(val: bool) -> None:
    """
    Set whether `EqualityError` should be raised instead of a warning when performing
    an equality check between two `Uncertainty` objects with identical central values,
    but different standard deviations.
    """
    from auto_uncertainties.uncertainty import uncertainty_containers

    uncertainty_containers.ERROR_ON_EQ = val


def set_downcast_error(val: bool) -> None:
    """Set whether `DowncastError` should be raised when uncertainty is stripped."""
    from auto_uncertainties.uncertainty import uncertainty_containers

    uncertainty_containers.ERROR_ON_DOWNCAST = val


def set_compare_rtol(val: float) -> None:
    """
    Set the comparison relative tolerance for error when performing equality
    operations on `Uncertainty` objects.
    """
    from auto_uncertainties.uncertainty import uncertainty_containers

    uncertainty_containers.COMPARE_RTOL = val
