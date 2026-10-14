"""Optional quantity conversion without importing Pint for ordinary inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ()


def split_units(value: Any, error: Any) -> tuple[Any, Any, Any]:
    """
    Extract compatible quantity magnitudes and the units to restore.

    :param value: Central values, optionally quantities or a sequence of quantities.
    :param error: Standard deviations in compatible units, or plain magnitudes.
    :return: Nominal magnitudes, error magnitudes, and units or ``None``.
    :raises pint.errors.DimensionalityError: Quantities have incompatible units.
    """
    value_units = getattr(value, "units", None)
    error_units = getattr(error, "units", None)
    if value_units is not None:
        return value.m, error.to(value_units).m if error_units is not None else error, value_units
    if error_units is not None:
        # Preserve Pint's dimensionality validation for a bare value with a quantity error.
        nominal = error_units._REGISTRY.Quantity(value).to(error_units).m
        return nominal, error.m, error_units
    if isinstance(value, Sequence) and len(value):
        value_units = getattr(value[0], "units", None)
    if isinstance(error, Sequence) and len(error):
        error_units = getattr(error[0], "units", None)
    units = value_units if value_units is not None else error_units
    if units is not None:
        value = magnitudes(value, units)
        error = magnitudes(error, units)
    return value, error, units


def magnitudes(value: Any, units: Any) -> Any:
    """
    Convert quantity elements of a sequence to common units.

    :param value: A sequence or unchanged scalar input.
    :param units: Units used for quantity magnitudes.
    :return: Converted magnitudes, preserving exact elements.
    """
    if isinstance(value, Sequence):
        return [item.to(units).m if hasattr(item, "units") else item for item in value]
    return value
