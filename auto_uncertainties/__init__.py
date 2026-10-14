"""Automatic first-order uncertainty propagation."""

__flattened__ = (
    "config",
    "exceptions",
    "jittable_function_wrapper",
    "uncertainty",
)
__private__ = ("arithmetic", "display_format", "numpy", "propagation", "units", "util")

# <geninit>
lazy from .config import Settings
lazy from .config import configure
lazy from .config import get_settings
lazy from .exceptions import DowncastError
lazy from .exceptions import DowncastWarning
lazy from .exceptions import EqualityError
lazy from .exceptions import EqualityWarning
lazy from .exceptions import NegativeStdDevError
lazy from .jittable_function_wrapper import propagate_uncertainties
lazy from .uncertainty import UType
lazy from .uncertainty import Uncertainty
lazy from .uncertainty import nominal_values
lazy from .uncertainty import std_devs

__all__ = (
    "DowncastError",
    "DowncastWarning",
    "EqualityError",
    "EqualityWarning",
    "NegativeStdDevError",
    "Settings",
    "UType",
    "Uncertainty",
    "configure",
    "get_settings",
    "nominal_values",
    "propagate_uncertainties",
    "std_devs",
)
# </geninit>
