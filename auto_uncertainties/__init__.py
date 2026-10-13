"""Automatic first-order uncertainty propagation."""

__public__ = (
    "config",
    "display_format",
    "exceptions",
    "jittable_function_wrapper",
    "uncertainty",
)
__private__ = ("_propagation", "util")
__protected__ = ("numpy",)

# <geninit>
lazy from . import config
lazy from . import display_format
lazy from . import exceptions
lazy from . import jittable_function_wrapper
lazy from . import numpy
lazy from . import uncertainty
lazy from .config import Settings
lazy from .config import configure
lazy from .config import get_settings
lazy from .display_format import ScalarDisplay
lazy from .display_format import UncertaintyDisplay
lazy from .display_format import VectorDisplay
lazy from .exceptions import DowncastError
lazy from .exceptions import DowncastWarning
lazy from .exceptions import EqualityError
lazy from .exceptions import EqualityWarning
lazy from .exceptions import NegativeStdDevError
lazy from .jittable_function_wrapper import P
lazy from .jittable_function_wrapper import R
lazy from .jittable_function_wrapper import elementwise_value_and_grad
lazy from .jittable_function_wrapper import propagate_uncertainties
lazy from .uncertainty import ScalarUncertainty
lazy from .uncertainty import UType
lazy from .uncertainty import Uncertainty
lazy from .uncertainty import VectorUncertainty
lazy from .uncertainty import nominal_values
lazy from .uncertainty import std_devs
lazy from .uncertainty import uncertainty_containers

__all__ = (
    "DowncastError",
    "DowncastWarning",
    "EqualityError",
    "EqualityWarning",
    "NegativeStdDevError",
    "P",
    "R",
    "ScalarDisplay",
    "ScalarUncertainty",
    "Settings",
    "UType",
    "Uncertainty",
    "UncertaintyDisplay",
    "VectorDisplay",
    "VectorUncertainty",
    "config",
    "configure",
    "display_format",
    "elementwise_value_and_grad",
    "exceptions",
    "get_settings",
    "jittable_function_wrapper",
    "nominal_values",
    "numpy",
    "propagate_uncertainties",
    "std_devs",
    "uncertainty",
    "uncertainty_containers",
)
# </geninit>
