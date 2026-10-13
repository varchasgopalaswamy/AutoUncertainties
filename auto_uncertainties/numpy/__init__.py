"""NumPy dispatch with uncertainty propagation."""

__public__ = ("numpy_wrappers",)

# <geninit>
lazy from . import numpy_wrappers
lazy from .numpy_wrappers import HANDLED_FUNCTIONS
lazy from .numpy_wrappers import HANDLED_UFUNCS
lazy from .numpy_wrappers import wrap_numpy

__all__ = (
    "HANDLED_FUNCTIONS",
    "HANDLED_UFUNCS",
    "numpy_wrappers",
    "wrap_numpy",
)
# </geninit>
