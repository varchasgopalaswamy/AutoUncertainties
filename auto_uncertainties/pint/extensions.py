"""
This module contains classes for use with `pint` registries.

For usage with Pint, see `here <https://pint.readthedocs.io/en/stable/advanced/custom-registry-class.html>`_.
"""

from __future__ import annotations

from typing import TypeAlias

from auto_uncertainties import Uncertainty

try:
    import pint
except ImportError:
    pint = None
    print("Failed to load Pint extensions (Pint is not currently installed).")
    print("Run 'pip install pint' to install it.")


class UncertaintyQuantity(pint.UnitRegistry.Quantity):
    @property
    def value(self):
        if isinstance(self._magnitude, Uncertainty):
            return self._magnitude.value * self.units
        else:
            return self._magnitude * self.units

    @property
    def error(self):
        if isinstance(self._magnitude, Uncertainty):
            return self._magnitude.error * self.units
        else:
            return (0 * self._magnitude) * self.units

    def plus_minus(self, err):
        from auto_uncertainties import nominal_values, std_devs

        my_value = nominal_values(self._magnitude) * self.units
        my_err = std_devs(self._magnitude) * self.units

        new_err = (my_err**2 + err**2) ** 0.5

        return Uncertainty.from_quantities(my_value, new_err)


class UncertaintyUnit(pint.UnitRegistry.Unit):
    pass


class UncertaintyRegistry(
    pint.registry.GenericUnitRegistry[UncertaintyQuantity, pint.Unit]
):
    Quantity: TypeAlias = UncertaintyQuantity
    Unit: TypeAlias = UncertaintyUnit
