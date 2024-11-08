"""
This module contains classes for use with `pint` registries.

For usage with Pint, see `here <https://pint.readthedocs.io/en/stable/advanced/custom-registry-class.html>`_.

.. important::

   The `pint` package must be installed in order to use these extensions.

"""

from __future__ import annotations

from auto_uncertainties import Uncertainty

try:
    import pint
except ImportError as e:
    msg = "Failed to load Pint extensions (Pint is not currently installed). Run 'pip install pint' to install it."
    raise ImportError(msg) from e


__all__ = ["UncertaintyQuantity", "UncertaintyUnit", "UncertaintyRegistry"]


class UncertaintyQuantity(pint.UnitRegistry.Quantity):
    """
    Extension of `pint.Quantity` to allow the `value` and `error` attributes for an
    `Uncertainty` to be returned with their proper units.
    """

    @property
    def value(self):
        """
        The central value of the `Uncertainty` object.

        .. seealso:: * `auto_uncertainties.uncertainty.uncertainty_containers.Uncertainty.value`
        """
        if isinstance(self._magnitude, Uncertainty):
            return self._magnitude.value * self.units
        else:
            return self._magnitude * self.units

    @property
    def error(self):
        """
        The uncertainty (error) value of the `Uncertainty` object.

        .. seealso:: * `auto_uncertainties.uncertainty.uncertainty_containers.Uncertainty.error`
        """
        if isinstance(self._magnitude, Uncertainty):
            return self._magnitude.error * self.units
        else:
            return (0 * self._magnitude) * self.units

    def plus_minus(self, err):
        """
        Add an error to the `Uncertainty` object.

        .. seealso:: * `auto_uncertainties.uncertainty.uncertainty_containers.Uncertainty.plus_minus`
        """
        from auto_uncertainties import nominal_values, std_devs

        my_value = nominal_values(self._magnitude) * self.units
        my_err = std_devs(self._magnitude) * self.units

        new_err = (my_err**2 + err**2) ** 0.5

        return Uncertainty.from_quantities(my_value, new_err)


class UncertaintyUnit(pint.UnitRegistry.Unit):
    """
    Unit used for `UncertaintyRegistry`.
    """


class UncertaintyRegistry(
    pint.registry.GenericUnitRegistry[UncertaintyQuantity, pint.Unit]
):
    """
    Alternative `pint` unit registry where the default `~pint.Quantity` class is
    `UncertaintyQuantity`.
    """

    Quantity = UncertaintyQuantity
    Unit = UncertaintyUnit
