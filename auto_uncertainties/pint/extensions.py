"""
This module contains classes for use with `pint` registries.

For usage with Pint, see `here <https://pint.readthedocs.io/en/stable/advanced/custom-registry-class.html>`_.

.. important::

   The `pint` package must be installed in order to use these extensions.

.. warning::

   To use `Uncertainty` objects with `pint` quantities, always make sure to use
   the `UncertaintyQuantity` extension. Simply passing an `Uncertainty` object
   into `Quantity()` will *not* automatically use `UncertaintyQuantity`, and may cause
   problems.

   A small exception to this is when the `UncertaintyRegistry` class is used. In that case,
   it is supported to pass `Uncertainty` objects directly into `Quantity()`.

   See the examples below for clarification.

.. code-block:: python
   :caption: Supported use of UncertaintyQuantity

   >>> from auto_uncertainties import Uncertainty
   >>> from auto_uncertainties.pint import UncertaintyQuantity

   # Units will be preserved when accessing the 'value' and 'error' attributes
   >>> x = Uncertainty(1.0, 0.5)
   >>> q = UncertaintyQuantity(x, 'radian')
   >>> q.value
   <Quantity(1.0, 'radian')>
   >>> q.error
   <Quantity(0.5, 'radian')>

.. code-block:: python
   :caption: Unsupported use example

   >>> from auto_uncertainties import Uncertainty
   >>> from pint import Quantity

   # Units are NOT automatically preserved when accessing the 'value' and 'error' attributes
   >>> x = Uncertainty(1.0, 0.5)
   >>> q = Quantity(x, 'radian')
   >>> q.value
   1.0
   >>> q.error
   0.5

.. code-block:: python
   :caption: Supported use with the custom unit registry

   >>> from auto_uncertainties import Uncertainty
   >>> from auto_uncertainties.pint import UncertaintyQuantity, UncertaintyRegistry

   # UncertaintyRegistry overrides the default Pint Quantity class
   >>> reg = UncertaintyRegistry()
   >>> x = Uncertainty(1.0, 0.5)
   >>> q = reg.Quantity(x, 'radian')
   >>> type(q)
   <class 'pint.UncertaintyQuantity'>
   >>> q.value
   <Quantity(1.0, 'radian')>
   >>> q.error
   <Quantity(0.5, 'radian')>

"""

from __future__ import annotations

from typing import Generic

from auto_uncertainties import Uncertainty

try:
    import pint
    from pint.facets.plain import MagnitudeT, PlainQuantity
except ImportError as e:
    msg = "Failed to load Pint extensions (Pint is not currently installed). Run 'pip install pint' to install it."
    raise ImportError(msg) from e


__all__ = ["UncertaintyQuantity", "UncertaintyUnit", "UncertaintyRegistry"]


class UncertaintyQuantity(Generic[MagnitudeT], PlainQuantity[MagnitudeT]):
    """
    Extension of `pint.facets.plain.PlainQuantity` to allow the `value` and `error`
    attributes for an `Uncertainty` to be returned with their proper units.
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
