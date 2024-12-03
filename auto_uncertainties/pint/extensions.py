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

from typing import Generic, TypeVar

from auto_uncertainties import Uncertainty, UType

try:
    from pint._typing import Magnitude, UnitLike
    import pint.facets.context.objects
    import pint.facets.dask
    import pint.facets.measurement.objects
    import pint.facets.nonmultiplicative.objects
    import pint.facets.numpy.quantity
    import pint.facets.plain
    import pint.facets.system.objects
except ImportError as e:
    msg = "Failed to load Pint extensions (Pint is not currently installed). Run 'pip install pint' to install it."
    raise ImportError(msg) from e


__all__ = ["UncertaintyQuantity", "UncertaintyUnit", "UncertaintyRegistry"]


UMagnitudeT = TypeVar("UMagnitudeT", bound=Magnitude | Uncertainty)
"""`TypeVar` extending `~pint.facets.plain.MagnitudeT` to include `Uncertainty`."""


class UncertaintyQuantity(
    Generic[UMagnitudeT],
    pint.facets.system.objects.SystemQuantity[UMagnitudeT],  # type: ignore
    pint.facets.context.objects.ContextQuantity[UMagnitudeT],  # type: ignore
    pint.facets.dask.DaskQuantity[UMagnitudeT],  # type: ignore
    pint.facets.numpy.quantity.NumpyQuantity[UMagnitudeT],  # type: ignore
    # pint.facets.measurement.objects.MeasurementQuantity[UMagnitudeT],  # type: ignore
    pint.facets.nonmultiplicative.objects.NonMultiplicativeQuantity[UMagnitudeT],  # type: ignore
    pint.facets.plain.PlainQuantity[UMagnitudeT],  # type: ignore
):
    """
    Generic extension of `pint.Quantity` to allow properties and methods
    for an `Uncertainty` to be returned with their proper units.
    """

    # TODO: This is still a bit sketchy from a typing perspective...
    def __new__(
        cls, value: Uncertainty[UType], units: UnitLike | None = None
    ) -> UncertaintyQuantity[UMagnitudeT]:
        return super().__new__(cls, value, units)  # type: ignore

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
    # pint.registry.GenericUnitRegistry[UncertaintyQuantity, pint.Unit]
    pint.UnitRegistry
):
    """
    Alternative `pint` unit registry where the default `~pint.Quantity` class is
    `UncertaintyQuantity`.

    TODO: Subclasses of GenericUnitRegistry cannot be used with pint.set_application_registry because
    TODO: they are not subclasses of pint.UnitRegistry. We can directly subclass pint.UnitRegistry,
    TODO: bu it is not generic, and type checkers complain.
    """

    Quantity = UncertaintyQuantity
    Unit = UncertaintyUnit
