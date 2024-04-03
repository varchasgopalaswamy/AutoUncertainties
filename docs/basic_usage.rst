Basic Usage
================

The goal is to have minimal changes to your code in order to enable uncertainty propagation.

Creating a scalar Uncertainty variable is relatively simple:

.. code-block:: python

    >>> from auto_uncertainties import Uncertainty
    >>> value = 1.0
    >>> error = 0.1
    >>> u = Uncertainty(value,error)
    >>> u
    1.0 +/- 0.1


Scalar uncertainties implement all mathematical and logical `dunder methods <https://docs.python.org/3/reference/datamodel.html#object.__repr__>`_ explicitly using linear uncertainty propagation.

.. code-block:: python

    >>> from auto_uncertainties import Uncertainty
    >>> u = Uncertainty(10.0, 3.0)
    >>> v = Uncertainty(20.0, 4.0)
    >>> u + v
    30.0 +/- 5.0


The central value, uncertainty and relative error are available as attributes

.. code-block:: python

    >>> from auto_uncertainties import Uncertainty
    >>> u = Uncertainty(10.0, 3.0)
    >>> u.value
    10.0
    >>> u.error
    3.0
    >>> u.rel
    0.33333

To strip central values and uncertainty from arbitrary variables, accessor functions `nominal_values` and `std_devs` are provided

.. code-block:: python

    >>> from auto_uncertainties import nominal_values, std_devs
    >>> u = Uncertainty(10.0, 3.0)
    >>> v = 5.0
    >>> nominal_values(u)
    10.0
    >>> std_devs(u)
    3.0
    >>> nominal_values(v)
    5.0
    >>> std_devs(v)
    0.0
