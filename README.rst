.. image:: https://img.shields.io/pypi/v/auto-uncertainties.svg
    :target: https://pypi.org/project/auto-uncertainties/
    :alt: Latest Version

.. image:: https://img.shields.io/pypi/l/auto-uncertainties.svg
    :target: https://pypi.org/project/auto-uncertainties/
    :alt: License

.. image:: https://github.com/varchasgopalaswamy/AutoUncertainties/actions/workflows/python-app.yml/badge.svg
    :target: https://github.com/varchasgopalaswamy/AutoUncertainties/actions?query=workflow
    :alt: Tests

AutoUncertainties
========================

AutoUncertainties is a package that makes handling linear uncertainty propagation for scientific applications straightforward and automatic using auto-differentiation.

Supported Features
#####################

☑ Scalars

☑ Arrays, with support for most NumPy ufuncs and functions

☐ Pandas Extension Type

Usage
================

Creating a scalar Uncertainty variable is relatively simple:

.. code-block:: python

    >>> from auto_uncertainties import Uncertainty
    >>> value = 1.0
    >>> error = 0.1
    >>> u = Uncertainty(value,error)
    >>> u
    1.0 +/- 0.1

as is creating a numpy array of  Uncertainties:

.. code-block:: python

    >>> from auto_uncertainties import Uncertainty
    >>> import numpy as np
    >>> value = np.linspace(start=0,stop=10,num=5)
    >>> error = np.ones_like(value)*0.1
    >>> u = Uncertainty(value,error)
    >>> u
    [ 0.   2.5  5.   7.5 10. ] +/- [0.1 0.1 0.1 0.1 0.1]

Scalar uncertainties implement all mathematical and logical `dunder methods <https://docs.python.org/3/reference/datamodel.html#object.__repr__>`_ explicitly.

.. code-block:: python

    >>> from auto_uncertainties import Uncertainty
    >>> u = Uncertainty(10.0, 3.0)
    >>> v = Uncertainty(20.0, 4.0)
    >>> u + v
    30.0 +/- 5.0

Array uncertainties implement a large subset of the numpy ufuncs and methods using :code:`jax.grad` or :code:`jax.jacfwd`, depending on the output shape.

.. code-block:: python

    >>> from auto_uncertainties import Uncertainty
    >>> import numpy as np
    >>> value = np.linspace(start=0,stop=10,num=5)
    >>> error = np.ones_like(value)*0.1
    >>> u = Uncertainty(value,error)
    >>> np.exp(u)
    [1.00000000e+00 1.21824940e+01 1.48413159e+02 1.80804241e+03
 2.20264658e+04] +/- [1.00000000e-01 1.21824940e+00 1.48413159e+01 1.80804241e+02
 2.20264658e+03]
    >>> np.sum(u)
    25.0 +/- 0.223606797749979
    >>> u.sum()
    25.0 +/- 0.223606797749979
    >>> np.sqrt(np.sum(error**2))
    0.223606797749979

The mean value and the standard deviation (the measurements are assumed to be normally distributed) can be accessed via

.. code-block:: python

    >>> from auto_uncertainties import Uncertainty
    >>> u = Uncertainty(10.0, 3.0)
    >>> u.value
    10.0
    >>> u.error
    3.0

Prerequisites
===========

For array support:

* jax
* jaxlib (must be built from source if you are not on Linux machine with AVX instruction)
* numpy

sets.

Installation
===============

To install simply run :code:`pip install auto_uncertainties`

Inspirations
================

The class structure of :code:`Uncertainty`, and the NumPy ufunc implementation is heavily inspired by the excellent package `pint <https://github.com/hgrecco/pint>`_.
