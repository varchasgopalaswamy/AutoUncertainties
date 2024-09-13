
Welcome to AutoUncertainties's documentation!
=============================================

AutoUncertainties is a package that makes handling linear uncertainty propagation for scientific applications
straightforward and automatic using auto-differentiation.

   For instructions on how to install AutoUncertainties, see :doc:`getting_started`.


Supported Features
------------------

- 🗹 Scalars
- 🗹 Arrays, with support for most `numpy` ufuncs and functions
- ☐ Pandas Extension Type

Usage
-----

* See :doc:`basic_usage`

Inspirations
------------

The class structure of `~auto_uncertainties.uncertainty.uncertainty_containers.Uncertainty`, and the `numpy`
ufunc implementation is heavily inspired by the excellent package `pint <https://github.com/hgrecco/pint>`_.


Indices and tables
==================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   basic_usage
   numpy_integration
   pandas_integration

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/auto_uncertainties/index


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
