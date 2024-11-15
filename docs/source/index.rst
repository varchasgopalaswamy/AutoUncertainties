
Welcome to AutoUncertainties's documentation!
=============================================

.. image:: https://img.shields.io/badge/GitHub-AutoUncertainties-blue?logo=github&labelColor=black
   :target: https://github.com/varchasgopalaswamy/AutoUncertainties
   :alt: Static Badge

.. image:: https://img.shields.io/github/v/release/varchasgopalaswamy/AutoUncertainties?label=Current%20Release&color
   :target: https://github.com/varchasgopalaswamy/AutoUncertainties/releases
   :alt: GitHub Release

.. image:: https://img.shields.io/badge/Python-3.11%20%7C%203.12-ffed57?logo=python&logoColor=white
   :target: https://www.python.org/downloads/
   :alt: python


AutoUncertainties is a package that makes handling linear uncertainty propagation for scientific applications
straightforward and automatic using auto-differentiation.

   For instructions on how to install AutoUncertainties, see :doc:`getting_started`.

Supported Features
------------------

.. raw:: html

  <div>
    <input type="checkbox" id="scalars" name="scalars" checked disabled />
    <label for="scalars">Scalars</label>
  </div>
  <div>
    <input type="checkbox" id="arrays" name="arrays" checked disabled />
    <label for="arrays">Arrays, with support for most NumPy ufuncs and functions</label>
  </div>
  <div>
    <input type="checkbox" id="pint" name="pint" checked disabled />
    <label for="arrays">Integration with <a href="https://pint.readthedocs.io/en/stable/user/defining-quantities.html">Pint</a> Quantity objects</label>
  </div>
  <div>
    <input type="checkbox" id="pandas" name="pandas" unchecked disabled />
    <label for="arrays">Pandas Extension Type (see <a href="https://pandas.pydata.org/docs/reference/api/pandas.api.extensions.ExtensionDtype.html">here</a>)</label>
  </div>


Usage
-----

* See :doc:`basic_usage`
* See :doc:`Pint extensions <api/auto_uncertainties/pint/extensions/index>`


Quick Reference
---------------

* `~auto_uncertainties.uncertainty.uncertainty_containers.Uncertainty`
* `~auto_uncertainties.uncertainty.uncertainty_containers.VectorUncertainty`
* `~auto_uncertainties.uncertainty.uncertainty_containers.ScalarUncertainty`
* :doc:`Exceptions <api/auto_uncertainties/exceptions/index>`


Inspirations
------------

The class structure of `~auto_uncertainties.uncertainty.uncertainty_containers.Uncertainty`, and the `numpy`
ufunc implementation is heavily inspired by the excellent package `Pint <https://github.com/hgrecco/pint>`_.


Indices and Tables
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
