
Welcome to AutoUncertainties's documentation!
=============================================

.. image:: https://img.shields.io/badge/GitHub-AutoUncertainties-blue?logo=github&labelColor=black
   :target: https://github.com/varchasgopalaswamy/AutoUncertainties
   :alt: Static Badge

.. image:: https://img.shields.io/github/v/release/varchasgopalaswamy/AutoUncertainties?label=Current%20Release&color
   :target: https://github.com/varchasgopalaswamy/AutoUncertainties/releases
   :alt: GitHub Release

.. image:: https://img.shields.io/badge/Python-3.15%2B-ffed57?logo=python&logoColor=white
   :target: https://www.python.org/downloads/
   :alt: python

.. image:: https://joss.theoj.org/papers/d357e888e33e56df674e15c82b82dcac/status.svg
   :target: https://joss.theoj.org/papers/d357e888e33e56df674e15c82b82dcac
   :alt: JOSS Paper


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


Usage
-----

* See :doc:`basic_usage`


Quick Reference
---------------

* `~auto_uncertainties.uncertainty.uncertainty_containers.Uncertainty`
* :doc:`Exceptions <api/auto_uncertainties/exceptions/index>`
* :doc:`Contribution Information <contribution>`


Statement of Need
-----------------

AutoUncertainties is a Python package for uncertainty propagation of independent random variables. 
It provides a drop-in mechanism to add uncertainty information to Python scalar and NumPy array 
objects. It implements manual propagation rules for the Python dunder math methods, and uses automatic 
differentiation via JAX to propagate uncertainties for most NumPy methods applied to both scalar 
and NumPy array variables. In doing so, it eliminates the need for carrying around additional 
uncertainty variables or for implementing custom propagation rules for any NumPy operator with 
a gradient rule implemented by JAX. In  most cases, it requires minimal modification to 
existing code—typically only when uncertainties are  attached to central values.

One of the most important aspects of AutoUncertainties is its seamless support for NumPy:

.. code-block:: python
   :caption: Example

   >>> import numpy as np
   >>> from auto_uncertainties import Uncertainty
   >>> vals = np.array([0.5, 0.75])
   >>> errs = np.array([0.05, 0.3])
   >>> u = Uncertainty(vals, errs)
   >>> print(np.cos(u))
   [0.878 +/- 0.024, 0.73 +/- 0.20]

This is in contrast to the `uncertainties` package, which would have required the use of `unumpy`,
a module containing several hand-implemented analogs of the true NumPy functions. 


Current Limitations and Future Work
-----------------------------------

Dependent Random Variables
**************************

To simplify operations on `~auto_uncertainties.uncertainty.uncertainty_containers.Uncertainty` objects,
`AutoUncertainties` assumes all variables are independent. This means that, in the case where the 
programmer assumes dependence between two or more `Uncertainty` objects, unexpected and counter-intuitive 
behavior may arise during uncertainty propagation. This is a common pitfall when working with `Uncertainty` 
objects, especially since the package will not prevent you from manipulating variables in a manner that 
implies dependence.

- **Subtracting Equivalent Uncertainties**

  Subtracting an `Uncertainty` from itself will not result in a standard deviation of zero.

  .. code-block:: python
     :caption: Example

     >>> x = Uncertainty(5.0, 0.5)
     >>> x - x  
     0.0 +/- 0.7

- **Mean Error Propagation**

  When multiplying a vector by a scalar `Uncertainty` object, each component of the resulting vector
  is assumed to be a multivariate normal distribution with no covariance,
  which may not be the desired behavior. For instance, taking the mean of such a
  vector will return an `Uncertainty` object with an unexpectedly small standard deviation.

  .. code-block:: python
     :caption: Example

     >>> u = Uncertainty(5.0, 0.5)
     >>> arr = np.ones(10) * 10
     >>> np.mean(u * arr) 
     50.0 +/- 1.6, rather than 50 +/- 5 as expected

  To obtain the uncertainty corresponding to the case where each element of the array is fully correlated,
  two workaround techniques can be used:

  1. Separate the central value from the relative error, multiply the vector by the central value, take the mean
     of the resulting vector, and then multiply by the previously stored relative error.

     .. code-block:: python
        :caption: Example

        >>> u = Uncertainty(5.0, 0.5)
        >>> scale_error = Uncertainty(1, u.relative)  # collect relative error
        >>> scale_value = u.value                     # collect central value

        >>> arr = np.ones(10) * 10
        >>> np.mean(scale_value * arr) * scale_error
        50 +/- 5

  2. Take the mean of the vector, and then multiply by the `Uncertainty`:

     .. code-block:: python

        >>> u = Uncertainty(5.0, 0.5)
        >>> arr = np.ones(10) * 10
        >>> u * np.mean(arr)
        50 +/- 5

These workarounds are nevertheless cumbersome, and cause `AutoUncertainties` to fall somewhat short of the original
goals of automated error propagation. In principle, this could be addressed by storing a full computational
graph of the result of chained operations, similar to what is done in `uncertainties`. However, the complexity
of such a system places it out of scope for `AutoUncertainties` at this time.

It should be noted that, in cases where random variables have covariance that lies somewhere between 
fully correlated and fully independent, calculations like those described above would be more complex.
To accurately propagate uncertainty, one would need to specify individual correlations between each 
variable, and adjust the computation as necessary. This is also currently out of scope for `AutoUncertainties`. 


Inspirations
------------

The class structure of `~auto_uncertainties.uncertainty.uncertainty_containers.Uncertainty`, and the NumPy
ufunc implementation is heavily inspired by the excellent package `Pint <https://github.com/hgrecco/pint>`_.


Indices and Tables
==================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   basic_usage
   numpy_integration
   contribution
   migration

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/auto_uncertainties/index


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
