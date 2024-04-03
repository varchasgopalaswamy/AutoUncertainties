Numpy Integration
================

Using `Jax <https://jax.readthedocs.io/en/latest/>`_ to provide auto-differentiation capabilities (either :code:`jax.grad` or :code:`jax.jacfwd`), linear uncertainty propagation is enabled for most numpy operations

.. code-block:: python

    >>> from auto_uncertainties import Uncertainty
    >>> import numpy as np
    >>> value = np.linspace(start=0,stop=10,num=5)
    >>> error = np.ones_like(value)*0.1
    >>> u = Uncertainty(value,error)
    >>> u
    [ 0.   2.5  5.   7.5 10. ] +/- [0.1 0.1 0.1 0.1 0.1]


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
