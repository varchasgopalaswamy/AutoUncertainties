NumPy Integration
=================

Using `JAX <https://jax.readthedocs.io/en/latest/>`_ automatic differentiation,
linear uncertainty propagation is enabled for most `numpy` operations.

.. code-block:: python

   >>> from auto_uncertainties import Uncertainty
   >>> import numpy as np
   >>> value = np.linspace(start=0, stop=10, num=5)
   >>> error = np.ones_like(value)*0.1
   >>> u = Uncertainty(value, error)
   >>> u
   [0.00 +/- 0.10, 2.50 +/- 0.10, 5.00 +/- 0.10, 7.50 +/- 0.10, 10.00 +/- 0.10]


.. code-block:: python

   >>> from auto_uncertainties import Uncertainty
   >>> import numpy as np
   >>> value = np.linspace(start=0, stop=10, num=5)
   >>> error = np.ones_like(value)*0.1
   >>> u = Uncertainty(value, error)
   >>> np.exp(u)
   [1.00 +/- 0.10, 12.2 +/- 1.2, 148 +/- 15, 1.81e+3 +/- 1.8e+2, 2.20e+4 +/- 2.2e+3]
   >>> np.sum(u)
   25.00 +/- 0.22
   >>> u.sum()
   25.00 +/- 0.22
   >>> np.sqrt(np.sum(error**2))
   0.223606797749979
