NumPy Integration
=================

Using `Jax <https://jax.readthedocs.io/en/latest/>`_ to provide auto-differentiation capabilities
(either `jax.grad` or `jax.jacfwd`), linear uncertainty propagation is enabled for most `numpy` operations.

.. code-block:: python

   >>> from auto_uncertainties import Uncertainty
   >>> import numpy as np
   >>> value = np.linspace(start=0, stop=10, num=5)
   >>> error = np.ones_like(value)*0.1
   >>> u = Uncertainty(value, error)
   >>> u
   [0 +/- 0.1, 2.5 +/- 0.1, 5 +/- 0.1, 7.5 +/- 0.1, 10 +/- 0.1]


.. code-block:: python

   >>> from auto_uncertainties import Uncertainty
   >>> import numpy as np
   >>> value = np.linspace(start=0, stop=10, num=5)
   >>> error = np.ones_like(value)*0.1
   >>> u = Uncertainty(value, error)
   >>> np.exp(u)
   [1 +/- 0.1, 12.1825 +/- 1.21825, 148.413 +/- 14.8413, 1808.04 +/- 180.804, 22026.5 +/- 2202.65]
   >>> np.sum(u)
   25 +/- 0.223607
   >>> u.sum()
   25 +/- 0.223607
   >>> np.sqrt(np.sum(error**2))
   0.223606797749979
