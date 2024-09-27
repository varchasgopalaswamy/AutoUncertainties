Basic Usage
===========

The goal is to have minimal changes to your code in order to enable uncertainty propagation.

* Creating a scalar `~auto_uncertainties.uncertainty.uncertainty_containers.Uncertainty` variable is relatively simple:

  .. code-block:: python

     >>> from auto_uncertainties import Uncertainty
     >>> value = 1.0
     >>> error = 0.1
     >>> u = Uncertainty(value,error)
     >>> u
     1.0 +/- 0.1

  As is creating a `numpy` array of Uncertainties:

  .. code-block:: python

     >>> from auto_uncertainties import Uncertainty
     >>> import numpy as np
     >>> value = np.linspace(start=0,stop=10,num=5)
     >>> error = np.ones_like(value)*0.1
     >>> u = Uncertainty(value,error)

  - (though, they are actually different classes!)

    .. code-block:: python

       >>> from auto_uncertainties import Uncertainty
       >>> value = 1.0
       >>> error = 0.1
       >>> u = Uncertainty(value,error)
       >>> type(u)
       auto_uncertainties.uncertainty.ScalarUncertainty

    .. code-blocK:: python

       >>> from auto_uncertainties import Uncertainty
       >>> import numpy as np
       >>> value = np.linspace(start=0,stop=10,num=5)
       >>> error = np.ones_like(value)*0.1
       >>> u = Uncertainty(value,error)
       >>> u
       auto_uncertainties.uncertainty.VectorUncertainty

* Scalar uncertainties implement all mathematical and logical
  `dunder methods <https://docs.python.org/3/reference/datamodel.html#object.__repr__>`_ explicitly using linear
  uncertainty propagation.

  .. code-block:: python

     >>> from auto_uncertainties import Uncertainty
     >>> u = Uncertainty(10.0, 3.0)
     >>> v = Uncertainty(20.0, 4.0)
     >>> u + v
     30.0 +/- 5.0

* Array uncertainties implement a large subset of the numpy ufuncs and methods using `jax.grad` or
  `jax.jacfwd`, depending on the output shape.

  .. code-blocK:: python

     >>> from auto_uncertainties import Uncertainty
     >>> import numpy as np
     >>> value = np.linspace(start=0,stop=10,num=5)
     >>> error = np.ones_like(value)*0.1
     >>> u = Uncertainty(value,error)
     >>> np.exp(u)
     Magnitude

     1, 12.182, 148.413, 1808.04, 22026.5

     Error

     0.1, 1.2, 15, 180, 2200
     >>> np.sum(u)
     25.0 +/- 0.22
     >>> u.sum()
     25.0 +/- 0.22
     >>> np.sqrt(np.sum(error**2))
     0.223606797749979

* The central value, uncertainty, and relative error are available as attributes:

  .. code-block:: python

     >>> from auto_uncertainties import Uncertainty
     >>> u = Uncertainty(10.0, 3.0)
     >>> u.value
     10.0
     >>> u.error
     3.0
     >>> u.rel
     0.33333

* To strip central values and uncertainty from arbitrary variables, accessor functions `nominal_values`
  and `std_devs` are provided:

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

* Displayed values are automatically rounded according to the Particle Data Group standard.
  This can be turned off using `~auto_uncertainties.display_format.set_display_rounding`:

  .. code-block:: python

     >>> from auto_uncertainties import set_display_rounding
     >>> set_display_rounding(False)
     >>> from auto_uncertainties import Uncertainty
     >>> import numpy as np
     >>> value = np.linspace(start=0,stop=10,num=5)
     >>> error = np.ones_like(value)*0.1
     >>> u = Uncertainty(value,error)
     >>> np.sum(u)
     25.0 +/- 0.223606797749979

* If `numpy.array` is called on an `~auto_uncertainties.uncertainty.uncertainty_containers.Uncertainty` object, it will
  automatically get cast down to a `numpy` array (and lose uncertainty information!), and emit a warning.
  To make this an error, use `~auto_uncertainties.uncertainty.uncertainty_containers.set_downcast_error`:

  .. code-block:: python

     >>> from auto_uncertainties import set_downcast_error
     >>> set_downcast_error(False)
     >>> from auto_uncertainties import Uncertainty
     >>> import numpy as np
     >>> value = np.linspace(start=0,stop=10,num=5)
     >>> error = np.ones_like(value)*0.1
     >>> u = Uncertainty(value,error)
     >>> np.array(u)

     Exception: The uncertainty is stripped when downcasting to ndarray.

