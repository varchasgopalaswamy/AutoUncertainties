
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

```python
    >>> from auto_uncertainties import Uncertainty
    >>> value = 1.0
    >>> error = 0.1
    >>> u = Uncertainty(value,error)
    >>> u
    1.0 +/- 0.1
```

as is creating a numpy array of  Uncertainties:

```python

    >>> from auto_uncertainties import Uncertainty
    >>> import numpy as np
    >>> value = np.linspace(start=0,stop=10,num=5)
    >>> error = np.ones_like(value)*0.1
    >>> u = Uncertainty(value,error)
```

(though, they are actually different classes!)

```python

    >>> from auto_uncertainties import Uncertainty
    >>> value = 1.0
    >>> error = 0.1
    >>> u = Uncertainty(value,error)
    >>> type(u)
    auto_uncertainties.uncertainty.ScalarUncertainty

    >>> from auto_uncertainties import Uncertainty
    >>> import numpy as np
    >>> value = np.linspace(start=0,stop=10,num=5)
    >>> error = np.ones_like(value)*0.1
    >>> u = Uncertainty(value,error)
    >>> u
    auto_uncertainties.uncertainty.VectorUncertainty
```

Scalar uncertainties implement all mathematical and logical `dunder methods <https://docs.python.org/3/reference/datamodel.html#object.__repr__>`_ explicitly.

```python

    >>> from auto_uncertainties import Uncertainty
    >>> u = Uncertainty(10.0, 3.0)
    >>> v = Uncertainty(20.0, 4.0)
    >>> u + v
    30.0 +/- 5.0
```
Array uncertainties implement a large subset of the numpy ufuncs and methods using :code:`jax.grad` or :code:`jax.jacfwd`, depending on the output shape.

```python

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
```
The mean value and the standard deviation (the measurements are assumed to be normally distributed) can be accessed via

```python

    >>> from auto_uncertainties import Uncertainty
    >>> u = Uncertainty(10.0, 3.0)
    >>> u.value
    10.0
    >>> u.error
    3.0
```

Displayed values are automatically rounded according to the Particle Data Group standard. This can be turned off using `set_display_rounding`
```python

    >>> from auto_uncertainties import set_display_rounding
    >>> set_display_rounding(False)
    >>> from auto_uncertainties import Uncertainty
    >>> import numpy as np
    >>> value = np.linspace(start=0,stop=10,num=5)
    >>> error = np.ones_like(value)*0.1
    >>> u = Uncertainty(value,error)
    >>> np.sum(u)
    25.0 +/- 0.223606797749979
```
If `np.array` is called on an `Uncertainty` object, it will automatically get cast down to a numpy array (and lose uncertainty information!), and emit a warning. To make this an error, use `set_downcast_error`
```python

    >>> from auto_uncertainties import set_downcast_error
    >>> set_downcast_error(False)
    >>> from auto_uncertainties import Uncertainty
    >>> import numpy as np
    >>> value = np.linspace(start=0,stop=10,num=5)
    >>> error = np.ones_like(value)*0.1
    >>> u = Uncertainty(value,error)
    >>> np.array(u)

    Exception: The uncertainty is stripped when downcasting to ndarray.
```



Prerequisites
===========

For array support:

* jax
* jaxlib
* numpy



Installation
===============

To install simply run :code:`pip install auto_uncertainties`

Inspirations
================

The class structure of :code:`Uncertainty`, and the NumPy ufunc implementation is heavily inspired by the excellent package `pint <https://github.com/hgrecco/pint>`_.
