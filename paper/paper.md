---
title: 'AutoUncertainties: A Python package for uncertainty propagation'
tags:
  - Python
  - uncertainty propagation
authors:
  - name: Varchas Gopalaswamy
    orcid: 0000-0002-8013-9314
    equal-contrib: true
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name: Laboratory for Laser Energetics, Rochester, USA
   index: 1
date: 3 April 2024

---

# Summary

Propagation of uncertainties is of great utility in the experimental sciences.
While the rules of (linear) uncertainty propagation
are simple, managing many variables with uncertainty information
can quickly become complicated in large scientific software stacks, and require
keeping track of many variables and implementing custom error propagation
rules for each mathematical operator. The python package `AutoUncertainties`,
described here, provides a solution to this problem.

# Statement of need

`AutoUncertainties` is Python package for uncertainty propagation. It provides
a drop-in mechanism to add uncertainty information to python scalar and `numpy`
array variables. It implements manual propagation rules for the python dunder math
methods, and uses automatic differentiation via `JAX` to propagate uncertainties
for most numpy methods applied to both scalar and numpy array variables. In doing so,
it eliminates the need for carrying around additional uncertainty variables,
needing to implement custom propagation rules for any numpy operator with a gradient
rule implemented by `jax`, and in most cases requires minimal modification to existing code,
typically only when uncertainties are attached to central values.

# Prior Work

To the author's knowledge, the only existing error propagation library in python is the `uncertainties` package,
which inspired the current work. While extremely useful, the `uncertainties` package
relies on hand-implemented rules and functions for uncertainty propagation of array and scalar data. While
this is transparent for the intrinsic dunder methods such as `__add__`, it becomes problematic for advanced
mathematical operators. For instance, calculating the uncertainty
propagation due to the cosine requires the import of separate math libraries

```python

import numpy as np
from uncertainties import unumpy, ufloat
arr = np.array([ufloat(1, 0.1), ufloat(2, 0.002)])
unumpy.cos(arr)

```

rather than being able to use `numpy` directly

```python
import numpy as np
from uncertainties import ufloat
arr = np.array([ufloat(1, 0.1), ufloat(2, 0.002)])
np.cos(arr)

```

# Implementation

Linear uncertainty propagation of a function $f(x) : \mathbb{R}^n \rightarrow \mathbb{R}^m$ can be computed
via the simple rule $$ \delta f_j (x)^2 = \left ( \dfrac{\partial f_j}{\partial x_i}\left( x \right ) \delta x_i  \right ) ^2 $$

To compute $\dfrac{\partial f_j}{\partial x_i}$ for arbitrary $f$, the implementation in `AutoUncertainties` relies on automatic
differentiaion provided by `jax`. Calls to any `numpy` array function or ufunc are intercepted via the `__array_function`
and `__array_ufunc__` mechanism, and dispatched to a numpy wrapper routine that computes the Jacobian matrix via `jax.jacfwd`.

The user API for the `Uncertainty` object exposes only a small set of properties.
- `value -> float`: The cenral value of the object
- `error -> float`: The error of the object
- `relative -> float`: The relative error (i.e. error / value) of the object
- `plus_minus(self, err:float) -> Uncertainty`: Adds error (in quadrature)
- `from_sequence(self, seq: List[ScalarUncertainty]) -> VectorUncertainty`: Constructs an array `Uncertainty` object from a list of scalar `Uncertainty` objects

To extract errors/central values from arbitrary objects, the accessors `nominal_values` and `std_devs` are provided. These
functions return
- The central values and errors respectively if the input is an `Uncertainty` object
- The input and zero if the input is any other kind of object

`Uncertainty` objects are displayed using rounding rules based on the uncertainty, i.e.

- Error to 2 significant digits
- Central value to first signficant digit of error, or two significant figures (whichever is more significant digits)

This behavior can be toggled using `set_display_rounding`:

```python
from auto_uncertainties import set_display_rounding
set_display_rounding(False)
```

Calling `__array__`, whether via `np.array` or any other method, will by default raise an error. This can be disabled so that a warning is issued instead
and the `Uncertainty` object is converted to an equivalent array of its nominal values via `set_downcast_error`

```python
from auto_uncertainties import set_downcast_error
set_downcast_error(False)
```

## Pandas

Support for `pandas` via the ExtensionArray mechanism is largely functional.




# Acknowledgements

This material is based upon work supported by the Department of Energy [National Nuclear Security Administration] University of Rochester “National Inertial Confinement Fusion Program” under Award Number(s) DE-NA0004144, and Department of Energy [Office of Fusion Energy Sciences] University of Rochester “Applications of Machine Learning and Data Science to predict, design and improve laser-fusion implosions for inertial fusion energy” under Award Number(s) DE-SC0024381.

This report was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor any agency thereof, nor any of their employees, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
