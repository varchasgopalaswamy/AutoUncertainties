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

Propagation of uncertainties is critically important in experimental sciences.
While the rules of (linear) uncertainty propagation
are simple, managing many variables with uncertainty information
can quickly become complicated in large scientific software stacks, and require
keeping track of many variables and implementing custom error propagation
rules for each mathematical operator.

# Statement of need

`AutoUncertainties` is Python package for uncertainty propagation. It provides
a drop-in mechanism to add uncertainty information to python scalar and `numpy`
array variables. It implements manual propagation rules for the python dunder math
methods, and uses automatic differentiation via `JAX` to propagate uncertainties
for most numpy methods applied to both scalar and numpy array variables. In doing so,
it eliminates the need for carrying around additional uncertainty variables,
needing to implement custom propagation rules for any numpy operator with a gradient
rule implemented by `jax`, and in most cases requires minimal re-writing of existing code,
typically only when uncertainties are attached to central values.

# Prior Work

To the author's knowledge, the only existing error propagation library in python is the `uncertainties` package,
which inspired the current work. While extremely useful, the `uncertainties` package
relies on hand-implemented rules for uncertainty propagation of array and scalar data, therefore
making it somewhat to evaluate complex mathematical expressions. For instance, calculating the uncertainty
propagation due to the cosine must be handled as

```python
{
import numpy as np
from uncertainties import unumpy, ufloat
arr = np.array([ufloat(1, 0.1), ufloat(2, 0.002)])
unumpy.cos(arr)
}
```

rather than the desired

```python
{
import numpy as np
from uncertainties import ufloat
arr = np.array([ufloat(1, 0.1), ufloat(2, 0.002)])
np.cos(arr)
}
```

# Implementation

Linear uncertainty propagation of a function $f(x) : \mathbb{R}^n \rightarrow \mathbb{R}^m$ can be computed
via the simple rule $\delta f_j (x)^2 = \left ( \dfrac{\del f_j}{\del x_i}\left( x \right ) \delta x_i  \right ) ^2 $

To compute $\dfrac{\del f_j}{\del x_i}$ for arbitrary $f$, the implementation in `AutoUncertainties` relies on automatic
differentiaion provided by `jax`. Calls to any `numpy` array function or ufunc are intercepted via the `__array_function`
and `__array_ufunc__` mechanism, and dispatched to a numpy wrapper routine that computes the Jacobian matrix via `jax.jacfwd`.


# Acknowledgements

This material is based upon work supported by the Department of Energy [National Nuclear Security Administration] University of Rochester “National Inertial Confinement Fusion Program” under Award Number(s) DE-NA0004144, and Department of Energy [Office of Fusion Energy Sciences] University of Rochester “Applications of Machine Learning and Data Science to predict, design and improve laser-fusion implosions for inertial fusion energy” under Award Number(s) DE-SC0024381.

This report was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor any agency thereof, nor any of their employees, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
