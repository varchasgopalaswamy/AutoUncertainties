---
title: 'AutoUncertainties: A Python Package for Uncertainty Propagation'

tags:
  - Python
  - uncertainty propagation
  
authors:
  - name: Varchas Gopalaswamy
    orcid: 0000-0002-8013-9314
    equal-contrib: true
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Ethan Mentzer
    orcid: 0009-0003-8206-5050
    affiliation: "1"
    equal-contrib: false
    
affiliations:
 - name: Laboratory for Laser Energetics, Rochester, USA
   index: 1

bibliography: paper.bib
date: 2 February 2025
---

# Summary

Propagation of uncertainties is of great utility in the experimental sciences.
While the rules of (linear) uncertainty propagation are simple, managing many 
variables with uncertainty information can quickly become complicated in large 
scientific software stacks, and require keeping track of many variables and 
implementing custom error propagation rules for each mathematical operator. 
The Python package `AutoUncertainties`, described here, provides a solution to this problem.


# Statement of Need

`AutoUncertainties` is Python package for uncertainty propagation. It provides
a drop-in mechanism to add uncertainty information to Python scalar and `NumPy`
[@harris2020array] array variables. It implements manual propagation rules for the Python dunder
math methods, and uses automatic differentiation via `JAX` [@jax2018github] to propagate uncertainties
for most `NumPy` methods applied to both scalar and `NumPy` array variables. In doing so,
it eliminates the need for carrying around additional uncertainty variables,
needing to implement custom propagation rules for any `NumPy` operator with a gradient
rule implemented by `JAX`, and in most cases requires minimal modification to existing code,
typically only when uncertainties are attached to central values.


# Prior Work

To the author's knowledge, the only existing error propagation library in Python is the `uncertainties` 
[@lebigot2024] package, which inspired the current work. While extremely useful, the `uncertainties` 
package relies on hand-implemented rules and functions for uncertainty propagation of array and scalar data. 
While this is transparent for the intrinsic dunder methods such as `__add__`, it becomes problematic for advanced 
mathematical operators. For instance, calculating the uncertainty propagation due to the cosine requires the 
import of separate math libraries

```python
import numpy as np
from uncertainties import unumpy, ufloat
arr = np.array([ufloat(1, 0.1), ufloat(2, 0.002)])
unumpy.cos(arr)  # calculation succeeds
```

rather than being able to use `NumPy` directly.

```python
import numpy as np
from uncertainties import ufloat
arr = np.array([ufloat(1, 0.1), ufloat(2, 0.002)])
np.cos(arr)  # raises an exception
```


# Implementation

Linear uncertainty propagation of a function $f(x) : \mathbb{R}^n \rightarrow \mathbb{R}^m$ can be computed
via the simple rule $$ \delta f_j (x)^2 = \left ( \dfrac{\partial f_j}{\partial x_i}\left( x \right ) \delta x_i  \right ) ^2. $$

To compute $\dfrac{\partial f_j}{\partial x_i}$ for arbitrary $f$, the implementation in `AutoUncertainties` relies on
automatic differentiaion provided by `JAX`. Calls to any `NumPy` array function or universal function (ufunc) are 
intercepted via the `__array_function__` and `__array_ufunc__` mechanism, and dispatched to a `NumPy` wrapper routine 
that computes the Jacobian matrix via `jax.jacfwd`.

The user API for the `Uncertainty` object exposes only a small set of properties:

- `value -> float`: The central value of the object.
- `error -> float`: The error of the object.
- `relative -> float`: The relative error (i.e. error / value) of the object.
- `plus_minus(self, err: float) -> Uncertainty`: Adds error (in quadrature).
- `from_sequence(self, seq: List[ScalarUncertainty]) -> VectorUncertainty`: Constructs an array `Uncertainty` object 
  from a list of scalar `Uncertainty` objects.

To extract errors / central values from arbitrary objects, the accessors `nominal_values` and `std_devs` are provided. 
These functions return:

- The central values and errors, respectively, if the input is an `Uncertainty` object.
- The input and zero if the input is any other kind of object.

`Uncertainty` objects are displayed using rounding rules based on the uncertainty, i.e.,

- Error to 2 significant digits.
- Central value to first signficant digit of error, or two significant figures (whichever is more significant digits).

This behavior can be toggled using `set_display_rounding`:

```python
from auto_uncertainties import set_display_rounding
set_display_rounding(False)
```

Calling `__array__`, whether via `numpy.array` or any other method, will by default issue a warning, and convert
the `Uncertainty` object into an equivalent array of its nominal values, stripping all error information. To prevent 
this behavior, the `set_downcast_error` function can be called so that an exception is raised instead:

```python
from auto_uncertainties import set_downcast_error
set_downcast_error(True)
```


## Pint

`AutoUncertainties` provides some support for working with objects from the `Pint` package [@pint]. 
For example, `Uncertainty` objects can be instantiated from `pint.Quantity` objects, and then 
automatically wrapped into new `pint.Quantity` objects via the `from_quantities` method. This 
guarantees that unit information is preserved when moving between `Uncertainty` objects and
`pint.Quantity` objects.


## Pandas

Support for `pandas` [@pandas2024] via the `ExtensionArray` mechanism is largely functional.



# Acknowledgements

This material is based upon work supported by the Department of Energy [National Nuclear Security Administration] 
University of Rochester "National Inertial Confinement Fusion Program" under Award Number(s) DE-NA0004144, and 
Department of Energy [Office of Fusion Energy Sciences] University of Rochester "Applications of Machine Learning and 
Data Science to predict, design and improve laser-fusion implosions for inertial fusion energy" under Award Number(s) 
DE-SC0024381.

This report was prepared as an account of work sponsored by an agency of the United States Government. Neither 
the United States Government nor any agency thereof, nor any of their employees, makes any warranty, express or 
implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any 
information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned 
rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, 
or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States 
Government or any agency thereof. The views and opinions of authors expressed herein do not necessarily state or reflect 
those of the United States Government or any agency thereof.


# References
