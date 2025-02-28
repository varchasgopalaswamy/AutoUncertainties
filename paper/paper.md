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
    
affiliations:
 - name: Laboratory for Laser Energetics, Rochester, USA
   index: 1

bibliography: paper.bib
date: 7 February 2025
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

The user API for the `Uncertainty` object exposes a number of properties and methods, of which some of the most 
important are:

- `value -> float`: The central value of the object.
- `error -> float`: The error of the object.
- `relative -> float`: The relative error (i.e. error / value) of the object.
- `plus_minus(self, err: float) -> Uncertainty`: Adds error (in quadrature).
- `from_sequence(cls, seq: Sequence) -> VectorUncertainty`: Constructs an array `Uncertainty` object 
  from some existing sequence.

These attributes and methods can be used in the following manner:

```python
from auto_uncertainties import Uncertainty
u1 = Uncertainty(5.25, 0.75)
u2 = Uncertainty(1.85, 0.4)

print(u1)                  # 5.25 +/- 0.75
print(u1.value)            # 5.25
print(u1.error)            # 0.75
print(u1.relative)         # 0.142857
print(u1.plus_minus(0.5))  # 5.25 +/- 0.901388

seq = Uncertainty.from_sequence([u1, u2])
print(seq)  # [5.25 +/- 0.75, 1.85 +/- 0.4]
```

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


## Support for Pint

`AutoUncertainties` provides some support for working with objects from the `Pint` package [@pint]. 
For example, `Uncertainty` objects can be instantiated from `pint.Quantity` objects, and then 
automatically wrapped into new `pint.Quantity` objects via the `from_quantities` method. This 
guarantees that unit information is preserved when moving between `Uncertainty` objects and
`pint.Quantity` objects.

```python
from auto_uncertainties import Uncertainty
from pint import Quantity

val = Quantity(2.24, 'kg')
err = Quantity(0.208, 'kg')
new_quantity = Uncertainty.from_quantities(val, err)

print(new_quantity)        # 2.24 +/- 0.208 kilogram
print(type(new_quantity))  # <class 'pint.Quantity'>
```


## Pandas

Support for `pandas` [@pandas2024] via the `ExtensionArray` mechanism is largely functional. Upcoming
aditions to `AutoUncertainties` will further improve compatibility.



# Current Limitations and Future Work

## Dependent Random Variables

To simplify operations on `Uncertainty` objects, `AutoUncertainties` assumes all variables are independent
and identically distributed (i.i.d.). This means that, in the case where the programmer assumes dependence
between two or more `Uncertainty` objects, unexpected behavior may arise. Some examples of this phenomenon are
demonstrated in the following subsections, however programmers should be aware that there are likely many more 
edge cases related to the i.i.d. assumption. 

### Subtracting Equivalent Uncertainties

Subtracting an `Uncertainty` from itself will not result in a standard devation of zero, as demonstrated
in the following example.

```python
x = Uncertainty(5.0, 0.5)
print(x - x)  # 0 +/- 0.707107
```

### Mean Error Propagation

When multiplying a vector by a scalar `Uncertainty` object, each component of the resulting vector 
is assumed to be i.i.d., which may not be the desired behavior. For instance, taking the mean of such a 
vector will return an `Uncertainty` object with an unexpectedly small standard deviation. To avoid this
scenario, the programmer can take the mean assuming *full correlation* between the components of the vector
by using one of two workaround techniques:

1. Separate the central value from the relative error, multiply the vector by the central value, take the mean
   of the resulting vector, and then multiply by the previously stored relative error.

   ```python
   u = Uncertainty(5.0, 0.5)
   scale_error = Uncertainty(1, u.relative)  # collect relative error
   scale_value = u.value                     # collect central value

   arr = np.ones(10) * 10
   result = np.mean(scale_value * arr) * scale_error  # 50 +/- 5
   ```

2. Take the mean of the vector, and then multiply by the `Uncertainty`:

   ```python
   u = Uncertainty(5.0, 0.5)
   arr = np.ones(10) * 10
   result = u * np.mean(arr)  # 50 +/- 5
   ```

These two workarounds are in contrast to the following method of taking the mean of a vector multiplied
by a scalar `Uncertainty`, which, as previously mentioned, would result in a reduced standard deviation
becase of the i.i.d. assumption.

```python
u = Uncertainty(5.0, 0.5)
arr = np.ones(10) * 10
result = np.mean(u * arr)  # 50 +/- 1.58114
```

While it would be theoretically possible for `AutoUncertainties` to automatically determine the dependence between
variables, an implementation of this feature would likely come at the cost of reduced performance and high complexity.
Therefore, programmers should exercise caution when working with dependent random variables.


## Typing System

Type hinting is employed throughout `AutoUncertainties` to aid static analysis of the package. At this time,
however, many typing inconsistencies can be detected by static type enforcement tools like 
[Mypy](https://mypy-lang.org/) and [Pyright](https://microsoft.github.io/pyright/). Future improvements 
to `AutoUncertainties` will likely include typing adjustments to the code, in order to avoid subtle bugs.



# Further Information

Additional API information and usage examples can be found on the 
[documentation website](https://autouncertainties.readthedocs.io/en/latest/). All source
code for the project is stored and maintained on the `AutoUncertainties` 
[GitHub repository](https://github.com/varchasgopalaswamy/AutoUncertainties), where 
contributions, suggestions, and bug reports are welcome. 



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
