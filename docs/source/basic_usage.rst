Basic Usage
===========

The goal is to have minimal changes to your code in order to enable uncertainty propagation.

* Creating a scalar `~auto_uncertainties.uncertainty.uncertainty_containers.Uncertainty` variable is relatively simple:

  .. code-block:: python

     >>> from auto_uncertainties import Uncertainty
     >>> value = 1.0
     >>> error = 0.1
     >>> u = Uncertainty(value, error)
     >>> u
     1.00 +/- 0.10

  As is creating a `numpy` array of Uncertainties:

  .. code-block:: python

     >>> from auto_uncertainties import Uncertainty
     >>> import numpy as np
     >>> value = np.linspace(start=0, stop=10, num=5)
     >>> error = np.ones_like(value)*0.1
     >>> u = Uncertainty(value, error)
     >>> u
     [0.00 +/- 0.10, 2.50 +/- 0.10, 5.00 +/- 0.10, 7.50 +/- 0.10, 10.00 +/- 0.10]


  The `~auto_uncertainties.uncertainty.uncertainty_containers.Uncertainty` class automatically determines
  which methods should be implemented based on whether it represents a vector uncertainty, or a scalar
  uncertainty. When instantiated with a sequence or `numpy` array, vector-based operations are enabled;
  when instantiated with scalars, only scalar operations are permitted.

* Scalar uncertainties implement all mathematical and logical
  `dunder methods <https://docs.python.org/3/reference/datamodel.html#object.__repr__>`_ explicitly using linear
  uncertainty propagation.

  .. code-block:: python

     >>> from auto_uncertainties import Uncertainty
     >>> u = Uncertainty(10.0, 3.0)
     >>> v = Uncertainty(20.0, 4.0)
     >>> u + v
     30 +/- 5

* Array uncertainties implement a large subset of the numpy ufuncs and methods using
  JAX automatic differentiation.

  .. code-blocK:: python

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

* The central value, uncertainty, and relative error are available as attributes:

  .. code-block:: python

     >>> from auto_uncertainties import Uncertainty
     >>> u = Uncertainty(10.0, 3.0)
     >>> u.value
     10.0
     >>> u.error
     3.0
     >>> u.rel
     0.3

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

* Displayed values follow the Particle Data Group (PDG) rounding rules by default.
  Use ``configure(round_on_display=False)`` to disable PDG rounding and display values
  with the ``g`` format specifier:

  .. code-block:: python

     >>> from auto_uncertainties import Uncertainty, configure
     >>> import numpy as np
     >>> value = np.linspace(start=0, stop=10, num=5)
     >>> error = np.ones_like(value)*0.1
     >>> u = Uncertainty(value, error)
     >>> np.sum(u)
     25.00 +/- 0.22
     >>> configure(round_on_display=False)  # disable PDG rounding
     >>> np.sum(u)
     25 +/- 0.223607
     >>> configure(round_on_display=True)   # restore the default

  PDG rounding displays `Uncertainty` objects with:
   
  - Error to one or two significant digits, following the PDG thresholds.
  - Central value to the same decimal place as the rounded error.

  Rounding affects display only; ``.value`` and ``.error`` retain their stored precision.

* If `numpy.array` is called on an `~auto_uncertainties.uncertainty.uncertainty_containers.Uncertainty` object, it will
  automatically get cast down to a `numpy` array (losing all uncertainty information!), and emit a warning.
  To force an exception to be raised instead, use `~auto_uncertainties.config.configure`:

  .. code-block:: python

     >>> from auto_uncertainties import Uncertainty, configure
     >>> import numpy as np
     >>> configure(error_on_downcast=True)
     >>> value = np.linspace(start=0, stop=10, num=5)
     >>> error = np.ones_like(value)*0.1
     >>> u = Uncertainty(value, error)
     >>> np.array(u)
     Traceback (most recent call last):
         ...
     auto_uncertainties.exceptions.DowncastError: The uncertainty is stripped when downcasting to ndarray.


Custom functions
----------------

Use :func:`auto_uncertainties.propagate_uncertainties` as
``@propagate_uncertainties`` or ``@propagate_uncertainties()`` to propagate errors
through a JAX-compatible function. Write array operations using ``jax.numpy``;
the function receives nominal values. Uncertain inputs can appear in positional
or keyword arguments, including nested lists, tuples, and dictionaries. Outputs
can have different shapes from inputs and can use these nested structures too.

.. code-block:: python

   >>> import jax.numpy as jnp
   >>> from auto_uncertainties import Uncertainty, propagate_uncertainties
   >>> @propagate_uncertainties
   ... def scale_and_sum(values, *, gain):
   ...     scaled = values * gain
   ...     return {"scaled": scaled, "total": jnp.sum(scaled)}
   >>> result = scale_and_sum(
   ...     Uncertainty([1.0, 2.0], [0.1, 0.2]), gain=Uncertainty(2.0, 0.05)
   ... )
   >>> result["scaled"]
   [2.00 +/- 0.21, 4.0 +/- 0.4]
   >>> result["total"]
   6.0 +/- 0.5

The full Jacobian combines independent input errors for each output element.
Reusing an input inside the function is accounted for by differentiation.
Correlations between separate returned values are not stored, so keep a complete
calculation inside the decorated function when shared inputs affect its result.

Functions that mix array elements are supported. A tuple result preserves its structure:

.. code-block:: python

   >>> @propagate_uncertainties()
   ... def center_and_mean(values):
   ...     mean = jnp.mean(values)
   ...     return values - mean, mean
   >>> centered, mean = center_and_mean(Uncertainty([1.0, 2.0], [0.1, 0.2]))
   >>> centered
   [-0.50 +/- 0.11, 0.50 +/- 0.11]
   >>> mean
   1.50 +/- 0.11

Each centered value depends on both inputs, so its error is
``sqrt(0.1**2 + 0.2**2) / 2``, approximately ``0.1118``. The mean has the same
error. Correlations are not retained between separate calls or subsequent
operations on returned objects.

Dtypes follow JAX promotion. Float64 inputs enable JAX's 64-bit mode locally for
the call, restoring the caller's setting afterward. Results use NumPy-backed
containers; the decorated wrapper itself cannot be used inside ``jax.jit``.


Configuration
-------------

All settings are managed by :func:`auto_uncertainties.config.configure` and
:func:`auto_uncertainties.config.get_settings`. Initial values are read from
case-insensitive ``AUTO_UNCERTAINTIES_*`` environment variables when configuration
is first imported. 

.. list-table:: Settings
   :header-rows: 1

   * - Setting (append uppercase name to ``AUTO_UNCERTAINTIES_``)
     - Default
   * - ``round_on_display``
     - ``True``
   * - ``error_on_downcast``
     - ``False``
   * - ``error_on_equality``
     - ``False``
   * - ``compare_rtol``
     - ``1e-9``
   * - ``error_on_numpy_kwargs``
     - ``True``
   * - ``warn_on_numpy_kwargs``
     - ``True``
   * - ``nonfinite_error_policy``
     - ``"zero"``

Comparison tolerance must be finite and nonnegative. ``error_on_numpy_kwargs`` and
``warn_on_numpy_kwargs`` remain accepted for configuration compatibility and no
longer affect propagation. Uncertain keyword arguments are supported.

.. code-block:: bash

   export AUTO_UNCERTAINTIES_NONFINITE_ERROR_POLICY=propagate
   export AUTO_UNCERTAINTIES_ROUND_ON_DISPLAY=true

.. code-block:: python

   >>> from auto_uncertainties import Uncertainty, configure, get_settings
   >>> configure(nonfinite_error_policy="propagate")
   >>> get_settings().nonfinite_error_policy
   'propagate'
   >>> Uncertainty(1.0, float("nan"))
   nan +/- nan

The default ``"zero"`` policy replaces non-finite errors with zero. Under
``"propagate"``, NaN in either component sets both to NaN, element by element for
arrays. Positive infinite errors are preserved; negative errors, including negative
infinity, raise ``NegativeStdDevError``. Omitted errors remain zero for valid values.
Construction and array assignment use the same policy. Non-finite components are
displayed explicitly, even with rounding enabled.

Runtime updates are validated before any changes are applied. They affect subsequent
operations and do not rewrite existing objects. Later changes to environment variables
do not automatically update configuration. Settings are shared across the process.

The public ``set_display_rounding``, ``set_downcast_error``, ``set_equality_error``,
``set_compare_rtol``, and ``exceptions.set_numpy_kwargs_propagate_error`` functions
have been removed. Replace calls with ``configure(round_on_display=...)``,
``configure(error_on_downcast=...)``, ``configure(error_on_equality=...)``,
``configure(compare_rtol=...)``, and ``configure(error_on_numpy_kwargs=...)``,
respectively.
