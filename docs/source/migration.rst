Migrating from 1.2
==================

These changes are intended for the next major release. Python 3.15 or newer is
required. Runtime, documentation, development, and build dependency minimums have
been raised to the stable releases recorded in ``pyproject.toml``. The lockfile
records the complete tested dependency set.

Public API
----------

Import ``Uncertainty``, ``nominal_values``, ``std_devs``,
``propagate_uncertainties``, configuration functions, and exceptions from
``auto_uncertainties`` as before. Automatic Pint quantity construction remains
supported; Pint remains optional.

Package exports contain only the supported symbols. Implementation modules and
NumPy dispatch registries are no longer re-exported as public API.

Internal modules, helpers, and storage attributes now use names without a leading
underscore. For example, ``_propagation`` is now ``propagation``, and the container
stores its components in ``nom`` and ``err``. These remain implementation details;
use the package exports and the ``value`` and ``error`` properties in application
code.

* Replace ``ScalarUncertainty`` and ``VectorUncertainty`` with ``Uncertainty``.
  Scalars and arrays continue to use the same class.
* Remove the ``implement_mode`` argument from ``propagate_uncertainties``.
  Replace ``@propagate_uncertainties()`` with ``@propagate_uncertainties``.
  Wrapping an existing function with ``propagate_uncertainties(func)`` remains
  supported.
* Replace ``Uncertainty.from_list(values, errors)`` with
  ``Uncertainty(values, errors)``. ``from_sequence`` and ``from_quantities`` remain.
* Remove ``skip`` from direct initializer calls. Construction initializes the
  object once and validates its components.
* Remove ``error_on_numpy_kwargs`` and ``warn_on_numpy_kwargs`` configuration
  arguments and environment variables. Argument validation is always enabled.
* The implementation's display classes, ``P``, ``R``, and
  ``elementwise_value_and_grad`` are no longer public exports. Use the ordinary
  object representations and ``propagate_uncertainties`` instead.

Array methods
-------------

Array methods are declared explicitly and have inspectable, annotated signatures.
Supported methods include ``reshape``, ``transpose``, ``ravel``, ``flatten``,
``astype``, ``squeeze``, ``swapaxes``, ``diagonal``, ``repeat``, ``take``,
``compress``, ``sum``, ``mean``, ``std``, ``var``, ``min``, ``max``, ``argmin``,
``argmax``, ``round``, ``sort``, ``searchsorted``, ``clip``, and ``conjugate``
(``conj``). Existing copying, indexing, and mutation methods remain available.

Other registered NumPy operations no longer appear as generated methods. Replace
``u.sin()`` with ``np.sin(u)``, ``u.add(x)`` with ``np.add(u, x)``, and
``u.median()`` with ``np.median(u)``. Metadata such as ``ndim`` is a property;
use ``u.ndim`` or ``np.ndim(u)``.

``sort`` continues to return a sorted copy. ``reshape`` and ``transpose`` accept
either a tuple or separate dimensions. Output arguments return the supplied
destination and update both values and errors only after validation. Nominal and
error destination buffers must not overlap one another; destinations may overlap
input storage.

Arithmetic
----------

Addition, subtraction, multiplication, division, and power use the same analytic
propagation rules through Python operators and NumPy. Nominal results follow
NumPy, including division by zero. Error calculations avoid unnecessary
overflow and underflow, and a zero derivative contributes zero even if its input
error is infinite. The configured nonfinite-error policy still applies to results.

For example, under ``nonfinite_error_policy="propagate"``,
``Uncertainty(2, float("inf")) * 0`` produces zero with zero error, and
``Uncertainty(2, 0.1) / 0`` produces infinite value and error. The latter follows
NumPy's floating-point handling instead of Python's ``ZeroDivisionError``.

Floor division, modulo, comparison policies, and the independence model retain
their previous behavior. Separate uncertain input occurrences are independent;
reuse of one parameter within a decorated calculation preserves that dependence.
Returned objects do not track covariance across later calculations.
