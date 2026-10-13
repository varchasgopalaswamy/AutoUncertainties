# Based heavily on the implementation of Pint's Quantity object
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
import copy
from functools import wraps
import locale
import math
import operator
from typing import (
    Any,
    Literal,
    Self,
    SupportsFloat,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)
import warnings
lazy from types import NotImplementedType

import numpy as np
import numpy.typing as npt
lazy from pint.facets.plain import PlainQuantity

from auto_uncertainties.config import get_settings
from auto_uncertainties.display_format import UncertaintyDisplay
from auto_uncertainties.exceptions import (
    DowncastError,
    DowncastWarning,
    EqualityError,
    EqualityWarning,
    NegativeStdDevError,
)
from auto_uncertainties.numpy import HANDLED_FUNCTIONS, HANDLED_UFUNCS, wrap_numpy
from auto_uncertainties.util import deprecated, ignore_runtime_warnings

__all__ = [
    "ScalarUncertainty",
    "UType",
    "Uncertainty",
    "VectorUncertainty",
    "nominal_values",
    "std_devs",
]

_F = TypeVar("_F", bound=Callable[..., Any])
T = TypeVar("T", float, np.floating, npt.NDArray[np.floating])
"""`TypeVar` specifying the underlying data types supporting `Uncertainty` objects."""

UType: TypeAlias = float | np.floating | npt.NDArray[np.floating]
"""Type alias for the underlying data types supporting `Uncertainty` objects."""

ScalarT: TypeAlias = int | float | np.number
"""Scalar types used throughout AutoUncertainties."""

type SupportedSequence = Sequence[ScalarT | Uncertainty[float] | Uncertainty[np.floating]]
"""Sequences supported by the `Uncertainty` constructor."""

type ValT = ScalarT | SupportedSequence | Uncertainty[Any] | npt.NDArray[np.number]
"""Types supported in the ``value`` parameter of the `Uncertainty` constructor."""

ErrT: TypeAlias = ScalarT | Sequence[ScalarT] | npt.NDArray[np.number]
"""Types supported in the ``error`` parameter of the `Uncertainty` constructor."""


# Helper decorator to raise an informative TypeError when an operation is unsupported.
def _unsupported_type(t: str) -> Callable[[_F], _F]:
    """
    Restrict a method to scalar or vector uncertainties.

    :param t: Kind to reject: ``"scal"`` for scalars or ``"vec"`` for vectors.
    :return: A decorator that checks the receiver before calling the method.
    :raises TypeError: The decorated method is called on a rejected kind or the kind
        selector is invalid.
    """

    def decorator(func: _F) -> _F:
        @wraps(func)
        def wrapper(self: Uncertainty[Any], *args: Any, **kwargs: Any) -> Any:
            if t == "vec":
                if self.is_vector:
                    raise TypeError(_type_error_msg("Vector", func.__qualname__))
            elif t == "scal":
                if not self.is_vector:
                    raise TypeError(_type_error_msg("Scalar", func.__qualname__))
            else:
                msg = "Invalid arguments passed to _unsupported_type decorator"
                raise TypeError(msg)
            return func(self, *args, **kwargs)

        return cast("_F", wrapper)

    return decorator


def _power_uncertainty(
    value: npt.ArrayLike,
    error: npt.ArrayLike,
    exponent: npt.ArrayLike,
    exponent_error: npt.ArrayLike,
) -> tuple[np.number | npt.NDArray[np.number], np.number | npt.NDArray[np.number]]:
    """
    Evaluate a real power and propagate independent errors at the nominal values.

    :param value: Nominal base values.
    :param error: Base standard deviations.
    :param exponent: Nominal exponents.
    :param exponent_error: Exponent standard deviations.
    :return: The nominal power and its propagated standard deviation, with broadcast
        operand shapes.
    """
    with np.errstate(all="ignore"):
        # Preserve NumPy's scalar promotion rules before broadcasting operands.
        nominal = np.power(value, exponent)
        dtype = np.result_type(nominal, error, exponent_error)
        value, error, exponent, exponent_error = np.broadcast_arrays(
            value, error, exponent, exponent_error
        )
        base_derivative = np.where(exponent == 0, 0, exponent * np.power(value, exponent - 1))
        exponent_derivative = np.where(
            (value == 0) & (exponent > 0),
            0,
            nominal * np.log(np.abs(value)),
        )
        base_term = np.zeros(nominal.shape, dtype=dtype)
        exponent_term = np.zeros(nominal.shape, dtype=dtype)
        # Exact operands and zero derivatives contribute no error, including at
        # boundaries or when the corresponding standard deviation is infinite.
        np.multiply(
            base_derivative,
            error,
            out=base_term,
            where=(error != 0) & (base_derivative != 0),
        )
        np.multiply(
            exponent_derivative,
            exponent_error,
            out=exponent_term,
            where=(exponent_error != 0) & (exponent_derivative != 0),
        )
        propagated = np.hypot(base_term, exponent_term)
    if nominal.ndim == 0:
        return nominal[()], propagated[()]
    return nominal, propagated


class Uncertainty(UncertaintyDisplay[T]):
    """
    Represent a central value and its associated uncertainty.

    Parameters can be numbers, sequences, `numpy` arrays, `pint.Quantity` objects,
    other `Uncertainty` objects, or lists / tuples of `Uncertainty` objects.

    The `Uncertainty` class automatically determines which methods should be implemented based on
    whether it represents a vector uncertainty, or a scalar uncertainty. When instantiated with
    sequences or `numpy` arrays, vector-based operations are enabled; when instantiated with scalars,
    only scalar operations are permitted.

    `Uncertainty` objects only support float-based data types. If integers or
    integer arrays are passed as parameters to the `Uncertainty` constructor,
    they will be cast to `float` (or `numpy.float64` if a `numpy.integer` subclass
    is detected).

    :param value: Central values or an existing uncertainty.
    :param error: Standard deviations; omitted errors are zero. Existing
        uncertainties in ``value`` supply their own errors.
    :return: An initialized uncertainty, wrapped in a quantity when units are supplied.
    :raises NegativeStdDevError: An error is negative under the configured
        nonfinite policy.
    :raises TypeError: The parameter types are unsupported or incompatible.
    :raises ValueError: The value and error shapes or sequence lengths differ.

    .. code-block:: python
       :caption: Example

       >>> u1 = Uncertainty(1.25, 0.25)
       >>> u2 = Uncertainty([1.4, 2.8, 0.09], [0.1, 0.14, 0.12])
       >>> u3 = Uncertainty([1.4, 2.8, 0.09], 0.1)
       >>> u4 = Uncertainty(u1)
       >>> u5 = Uncertainty(np.array([1.4, 2.8]), np.array([0.1, 0.14]))
       >>> u6 = Uncertainty(np.array([1.4, 2.8]), 0.1)

       >>> u3.value
       array([1.4 , 2.8 , 0.09])

       >>> u3.error
       array([0.1, 0.1, 0.1])

       >>> np.cos(u1)
       0.32 +/- 0.24

       >>> u4 == u1
       True

    .. code-block:: python
       :caption: Pint Quantity Example

       >>> from pint import Quantity
       >>> val = Quantity(2.24, 'kg')
       >>> err = Quantity(0.208, 'kg')
       >>> new_quantity = Uncertainty(val, err)
       >>> new_quantity
       <Quantity(2.24 +/- 0.21, 'kilogram')>

    .. note::

       * If sequences (not NumPy arrays) are supplied for ``value`` and ``error``,
         their numeric values will always be converted to `numpy.float64`.

       * If `pint.Quantity` objects are supplied for either parameter, the behavior
         is exactly as described in the `from_quantities` method.

       * If an `Uncertainty` is supplied for ``value``, its ``error`` attribute will
         always override any ``error`` argument (if it is supplied).

       * The default ``nonfinite_error_policy="zero"`` replaces non-finite errors
         with zero. With ``"propagate"``, a NaN in either component sets both
         components to NaN, and positive infinite errors are preserved.

    .. seealso::

        * `from_quantities`
    """

    _nom: T
    _err: T

    # __new__ intercepts non-finite values, Pint Quantity inputs, and sequences of Quantity objects.
    @overload
    def __new__(
        cls,
        value: PlainQuantity | Sequence[PlainQuantity],
        error: PlainQuantity | Sequence[PlainQuantity] | ErrT | None = ...,
    ) -> PlainQuantity: ...
    @overload
    def __new__(
        cls, value: ValT, error: PlainQuantity | Sequence[PlainQuantity]
    ) -> PlainQuantity: ...
    @overload
    def __new__(cls, value: ValT, error: ErrT | None = ...) -> Uncertainty[Any]: ...
    def __new__(
        cls,
        value: ValT | PlainQuantity | Sequence[PlainQuantity],
        error: ErrT | PlainQuantity | Sequence[PlainQuantity] | None = None,
    ) -> Uncertainty[Any] | PlainQuantity:
        # Use from_quantities if one or more Pint Quantity objects were supplied.
        """
        Construct an uncertainty, preserving quantity units where present.

        :param value: Central values or an existing uncertainty.
        :param error: Standard deviations, or ``None`` for exact values.
        :return: An initialized uncertainty, wrapped in a quantity when the inputs
            supply units.
        """
        if _check_units(value, error)[2] is not None:
            return cls.from_quantities(value, error)

        # Use from_sequence if a sequence is supplied.
        if isinstance(value, Sequence):
            return cls.from_sequence(value, error)

        instance = super().__new__(cls)
        instance.__init__(cast("ValT", value), cast("ErrT | None", error), skip=False)
        return instance

    # List of __init__ overloads for static type checking.
    @overload
    def __init__(self: Uncertainty[float], value: int, error: ErrT | None = ...) -> None: ...
    @overload
    def __init__(
        self: Uncertainty[np.float64], value: np.integer, error: ErrT | None = ...
    ) -> None: ...
    @overload
    def __init__(
        self: Uncertainty[npt.NDArray[np.floating]],
        value: npt.NDArray[np.integer] | SupportedSequence,
        error: ErrT | None = ...,
    ) -> None: ...
    @overload
    def __init__(self: Uncertainty[float], value: Uncertainty[float]) -> None: ...
    @overload
    def __init__(self: Uncertainty[np.floating], value: Uncertainty[np.floating]) -> None: ...
    @overload
    def __init__(
        self: Uncertainty[npt.NDArray[np.floating]], value: Uncertainty[npt.NDArray[np.floating]]
    ) -> None: ...
    @overload
    def __init__(self: Self, value: T, error: ErrT | None = ...) -> None: ...
    @overload
    def __init__(self, value: ValT, error: ErrT | None = ..., skip: bool = ...) -> None: ...  # noqa: FBT001
    # Preserve the historical positional initializer flag for compatibility.
    def __init__(self, value: ValT, error: ErrT | None = None, skip: bool = True) -> None:  # noqa: FBT001, FBT002
        """
        Initialize and validate nominal values and standard deviations.

        Construction normally initializes the instance in :meth:`__new__`; the
        subsequent automatic initializer call is skipped.

        :param value: Central values or an uncertainty whose values and errors are
            reused.
        :param error: Standard deviations; omitted errors are zero and existing
            uncertainties supply their own errors.
        :param skip: Leave the initialized instance unchanged; internal construction
            passes ``False``.
        :return: ``None``.
        :raises NegativeStdDevError: An error is negative under the configured nonfinite
            policy.
        :raises TypeError: The input types are unsupported or incompatible.
        :raises ValueError: Value and error shapes or sequence lengths differ.
        """
        if skip:
            return

        # Avoid zero-dimensional arrays.
        value = value[()] if (isinstance(value, np.ndarray) and np.ndim(value) == 0) else value
        error = error[()] if (isinstance(error, np.ndarray) and np.ndim(error) == 0) else error

        # Case where __init__ acts as a sort of copy constructor.
        if isinstance(value, Uncertainty):
            self.__init__(value.value, value.error, skip=False)

        # Case where a sequence of values was passed.
        elif isinstance(value, Sequence):
            if (
                error is not None
                and not isinstance(error, Sequence)
                and not isinstance(error, ScalarT)
            ):
                msg = f"Error must be a sequence or scalar when value is a sequence (got {type(error)} instead)"
                raise TypeError(msg)
            self._init_seq(value, error)

        # Case where a vector uncertainty is instantiated.
        elif isinstance(value, np.ndarray):
            if (
                error is not None
                and not isinstance(error, np.ndarray)
                and not isinstance(error, ScalarT)
            ):
                msg = f"Error must be a NumPy array or scalar when value is a NumPy array (got {type(error)} instead)"
                raise TypeError(msg)
            self._init_vec(value, error)

        # Scalar case. Maintains NumPy data types if detected. Converts ints to floating point.
        elif isinstance(value, ScalarT):
            if error is not None and not isinstance(error, ScalarT):
                msg = f"Error must be a scalar when value is a scalar (got {type(error)} instead)"
                raise TypeError(msg)
            propagate = get_settings().nonfinite_error_policy == "propagate"
            if error is not None and error < 0 and (np.isfinite(error) or propagate):
                msg = f"Got negative value ({error}) for the standard deviation"
                raise NegativeStdDevError(msg)

            caster = np.float64 if isinstance(value, np.number) else float
            if isinstance(value, int | np.integer):
                self._nom = cast("T", caster(value))
                self._err = cast(
                    "T",
                    caster(error)
                    if (error is not None and (np.isfinite(error) or propagate))
                    else caster(0.0),
                )
            else:
                caster = (
                    (lambda x: x)
                    if isinstance(error, np.floating) and isinstance(value, np.floating)
                    else caster
                )
                self._nom = cast("T", value)
                self._err = cast(
                    "T",
                    caster(error)
                    if (error is not None and (np.isfinite(error) or propagate))
                    else caster(0.0),
                )

            if propagate and (np.isnan(self._nom) or np.isnan(self._err)):
                self._nom = cast("T", type(self._nom)(np.nan))
                self._err = cast("T", type(self._err)(np.nan))

        else:
            msg = f"Unsupported argument types (got type(value)={type(value)}, type(error)={type(error)})"
            raise TypeError(msg)

    def _init_seq(
        self, value: SupportedSequence, error: Sequence[ScalarT] | ScalarT | None
    ) -> None:
        """
        Initialize array storage from a sequence of values or uncertainties.

        :param value: Sequence of central values or scalar uncertainties.
        :param error: Matching errors, a shared error, or ``None`` for zero errors;
            uncertain elements supply their own errors.
        :return: ``None``.
        :raises TypeError: A sequence element has an unsupported type.
        :raises ValueError: The value and error sequence lengths differ.
        """
        if isinstance(error, Sequence) and len(error) != len(value):
            msg = f"Error sequence must be the same length as value sequence (got len(value)={len(value)}, len(error)={len(error)})"
            raise ValueError(msg)

        val = np.empty(len(value), dtype=np.float64)
        err = np.empty(len(value), dtype=np.float64)

        if len(value) > 0:
            error = 0.0 if error is None else error
            reshaped_error = (
                (np.ones(len(value), dtype=np.float64) * error)
                if isinstance(error, ScalarT)
                else error
            )
            for i, v in enumerate(value):
                if isinstance(v, Uncertainty):
                    val[i] = v.value
                    err[i] = v.error
                elif isinstance(v, ScalarT):
                    val[i] = v
                    if not isinstance(reshaped_error[i], ScalarT):
                        msg = f"Error sequence must be of scalars (found element of type {type(err[i])} instead)"
                        raise TypeError(msg)
                    err[i] = reshaped_error[i]
                else:
                    msg = f"Value sequence must be of scalars or Uncertainty objects (found element of type {type(v)} instead)"
                    raise TypeError(msg)

        self.__init__(val, err, skip=False)

    def _init_vec(
        self,
        value: npt.NDArray[np.number],
        error: npt.NDArray[np.number] | ScalarT | None,
    ) -> None:
        # Zero error.
        """
        Initialize array storage and normalize errors according to the settings.

        :param value: Array of central values.
        :param error: Matching error array, shared error, or ``None`` for zero errors.
        :return: ``None``.
        :raises ValueError: The value and error arrays have different shapes.
        :raises NegativeStdDevError: An error remains negative after normalization.
        """
        if error is None:
            error = np.zeros_like(value)

        # Constant error.
        elif isinstance(error, ScalarT):
            error = np.ones_like(value) * error

        elif np.ndim(error) != np.ndim(value) or np.shape(error) != np.shape(value):
            msg = f"Error must have the same shape as value (got value.shape={np.shape(value)}, error.shape={np.shape(error)})"
            raise ValueError(msg)

        # Normalize on copies only when needed, preserving caller-owned arrays.
        if get_settings().nonfinite_error_policy == "zero":
            invalid = ~np.isfinite(error)
            if np.any(invalid):
                error = np.where(invalid, 0, error)

        if np.any(error < 0):
            msg = f"Found {np.count_nonzero(error < 0)} negative values for the standard deviation"
            raise NegativeStdDevError(msg)

        # Convert int data to float
        if issubclass(value.dtype.type, np.integer):
            value = value.astype(np.float64)
        if issubclass(error.dtype.type, np.integer):
            error = error.astype(np.float64)

        if get_settings().nonfinite_error_policy == "propagate":
            invalid = np.isnan(value) | np.isnan(error)
            if np.any(invalid):
                value = np.where(invalid, np.nan, value)
                error = np.where(invalid, np.nan, error)

        self._nom = cast("T", value)
        self._err = cast("T", error)

    @property
    def is_vector(self) -> bool:
        """
        Check whether the uncertainty stores arrays.

        :return: Whether both nominal values and errors are NumPy arrays.
        """
        return isinstance(self._nom, np.ndarray) and isinstance(self._err, np.ndarray)

    @property
    def value(self) -> T:
        """
        Access the central values.

        :return: The stored nominal values.
        """
        return self._nom

    @property
    def error(self) -> T:
        """
        Access the absolute uncertainties.

        :return: The stored standard deviations.
        """
        return self._err

    @property
    def relative(self) -> T:
        """
        Compute uncertainty relative to the nominal magnitude.

        :return: The error divided by the absolute nominal value, with NaN at zero
            nominal values.
        """
        # Vector uncertainty
        if isinstance(self._nom, np.ndarray) and isinstance(self._err, np.ndarray):
            rel = np.zeros_like(self._nom)
            inf = np.isinf(self._nom)
            nan = self._nom == 0
            valid = ~inf & ~nan
            rel[valid] = self._err[valid] / np.abs(self._nom[valid])
            rel[inf] = np.inf
            rel[nan] = np.nan
            return rel

        # Scalar uncertainty
        try:
            return cast("T", self._err / abs(self._nom))
        except OverflowError:
            return cast("T", float("inf") if isinstance(self._err, float) else np.inf)
        except ZeroDivisionError:
            return cast("T", float("nan") if isinstance(self._err, float) else np.nan)

    @property
    def rel(self) -> T:
        """
        Access the relative uncertainty through its short alias.

        :return: The same result as :attr:`relative`.
        """
        return self.relative

    @property
    def rel2(self) -> T:
        """
        Compute the squared relative uncertainty.

        :return: The square of :attr:`relative`.
        """
        if self.is_vector:
            return cast("T", self.relative**2)
        try:
            return cast("T", self.relative**2)
        except OverflowError:
            return cast("T", float("inf") if isinstance(self._err, float) else np.inf)

    def plus_minus(self, error: UType) -> Uncertainty[T]:
        """
        Combine an additional independent error with the existing uncertainty.

        :param error: Additional standard deviation to combine in quadrature.
        :return: A new uncertainty with unchanged nominal values and the combined error.
        """
        old_err: T = self._err
        new_err: T
        result = np.sqrt(old_err**2 + error**2)

        # Check if float, NumPy float, or NumPy array.
        new_err = cast("T", float(result)) if isinstance(self._err, float) else cast("T", result)

        return self.__class__(self._nom, new_err)

    @classmethod
    def from_string(cls, string: str) -> Uncertainty:
        """
        Parse a nominal value and an optional error.

        :param string: Value optionally followed by ``+/-``, ``+-``, or ``±`` and an
            error, such as ``5.0 +/- 1.0``.
        :return: The parsed scalar uncertainty, with zero error if none is supplied.
        :raises ValueError: The components cannot be parsed as numbers.
        :raises NegativeStdDevError: The parsed error is negative under the configured
            nonfinite policy.
        """
        new_str = string.replace("+/-", "±")
        new_str = new_str.replace("+-", "±")
        if "±" not in new_str:
            return cls(float(string))
        else:
            u1, u2 = new_str.split("±")
            return cls(float(u1), float(u2))

    @classmethod
    def from_quantities(
        cls,
        value: ValT | PlainQuantity | Sequence[PlainQuantity],
        error: ErrT | PlainQuantity | Sequence[PlainQuantity] | None = None,
    ) -> Uncertainty[Any] | PlainQuantity:
        """
        Construct an uncertainty from quantities with compatible units.

        Errors carrying units are converted to the nominal value's units. A bare
        error is interpreted in the nominal value's units. If only the error carries
        units, the bare nominal value must be convertible from dimensionless units.
        Inputs without units produce an ordinary uncertainty.

        .. note::

           Calling ``Uncertainty(value, error)`` selects this constructor automatically
           when a quantity is supplied and provides more precise static type inference.

        :param value: Central values, optionally carrying units.
        :param error: Standard deviations, optionally carrying compatible units; omitted
            errors are zero.
        :return: An uncertainty wrapped in the selected units, or an ordinary
            uncertainty for unitless inputs.
        :raises pint.errors.DimensionalityError: The nominal values and errors have
            incompatible units.
        """
        value_, error_, units = _check_units(value, error)
        instance = cls(value_, error_)
        if units is not None:
            instance *= units
        return instance

    @classmethod
    def from_sequence(
        cls,
        value: ValT | Sequence[PlainQuantity],
        error: ErrT | PlainQuantity | Sequence[PlainQuantity] | None = None,
    ) -> Uncertainty[Any] | PlainQuantity:
        """
        Construct an uncertainty from a sequence, normalizing quantity units.

        When quantities are present, values are converted to the first value's units.
        Error quantities use those units when available, otherwise the first error's
        units. Existing uncertain elements supply their own errors.

        .. note::

           Calling ``Uncertainty(value, error)`` selects this constructor automatically
           for sequences and provides more precise static type inference.

        :param value: Sequence of central values, uncertainties, or compatible
            quantities.
        :param error: Matching errors, a shared error, or ``None`` for zero errors.
        :return: The constructed uncertainty, wrapped in quantity units when present.
        """
        if not isinstance(value, Sequence):
            return cls(cast("ValT", value), cast("ErrT | None", error))

        mag_units = err_units = None
        if len(value) > 0 and hasattr(value[0], "units"):
            # Converts all values to the same unit.
            mag_units = value[0].units
            value = cast(
                "ValT",
                [
                    cast("PlainQuantity", item).to(mag_units).m if hasattr(item, "units") else item
                    for item in value
                ],
            )
        if isinstance(error, Sequence) and len(error) > 0 and hasattr(error[0], "units"):
            # Convert all error units to value units, if possible. Otherwise, make sure all errors use same units.
            err_units = error[0].units
            to_units = err_units if mag_units is None else mag_units
            error = cast(
                "ErrT",
                [
                    cast("PlainQuantity", item).to(to_units).m if hasattr(item, "units") else item
                    for item in error
                ],
            )

        instance = super().__new__(cls)
        instance.__init__(cast("ValT", value), cast("ErrT | None", error), skip=False)
        units = err_units if mag_units is None else mag_units
        return instance if units is None else (instance * units)

    @classmethod
    @deprecated("call Uncertainty() directly instead")
    def from_list(cls, value: ValT, error: ErrT | None = None) -> Uncertainty:  # pragma: no cover
        """
        Construct an uncertainty using the deprecated sequence constructor alias.

        .. warning::

           This method is deprecated. Call ``Uncertainty(value, error)`` directly instead.

        :param value: Central values or existing uncertainties.
        :param error: Standard deviations, or ``None`` for zero errors.
        :return: The constructed uncertainty.
        """
        return cls(value, error)

    def __getstate__(self) -> dict[str, T]:
        """
        Collect the state used to serialize an uncertainty.

        :return: The nominal values and errors under ``nominal_value`` and ``std_devs``.
        """
        return {"nominal_value": self._nom, "std_devs": self._err}

    def __setstate__(self, state: dict[str, T]) -> None:
        """
        Restore previously serialized nominal values and errors.

        :param state: State containing the ``nominal_value`` and ``std_devs`` entries.
        :return: ``None``.
        """
        self._nom = state["nominal_value"]
        self._err = state["std_devs"]

    def __getnewargs__(self) -> tuple[T, T]:
        """
        Provide constructor arguments for deserialization.

        :return: The nominal values and errors, in constructor order.
        """
        return self._nom, self._err

    def __copy__(self) -> Uncertainty[T]:
        """
        Copy the nominal values and errors.

        :return: A new uncertainty with shallow copies of both components.
        """
        return self.__class__(copy.copy(self._nom), copy.copy(self._err))

    def __deepcopy__(self, memo: dict[int, Any]) -> Uncertainty[T]:
        """
        Deep-copy the nominal values and errors.

        :param memo: Objects already copied during the current deep-copy operation.
        :return: A new uncertainty with recursively copied components.
        """
        return self.__class__(copy.deepcopy(self._nom, memo), copy.deepcopy(self._err, memo))

    # =====================================================
    # ------------------ MATH OPERATIONS ------------------
    # =====================================================

    _HANDLED_TYPES = (float, int, np.ndarray)

    def __add__(self, other: Any) -> Uncertainty[Any] | NotImplementedType:
        """
        Evaluate ``self + other`` with uncertainty propagation.

        Independent errors are combined in quadrature.

        :param other: The other operand, which may be exact or uncertain.
        :return: The resulting uncertainty, or ``NotImplemented`` for unsupported
            operands.
        """
        if isinstance(other, Uncertainty):
            new_mag = self._nom + other._nom
            new_err = np.sqrt(self._err**2 + other._err**2)
        elif isinstance(other, self._HANDLED_TYPES):
            new_mag = self._nom + other
            new_err = (
                np.broadcast_to(self._err, np.shape(new_mag)).copy()
                if np.ndim(new_mag)
                else self._err
            )
        else:
            return NotImplemented
        try:
            return self.__class__(new_mag, new_err)
        except NotImplementedError:
            return NotImplemented

    __radd__ = __add__

    def __sub__(self, other: Any) -> Uncertainty[Any] | NotImplementedType:
        """
        Evaluate ``self - other`` with uncertainty propagation.

        Independent errors are combined in quadrature.

        :param other: The other operand, which may be exact or uncertain.
        :return: The resulting uncertainty, or ``NotImplemented`` for unsupported
            operands.
        """
        if isinstance(other, Uncertainty):
            new_mag = self._nom - other._nom
            new_err = np.sqrt(self._err**2 + other._err**2)
        elif isinstance(other, self._HANDLED_TYPES):
            new_mag = self._nom - other
            new_err = (
                np.broadcast_to(self._err, np.shape(new_mag)).copy()
                if np.ndim(new_mag)
                else self._err
            )
        else:
            return NotImplemented
        try:
            return self.__class__(new_mag, new_err)
        except NotImplementedError:
            return NotImplemented

    def __rsub__(self, other: Any) -> Uncertainty[Any] | NotImplementedType:
        """
        Evaluate ``other - self`` with uncertainty propagation.

        Independent errors are combined in quadrature.

        :param other: The other operand, which may be exact or uncertain.
        :return: The resulting uncertainty, or ``NotImplemented`` for unsupported
            operands.
        """
        return -self.__sub__(other)

    def __mul__(self, other: Any) -> Uncertainty[Any] | NotImplementedType:
        """
        Evaluate ``self * other`` with uncertainty propagation.

        Independent errors are propagated through the product.

        :param other: The other operand, which may be exact or uncertain.
        :return: The resulting uncertainty, or ``NotImplemented`` for unsupported
            operands.
        """
        if isinstance(other, Uncertainty):
            new_mag = self._nom * other._nom
            new_err = np.sqrt((other._nom * self._err) ** 2 + (self._nom * other._err) ** 2)
        elif isinstance(other, self._HANDLED_TYPES):
            new_mag = self._nom * other
            new_err = np.abs(self._err * other)
        else:
            return NotImplemented
        try:
            return self.__class__(new_mag, new_err)
        except NotImplementedError:
            return NotImplemented

    __rmul__ = __mul__

    @ignore_runtime_warnings
    def __truediv__(self, other: Any) -> Uncertainty[Any] | NotImplementedType:
        """
        Evaluate ``self / other`` with uncertainty propagation.

        Independent errors are propagated through the quotient.

        :param other: The other operand, which may be exact or uncertain.
        :return: The resulting uncertainty, or ``NotImplemented`` for unsupported
            operands.
        """
        if isinstance(other, Uncertainty):
            new_mag = self._nom / other._nom
            new_err = np.sqrt(
                (self._err / other._nom) ** 2 + (self._nom * other._err / other._nom**2) ** 2
            )
        elif isinstance(other, self._HANDLED_TYPES):
            new_mag = self._nom / other
            new_err = np.abs(self._err / other)
        else:
            return NotImplemented
        try:
            return self.__class__(new_mag, new_err)
        except NotImplementedError:
            return NotImplemented

    @ignore_runtime_warnings
    def __rtruediv__(self, other: Any) -> Uncertainty[Any] | NotImplementedType:
        # Other / Self
        """
        Evaluate ``other / self`` with uncertainty propagation.

        Independent errors are propagated through the quotient.

        :param other: The other operand, which may be exact or uncertain.
        :return: The resulting uncertainty, or ``NotImplemented`` for unsupported
            operands.
        """
        if isinstance(other, Uncertainty):
            raise TypeError
        if isinstance(other, self._HANDLED_TYPES):
            new_mag = other / self._nom
            new_err = np.abs(new_mag) * np.abs(self.rel)
        else:
            return NotImplemented
        try:
            return self.__class__(new_mag, new_err)
        except NotImplementedError:
            return NotImplemented

    __div__ = __truediv__
    __rdiv__ = __rtruediv__

    def __floordiv__(self, other: Any) -> Uncertainty[Any] | NotImplementedType:
        """
        Evaluate ``self // other`` with uncertainty propagation.

        The nominal quotient is floored while retaining the propagated division error.

        :param other: The other operand, which may be exact or uncertain.
        :return: The resulting uncertainty, or ``NotImplemented`` for unsupported
            operands.
        """
        if isinstance(other, Uncertainty):
            new_mag = self._nom // other._nom
        elif isinstance(other, self._HANDLED_TYPES):
            new_mag = self._nom // other
        else:
            return NotImplemented
        new_err = self.__div__(other).error

        return self.__class__(new_mag, new_err)

    def __rfloordiv__(self, other: Any) -> Uncertainty[Any] | NotImplementedType:
        """
        Evaluate ``other // self`` with uncertainty propagation.

        The nominal quotient is floored while retaining the propagated division error.

        :param other: The other operand, which may be exact or uncertain.
        :return: The resulting uncertainty, or ``NotImplemented`` for unsupported
            operands.
        """
        if isinstance(other, Uncertainty):
            return other.__floordiv__(self)
        elif isinstance(other, self._HANDLED_TYPES):
            new_mag = other // self._nom
            new_err = self.__rdiv__(other).error
            return self.__class__(new_mag, new_err)
        else:
            return NotImplemented

    def __mod__(self, other: Any) -> Uncertainty[Any] | NotImplementedType:
        """
        Evaluate ``self % other`` with uncertainty propagation.

        Errors are propagated with the nominal integer quotient held fixed locally.

        :param other: The other operand, which may be exact or uncertain.
        :return: The resulting uncertainty, or ``NotImplemented`` for unsupported
            operands.
        """
        if isinstance(other, Uncertainty):
            new_mag = self._nom % other._nom
            new_err = np.hypot(self._err, (self._nom // other._nom) * other._err)
        elif isinstance(other, self._HANDLED_TYPES):
            new_mag = self._nom % other
            new_err = (
                np.broadcast_to(self._err, np.shape(new_mag)).copy()
                if np.ndim(new_mag)
                else self._err
            )
        else:
            return NotImplemented
        return self.__class__(new_mag, new_err)

    def __rmod__(self, other: Any) -> Uncertainty[Any] | NotImplementedType:
        """
        Evaluate ``other % self`` with uncertainty propagation.

        Errors are propagated with the nominal integer quotient held fixed locally.

        :param other: The other operand, which may be exact or uncertain.
        :return: The resulting uncertainty, or ``NotImplemented`` for unsupported
            operands.
        """
        if isinstance(other, self._HANDLED_TYPES):
            new_mag = other % self._nom
            new_err = np.abs((other // self._nom) * self._err)
            return self.__class__(new_mag, new_err)
        else:
            return NotImplemented

    def __divmod__(
        self, other: Any
    ) -> tuple[Uncertainty[Any], Uncertainty[Any]]:  # pragma: no cover
        """
        Evaluate ``divmod(self, other)`` with uncertainty propagation.

        :param other: The other operand, which may be exact or uncertain.
        :return: The floor quotient and remainder, each with its propagated error.
        """
        return self // other, self % other

    def __rdivmod__(
        self, other: Any
    ) -> tuple[Uncertainty[Any], Uncertainty[Any]]:  # pragma: no cover
        """
        Evaluate ``divmod(other, self)`` with uncertainty propagation.

        :param other: The other operand, which may be exact or uncertain.
        :return: The floor quotient and remainder, each with its propagated error.
        """
        return other // self, other % self

    @ignore_runtime_warnings
    def __pow__(self, other: Any) -> Uncertainty[Any] | NotImplementedType:
        # Self ** other
        """
        Evaluate ``self ** other`` with uncertainty propagation.

        Independent base and exponent errors are propagated at the nominal values.

        :param other: The other operand, which may be exact or uncertain.
        :return: The resulting uncertainty, or ``NotImplemented`` for unsupported
            operands.
        """
        A = self._nom
        sA = self._err
        if isinstance(other, Uncertainty):
            B = other._nom
            sB = other._err

        elif isinstance(other, self._HANDLED_TYPES):
            B = other
            sB = 0
        else:
            return NotImplemented

        new_mag, new_err = _power_uncertainty(A, sA, B, sB)

        return self.__class__(new_mag, new_err)

    @ignore_runtime_warnings
    def __rpow__(self, other: Any) -> Uncertainty[Any] | NotImplementedType:
        # Other ** self
        """
        Evaluate ``other ** self`` with uncertainty propagation.

        Independent base and exponent errors are propagated at the nominal values.

        :param other: The other operand, which may be exact or uncertain.
        :return: The resulting uncertainty, or ``NotImplemented`` for unsupported
            operands.
        """
        B = self._nom
        sB = self._err
        if isinstance(other, Uncertainty):
            A = other._nom
            sA = other._err
        elif isinstance(other, self._HANDLED_TYPES):
            A = other
            sA = 0
        else:
            return NotImplemented
        new_mag, new_err = _power_uncertainty(A, sA, B, sB)

        return self.__class__(new_mag, new_err)

    def __abs__(self) -> Uncertainty[T]:
        """
        Evaluate ``abs(self)`` while preserving the error.

        :return: A new uncertainty with the transformed nominal values and unchanged
            errors.
        """
        return self.__class__(abs(self._nom), self._err)

    def __pos__(self) -> Uncertainty[T]:
        """
        Evaluate ``+self`` while preserving the error.

        :return: A new uncertainty with the transformed nominal values and unchanged
            errors.
        """
        return self.__class__(operator.pos(self._nom), self._err)

    def __neg__(self) -> Uncertainty[T]:
        """
        Evaluate ``-self`` while preserving the error.

        :return: A new uncertainty with the transformed nominal values and unchanged
            errors.
        """
        return self.__class__(operator.neg(self._nom), self._err)

    def _compare(
        self, other: Any, op: Callable[[Any, Any], Any]
    ) -> bool | np.bool_ | npt.NDArray[np.bool_]:
        """
        Compare nominal values using an ordering operation.

        :param other: Operand to compare, using its nominal values when uncertain.
        :param op: Comparison operation applied to the nominal operands.
        :return: The comparison result, without considering either operand error.
        """
        if isinstance(other, Uncertainty):
            return op(self._nom, other._nom)
        else:
            return op(self._nom, other)

    def __lt__(self, other: Any) -> bool | np.bool_ | npt.NDArray[np.bool_]:
        """
        Compare whether nominal values are less than the other operand.

        :param other: Exact or uncertain operand to compare.
        :return: The nominal less-than comparison, elementwise for arrays.
        """
        return self._compare(other, op=operator.lt)

    def __le__(self, other: Any) -> bool | np.bool_ | npt.NDArray[np.bool_]:
        """
        Compare whether nominal values are at most the other operand.

        :param other: Exact or uncertain operand to compare.
        :return: The nominal less-than-or-equal comparison, elementwise for arrays.
        """
        return self._compare(other, op=operator.le)

    def __ge__(self, other: Any) -> bool | np.bool_ | npt.NDArray[np.bool_]:
        """
        Compare whether nominal values are at least the other operand.

        :param other: Exact or uncertain operand to compare.
        :return: The nominal greater-than-or-equal comparison, elementwise for arrays.
        """
        return self._compare(other, op=operator.ge)

    def __gt__(self, other: Any) -> bool | np.bool_ | npt.NDArray[np.bool_]:
        """
        Compare whether nominal values are greater than the other operand.

        :param other: Exact or uncertain operand to compare.
        :return: The nominal greater-than comparison, elementwise for arrays.
        """
        return self._compare(other, op=operator.gt)

    def __bool__(self) -> bool:
        """
        Evaluate the truth value of the nominal value.

        :return: The nominal truth value.
        :raises ValueError: An array has no unambiguous truth value.
        """
        return bool(self._nom)

    __nonzero__ = __bool__

    # NumPy comparisons can return arrays rather than object's scalar boolean.
    def __ne__(self, other: object) -> bool | np.bool_ | npt.NDArray[np.bool_]:  # pyrefly: ignore [bad-override]
        """
        Negate the nominal equality comparison.

        :param other: Operand to compare with this uncertainty.
        :return: The logical inverse of :meth:`__eq__`.
        :raises EqualityError: The equality check rejects incompatible errors under the
            current settings.
        """
        out = self.__eq__(other)
        if self.is_vector:
            return np.logical_not(out)
        else:
            return not out

    def __eq__(self, other: object) -> bool | np.bool_ | npt.NDArray[np.bool_]:  # pyrefly: ignore [bad-override]
        """
        Compare nominal values, checking errors when nominal values match.

        Scalar comparisons use the configured relative tolerance. Vector comparisons
        use exact array equality. Matching nominal values with differing errors, or
        matching exact operands, emit an equality warning unless configured to raise.

        :param other: Exact or uncertain operand to compare.
        :return: Whether the nominal values compare equal.
        :raises EqualityError: Nominal values match but error information differs and
            ``error_on_equality`` is enabled.
        """
        if self.is_vector:
            # Compare vector Uncertainty with vector Uncertainty.
            if isinstance(other, Uncertainty):
                result = np.array_equal(self._nom, other._nom)
                if result and not np.array_equal(self._err, other._err):
                    msg = "Uncertainty objects have identical values but different standard deviations."
                    if get_settings().error_on_equality:
                        raise EqualityError(msg)
                    warnings.warn(msg, EqualityWarning, stacklevel=2)
                return result

            # Compare vector Uncertainty with other object.
            result = np.array_equal(self._nom, cast("npt.ArrayLike", other))
            if result:
                msg = "Compared Uncertainty object with non-Uncertainty object."
                if get_settings().error_on_equality:
                    raise EqualityError(msg)
                warnings.warn(msg, EqualityWarning, stacklevel=2)
            return result

        else:
            # Compare scalar Uncertainty wth scalar Uncertainty.
            if isinstance(other, Uncertainty):
                try:
                    val_result = math.isclose(
                        self._nom, other._nom, rel_tol=get_settings().compare_rtol
                    )
                    err_result = math.isclose(
                        self._err, other._err, rel_tol=get_settings().compare_rtol
                    )
                except TypeError:
                    val_result = self._nom == other._nom
                    err_result = self._err == other._err
                if val_result and not err_result:
                    msg = "Uncertainty objects have identical values but different standard deviations."
                    if get_settings().error_on_equality:
                        raise EqualityError(msg)
                    warnings.warn(msg, EqualityWarning, stacklevel=2)
                return val_result

            # Compare scalar Uncertainty with other object.
            try:
                result = math.isclose(
                    self._nom, cast("SupportsFloat", other), rel_tol=get_settings().compare_rtol
                )
            except TypeError:
                result = self._nom == other
            if result:
                msg = "Compared Uncertainty object with non-Uncertainty object."
                if get_settings().error_on_equality:
                    raise EqualityError(msg)
                warnings.warn(msg, EqualityWarning, stacklevel=2)
            return result

    def __round__(self, ndigits: int | None = 0) -> Self:
        """
        Round nominal values while preserving the errors.

        :param ndigits: Decimal places to retain; ``None`` is treated as zero.
        :return: The rounded uncertainty with unchanged errors.
        """
        if ndigits is None:
            ndigits = 0
        if isinstance(self._nom, np.ndarray | np.number):
            return self.__class__(np.round(self._nom, decimals=ndigits), self._err)
        else:
            return self.__class__(float(round(self._nom, ndigits=ndigits)), self._err)

    # Approximate equality and mutable arrays cannot support a consistent hash.
    __hash__ = None

    # ====================================================================
    # ------------------ NUMPY FUNCTION / UFUNC SUPPORT ------------------
    # ====================================================================

    @ignore_runtime_warnings
    def __array_function__(
        self,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """
        Implement the NumPy array-function dispatch protocol.

        :param func: NumPy function being called.
        :param types: Operand classes participating in protocol dispatch.
        :param args: Positional arguments supplied to the NumPy function.
        :param kwargs: Keyword arguments supplied to the NumPy function.
        :return: The registered operation result, or ``NotImplemented`` when dispatch is
            unsupported.
        """
        if func.__name__ not in HANDLED_FUNCTIONS or not any(
            issubclass(t, self.__class__) for t in types
        ):
            return NotImplemented
        else:
            return wrap_numpy("function", func, args, kwargs)

    @ignore_runtime_warnings
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *args: Any, **kwargs: Any) -> Any:
        """
        Implement the NumPy ufunc dispatch protocol.

        :param ufunc: NumPy ufunc being called.
        :param method: Ufunc method name; only ``"__call__"`` is supported.
        :param args: Positional arguments supplied to the ufunc.
        :param kwargs: Keyword arguments supplied to the ufunc.
        :return: The registered ufunc result.
        :raises NotImplementedError: The ufunc or its method is unsupported.
        """
        if method != "__call__":
            raise NotImplementedError
        if ufunc.__name__ not in HANDLED_UFUNCS:
            msg = f"Ufunc {ufunc.__name__} is not implemented!"
            raise NotImplementedError(msg) from None
        return wrap_numpy("ufunc", ufunc, args, kwargs)

    def __array__(
        self, dtype: npt.DTypeLike | None = None, *, copy: bool | None = None
    ) -> npt.NDArray[Any]:
        """
        Convert nominal values to a NumPy array, discarding uncertainty.

        The conversion emits a downcast warning unless configured to raise.

        :param dtype: Requested array dtype, or ``None`` to preserve the nominal dtype.
        :param copy: NumPy copy policy: always copy, never copy, or copy only when
            needed.
        :return: The nominal values as a NumPy array.
        :raises DowncastError: The settings prohibit discarding uncertainty.
        """
        msg = "The uncertainty is stripped when downcasting to ndarray."
        if get_settings().error_on_downcast:
            raise DowncastError(msg)
        warnings.warn(
            msg,
            DowncastWarning,
            stacklevel=2,
        )
        return np.asarray(self._nom, dtype=dtype, copy=copy)

    def __getattr__(self, item: str) -> Any:
        """
        Resolve supported array attributes and registered NumPy operations.

        :param item: Attribute or operation name to resolve.
        :return: The requested metadata, transformed uncertainty, or callable operation.
        :raises AttributeError: The requested attribute is not supported.
        """
        if item.startswith("__array_"):
            # Handle array protocol attributes other than `__array__`
            msg = f"Array protocol attribute {item} not available."
            raise AttributeError(msg)

        # Vector uncertainty
        if self.is_vector:
            if item in self.__ndarray_attributes__:
                return getattr(self._nom, item)
            elif item in self.__apply_to_both_ndarray__:
                val: npt.NDArray | np.number | ScalarT | Callable = getattr(self._nom, item)
                err = getattr(self._err, item)
                if callable(val):
                    return lambda *args, **kwargs: self.__class__(
                        val(*args, **kwargs), err(*args, **kwargs)
                    )
                else:
                    return self.__class__(val, err)

        if item in HANDLED_UFUNCS:
            return lambda *args, **kwargs: wrap_numpy("ufunc", item, [self, *list(args)], kwargs)
        elif item in HANDLED_FUNCTIONS:
            return lambda *args, **kwargs: wrap_numpy(
                "function", item, [self, *list(args)], kwargs
            )

        msg = f"Attribute {item} not available in Uncertainty, or as NumPy ufunc or function."
        raise AttributeError(msg) from None

    def __bytes__(self) -> bytes:
        """
        Encode the uncertainty display using the preferred locale encoding.

        :return: The encoded text representation.
        """
        return str(self).encode(locale.getpreferredencoding())

    # ===================================================================
    # ------------------ VECTOR-SPECIFIC FUNCTIONALITY ------------------
    # ===================================================================

    __apply_to_both_ndarray__ = (
        "flatten",
        "real",
        "imag",
        "astype",
        "T",
        "reshape",
    )
    __ndarray_attributes__ = ("dtype", "ndim", "size")

    __array_priority__ = 18

    @_unsupported_type("scal")
    def clip(self, *args: Any, **kwargs: Any) -> Uncertainty[T]:
        """
        Clip nominal values while preserving their existing errors.

        Available only for vector uncertainties. Options are forwarded to
        :meth:`numpy.ndarray.clip`.

        :param args: Positional clipping bounds and any positional output argument.
        :param kwargs: Keyword clipping bounds and NumPy clipping options.
        :return: A new uncertainty containing the clipped nominal values and original
            errors.
        """
        assert isinstance(self._nom, np.ndarray)
        return cast(
            "Uncertainty[T]",
            self.__class__(self._nom.clip(*args, **kwargs), self._err),
        )

    @_unsupported_type("scal")
    def fill(self, value: ScalarT | Uncertainty[Any]) -> None:
        """
        Fill a vector uncertainty in place.

        An uncertain fill value replaces both components. An exact fill value keeps
        the current errors. The configured nonfinite policy is applied before updating.

        :param value: Scalar nominal value or scalar uncertainty to repeat.
        :return: ``None``.
        :raises TypeError: The receiver or supplied uncertainty has an unsupported
            scalar/vector kind.
        """
        assert isinstance(self._nom, np.ndarray) and isinstance(self._err, np.ndarray)
        nominal = self._nom.copy()
        error = self._err.copy()
        if isinstance(value, Uncertainty):
            if value.is_vector:
                msg = "fill requires a scalar value"
                raise TypeError(msg)
            nominal.fill(value.value)
            error.fill(value.error)
        else:
            nominal.fill(value)
        normalized = self.__class__(nominal, error)
        self._nom[...] = normalized.value
        self._err[...] = normalized.error

    @_unsupported_type("scal")
    def put(
        self,
        indices: int | Sequence[int] | npt.NDArray[np.integer],
        values: Uncertainty[Any],
        mode: Literal["raise", "wrap", "clip"] = "raise",
    ) -> None:
        """
        Replace flattened entries of a vector uncertainty in place.

        Both nominal values and errors are validated before either is updated.

        :param indices: Flat positions to replace.
        :param values: Uncertainty values to insert, repeated as needed by NumPy.
        :param mode: Out-of-bounds index policy: ``"raise"``, ``"wrap"``, or ``"clip"``.
        :return: ``None``.
        :raises TypeError: The receiver is scalar or the inserted values are not an
            uncertainty.
        :raises IndexError: An index is out of bounds in ``"raise"`` mode.
        """
        assert isinstance(self._nom, np.ndarray) and isinstance(self._err, np.ndarray)
        if isinstance(values, Uncertainty):
            nominal = self._nom.copy()
            error = self._err.copy()
            nominal.put(indices, values._nom, mode)
            error.put(indices, values._err, mode)
            normalized = self.__class__(nominal, error)
            self._nom[...] = normalized.value
            self._err[...] = normalized.error
        else:
            msg = "Can only 'put' Uncertainty objects into Uncertainty objects"
            raise TypeError(msg)

    @_unsupported_type("scal")
    def copy(self) -> Uncertainty[T]:
        """
        Copy a vector uncertainty and both of its arrays.

        :return: A new uncertainty with independent nominal and error storage.
        :raises TypeError: The uncertainty is scalar.
        """
        assert isinstance(self._nom, np.ndarray) and isinstance(self._err, np.ndarray)
        return cast("Uncertainty[T]", self.__class__(self._nom.copy(), self._err.copy()))

    # Special properties.
    @property
    @_unsupported_type("scal")
    def flat(self) -> Iterator[Uncertainty[np.floating]]:
        """
        Iterate over scalar uncertainties in row-major order.

        :yields: A scalar uncertainty for each nominal value and matching error.
        :raises TypeError: The uncertainty is scalar.
        """
        assert isinstance(self._nom, np.ndarray) and isinstance(self._err, np.ndarray)
        for u, v in zip(self._nom.flat, self._err.flat, strict=False):
            yield self.__class__(u, v)

    @property
    @_unsupported_type("scal")
    def shape(self) -> tuple[int, ...]:
        """
        Access the shape of a vector uncertainty.

        :return: The shared shape of the nominal and error arrays.
        :raises TypeError: The uncertainty is scalar.
        """
        assert isinstance(self._nom, np.ndarray)
        return self._nom.shape

    @shape.setter
    @_unsupported_type("scal")
    def shape(self, value: int | tuple[int, ...]) -> None:
        """
        Change both vector array shapes in place after validating their layouts.

        :param value: New shape compatible with the existing array storage.
        :return: ``None``.
        :raises TypeError: The uncertainty is scalar.
        :raises ValueError: The new shape has an incompatible size.
        :raises AttributeError: An array cannot adopt the requested shape without
            copying.
        """
        assert isinstance(self._nom, np.ndarray) and isinstance(self._err, np.ndarray)
        nominal_view = self._nom.view()
        error_view = self._err.view()
        nominal_view.shape = value
        error_view.shape = value
        self._nom.shape = value
        self._err.shape = value

    @property
    @_unsupported_type("scal")
    def nbytes(self) -> int:
        """
        Count the storage occupied by a vector uncertainty.

        :return: The total bytes occupied by the nominal and error arrays.
        :raises TypeError: The uncertainty is scalar.
        """
        assert isinstance(self._nom, np.ndarray) and isinstance(self._err, np.ndarray)
        return self._nom.nbytes + self._err.nbytes

    @_unsupported_type("scal")
    def searchsorted(
        self,
        v: npt.ArrayLike,
        side: Literal["left", "right"] = "left",
        sorter: Sequence[int] | npt.NDArray[np.integer] | None = None,
    ) -> np.intp | npt.NDArray[np.intp]:
        """
        Find insertion positions in sorted nominal values of a vector uncertainty.

        :param v: Values to insert into the sorted nominal array.
        :param side: Use ``"left"`` for the first valid position or ``"right"`` for the
            last.
        :param sorter: Optional indices that sort the nominal array in ascending order.
        :return: The insertion positions returned by :meth:`numpy.ndarray.searchsorted`.
        :raises TypeError: The uncertainty is scalar.
        """
        assert isinstance(self._nom, np.ndarray)
        return self._nom.searchsorted(v, side=side, sorter=sorter)

    @_unsupported_type("scal")
    def tolist(self) -> list[Any] | Uncertainty[Any]:
        """
        Convert a vector uncertainty into nested lists of scalar uncertainties.

        :return: Nested lists preserving the array dimensions and each element error.
        :raises TypeError: The uncertainty is scalar.
        """
        assert isinstance(self._nom, np.ndarray) and isinstance(self._err, np.ndarray)
        try:
            nom = self._nom.tolist()
            err = self._err.tolist()
            if not isinstance(nom, list):
                return self.__class__(nom, err)
            else:
                return [
                    (
                        self.__class__(n, e).tolist()
                        if isinstance(n, list)
                        else self.__class__(n, e)
                    )
                    for n, e in zip(nom, err, strict=False)
                ]
        except AttributeError:
            msg = f"{type(self._nom).__name__}' does not support tolist."
            raise AttributeError(msg) from None

    @_unsupported_type("scal")
    def view(self) -> Self:
        """
        Create an uncertainty sharing the current vector storage.

        :return: A new uncertainty whose arrays are views of the nominal and error
            arrays.
        :raises TypeError: The uncertainty is scalar.
        """
        assert isinstance(self._nom, np.ndarray) and isinstance(self._err, np.ndarray)
        return self.__class__(self._nom.view(), self._err.view())

    @_unsupported_type("scal")
    def __iter__(self) -> Iterator[Uncertainty[Any]]:
        """
        Iterate over a vector uncertainty along its first dimension.

        :yields: An uncertainty for each nominal slice and its matching errors.
        :raises TypeError: The uncertainty is scalar.
        """
        assert isinstance(self._nom, np.ndarray) and isinstance(self._err, np.ndarray)
        for v, e in zip(self._nom, self._err, strict=False):
            yield self.__class__(v, e)

    @_unsupported_type("scal")
    def __len__(self) -> int:
        """
        Count entries along the first dimension of a vector uncertainty.

        :return: The length of the nominal array.
        :raises TypeError: The uncertainty is scalar.
        """
        assert isinstance(self._nom, np.ndarray)
        return len(self._nom)

    @_unsupported_type("scal")
    def __getitem__(self, key: Any) -> Uncertainty[Any]:
        """
        Select nominal values and their matching errors from a vector uncertainty.

        :param key: Index or slice selecting the desired entries.
        :return: An uncertainty containing the selected values and errors.
        :raises TypeError: The uncertainty is scalar.
        :raises IndexError: The index is invalid for the array.
        """
        assert isinstance(self._nom, np.ndarray) and isinstance(self._err, np.ndarray)
        try:
            return self.__class__(self._nom[key], self._err[key])
        except IndexError as e:
            msg = f"Index '{key}' not supported"
            raise IndexError(msg) from e

    @_unsupported_type("scal")
    def __setitem__(self, key: Any, value: Uncertainty[Any] | ScalarT) -> None:
        """
        Assign nominal values and matching errors to selected vector entries.

        :param key: Index or slice selecting entries to replace.
        :param value: Replacement uncertainty; a bare nonfinite scalar may mark missing
            values.
        :return: ``None``.
        :raises TypeError: The receiver is scalar or a bare finite replacement value is
            supplied.
        """
        assert isinstance(self._nom, np.ndarray) and isinstance(self._err, np.ndarray)
        if not isinstance(value, Uncertainty) and np.isfinite(value):
            msg = f"Can only pass Uncertainty type to __setitem__! Instead passed {type(value)}"
            raise TypeError(msg)

        value = self.__class__(value)
        if np.size(value._nom) == 1 and np.ndim(value._nom) > 0:
            self._nom[key] = value._nom[0]
            self._err[key] = value._err[0]
        else:
            self._nom[key] = value._nom
            self._err[key] = value._err

    # ===================================================================
    # ------------------ SCALAR-SPECIFIC FUNCTIONALITY ------------------
    # ===================================================================

    @_unsupported_type("vec")
    def __float__(self) -> float:
        """
        Apply ``float`` to the scalar nominal value, discarding its error.

        The conversion emits a downcast warning unless configured to raise.

        :return: The nominal value converted by ``float``.
        :raises TypeError: The uncertainty is vector-valued.
        :raises DowncastError: The settings prohibit discarding uncertainty.
        """
        msg = "The uncertainty is stripped when downcasting to float."
        if get_settings().error_on_downcast:
            raise DowncastError(msg)
        warnings.warn(
            msg,
            DowncastWarning,
            stacklevel=2,
        )
        return float(self._nom)

    @_unsupported_type("vec")
    def __int__(self) -> int:
        """
        Apply ``int`` to the scalar nominal value, discarding its error.

        The conversion emits a downcast warning unless configured to raise.

        :return: The nominal value converted by ``int``.
        :raises TypeError: The uncertainty is vector-valued.
        :raises DowncastError: The settings prohibit discarding uncertainty.
        """
        msg = "The uncertainty is stripped when downcasting to int."
        if get_settings().error_on_downcast:
            raise DowncastError(msg)
        warnings.warn(
            msg,
            DowncastWarning,
            stacklevel=2,
        )
        return int(self._nom)

    @_unsupported_type("vec")
    def __complex__(self) -> complex:
        """
        Apply ``complex`` to the scalar nominal value, discarding its error.

        The conversion emits a downcast warning unless configured to raise.

        :return: The nominal value converted by ``complex``.
        :raises TypeError: The uncertainty is vector-valued.
        :raises DowncastError: The settings prohibit discarding uncertainty.
        """
        msg = "The uncertainty is stripped when downcasting to float."
        if get_settings().error_on_downcast:
            raise DowncastError(msg)
        warnings.warn(
            msg,
            DowncastWarning,
            stacklevel=2,
        )
        return complex(self._nom)


VectorUncertainty = Uncertainty
"""Alias for `Uncertainty` to maintain backward compatibility."""


ScalarUncertainty = Uncertainty
"""Alias for `Uncertainty` to maintain backward compatibility."""


@overload
def nominal_values(x: Uncertainty[T]) -> T: ...
@overload
def nominal_values(x: object) -> Any: ...


def nominal_values(x: Any) -> Any:
    """
    Extract central values from an uncertainty or supported input sequence.

    :param x: Uncertainty, sequence, or exact value to inspect.
    :return: Nominal values after uncertainty construction when supported, otherwise the
        original object.
    """
    if isinstance(x, Uncertainty):
        return x.value
    else:
        try:
            x2 = Uncertainty(x)
        except Exception:  # noqa: BLE001 - Best-effort extraction preserves unsupported inputs.
            return x
        else:
            if isinstance(x2, ScalarT):
                return x2
            else:
                return x2.value


@overload
def std_devs(x: Uncertainty[T]) -> T: ...
@overload
def std_devs(x: object) -> Any: ...


def std_devs(x: Any) -> Any:
    """
    Extract standard deviations from an uncertainty or supported input sequence.

    :param x: Uncertainty, sequence, or exact value to inspect.
    :return: The extracted errors, or zeros for exact values.
    """
    if isinstance(x, Uncertainty):
        return x.error
    else:
        try:
            x2 = Uncertainty(x)
        except Exception:  # noqa: BLE001 - Unsupported exact inputs receive zero errors.
            return np.zeros_like(x) if np.ndim(x) > 0 else 0.0
        else:
            if isinstance(x2, ScalarT):
                return 0.0
            else:
                return x2.error


def _check_units(value: Any, err: Any) -> tuple[Any, Any, Any]:
    """
    Separate magnitudes and compatible units from nominal values and errors.

    :param value: Central values, possibly carrying units.
    :param err: Standard deviations, possibly carrying units.
    :return: The nominal magnitude, error magnitude, and selected units, or ``None`` for
        unitless inputs.
    :raises pint.errors.DimensionalityError: The supplied quantities have incompatible
        units.
    """
    mag_has_units = hasattr(value, "units")
    mag_units = getattr(value, "units", None)
    err_has_units = hasattr(err, "units")
    err_units = getattr(err, "units", None)

    if mag_has_units and mag_units is not None:
        Q = mag_units._REGISTRY.Quantity
        ret_val = Q(value.m, value.units).to(mag_units).m
        ret_err = Q(err.m, err.units).to(mag_units).m if err_has_units else err
        ret_units = mag_units
    # This branch will never actually work, but it's here
    # to raise a Dimensionality error without needing to import pint
    elif err_has_units:
        assert err_units is not None
        Q = err_units._REGISTRY.Quantity
        ret_val = Q(value).to(err_units).m
        ret_err = Q(err.m, err.units).to(err_units).m
        ret_units = err_units
    else:
        ret_units = None
        ret_val = value
        ret_err = err

    return ret_val, ret_err, ret_units


def _type_error_msg(u_type: str, operation: str) -> str:
    """
    Describe an operation unsupported by an uncertainty kind.

    :param u_type: Uncertainty kind to display in the message.
    :param operation: Name of the unsupported operation.
    :return: The formatted error message.
    """
    return f"{u_type} Uncertainty objects do not support the '{operation}' operation"
