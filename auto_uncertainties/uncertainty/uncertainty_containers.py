# Based heavily on the implementation of Pint's Quantity object
from __future__ import annotations

from collections.abc import Callable, Sequence
import copy
from functools import wraps
import locale
import math
import operator
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Self,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)
import warnings

import joblib
import numpy as np
import numpy.typing as npt

from auto_uncertainties import (
    DowncastError,
    DowncastWarning,
    EqualityError,
    EqualityWarning,
    NegativeStdDevError,
    UncertaintyDisplay,
)
from auto_uncertainties.numpy import HANDLED_FUNCTIONS, HANDLED_UFUNCS, wrap_numpy
from auto_uncertainties.util import deprecated, ignore_runtime_warnings

if TYPE_CHECKING:
    from pint.facets.plain import PlainQuantity


ERROR_ON_DOWNCAST = False
ERROR_ON_EQ = False
COMPARE_RTOL = 1e-9

__all__ = [
    "ScalarUncertainty",
    "UType",
    "Uncertainty",
    "VectorUncertainty",
    "nominal_values",
    "std_devs",
]

G = TypeVar("G")
T = TypeVar("T", float, np.floating, npt.NDArray[np.floating])
"""`TypeVar` specifying the underlying data types supporting `Uncertainty` objects."""

UType: TypeAlias = float | np.floating | npt.NDArray[np.floating]
"""Type alias for the underlying data types supporting `Uncertainty` objects."""

ScalarT: TypeAlias = int | float | np.number
"""Scalar types used throughout AutoUncertainties."""

SupportedSequence: TypeAlias = (
    "Sequence[ScalarT | Uncertainty[float] | Uncertainty[np.floating]]"
)
"""Sequences supported by the `Uncertainty` constructor."""

ValT: TypeAlias = "ScalarT | SupportedSequence | Uncertainty | npt.NDArray[np.number]"
"""Types supported in the ``value`` parameter of the `Uncertainty` constructor."""

ErrT: TypeAlias = ScalarT | Sequence[ScalarT] | npt.NDArray[np.number]
"""Types supported in the ``error`` parameter of the `Uncertainty` constructor."""


# Helper decorator to raise an informative TypeError when an operation is unsupported.
def _unsupported_type(t: str):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
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

        return wrapper

    return decorator


ErrT: TypeAlias = ScalarT | Sequence[ScalarT] | npt.NDArray[np.number]
"""Types supported in the ``error`` parameter of the `Uncertainty` constructor."""


class Uncertainty(Generic[T], UncertaintyDisplay):
    """
    Representation of a central value and its associated uncertainty.

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

    :param value: The central value(s)
    :param error: The uncertainty value(s). Zero if not provided.

    :raise NegativeStdDevError: If ``err`` is negative, or contains negative values
    :raise TypeError: If the parameters are of incompatible types
    :raise ValueError: If the parameters have incomaptible values (e.g., misaligned array sizes)

    :return: An initialized `Uncertainty` object

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
       0.315322 +/- 0.237246

       >>> u4 == u1
       True

    .. code-block:: python
       :caption: Pint Quantity Example

       >>> from pint import Quantity
       >>> val = Quantity(2.24, 'kg')
       >>> err = Quantity(0.208, 'kg')
       >>> new_quantity = Uncertainty(val, err)
       >>> new_quantity
       <Quantity(2.24 +/- 0.208, 'kilogram')>

    .. note::

       * If sequences (not NumPy arrays) are supplied for ``value`` and ``error``,
         their numeric values will always be converted to `numpy.float64`.

       * If `pint.Quantity` objects are supplied for either parameter, the behavior
         is exactly as described in the `from_quantities` method.

       * If an `Uncertainty` is supplied for ``value``, its ``error`` attribute will
         always override any ``error`` argument (if it is supplied).

       * If the ``error`` parameter is not finite, the resulting `Uncertainty` object
         will have its ``error`` attribute set to zero.

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
    def __new__(cls, value, error=...) -> Uncertainty: ...
    def __new__(cls, value, error=None):
        # Use from_quantities if one or more Pint Quantity objects were supplied.
        if _check_units(value, error)[2] is not None:
            return cls.from_quantities(value, error)

        # Use from_sequence if a sequence is supplied.
        if isinstance(value, Sequence):
            return cls.from_sequence(value, error)

        instance = super().__new__(cls)
        instance.__init__(value, error, skip=False)
        return instance

    # List of __init__ overloads for static type checking.
    @overload
    def __init__(self: Uncertainty[float], value: int, error: ErrT | None = ...): ...
    @overload
    def __init__(
        self: Uncertainty[np.float64], value: np.integer, error: ErrT | None = ...
    ): ...
    @overload
    def __init__(
        self: Uncertainty[npt.NDArray[np.floating]],
        value: npt.NDArray[np.integer] | SupportedSequence,
        error: ErrT | None = ...,
    ): ...
    @overload
    def __init__(self: Uncertainty[float], value: Uncertainty[float]): ...
    @overload
    def __init__(self: Uncertainty[np.floating], value: Uncertainty[np.floating]): ...
    @overload
    def __init__(
        self: Uncertainty[npt.NDArray[np.floating]],
        value: Uncertainty[npt.NDArray[np.floating]],
    ): ...
    @overload
    def __init__(self: Self, value: T, error: ErrT | None = ...): ...
    @overload
    def __init__(self, value: ValT, error: ErrT | None = ..., skip: bool = ...): ...
    def __init__(self, value, error=None, skip=True) -> None:
        if skip:
            return

        # Avoid zero-dimensional arrays.
        value = (
            value[()]
            if (isinstance(value, np.ndarray) and np.ndim(value) == 0)
            else value
        )
        error = (
            error[()]
            if (isinstance(error, np.ndarray) and np.ndim(error) == 0)
            else error
        )

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
            if error is not None and np.isfinite(error) and error < 0:
                msg = f"Got negative value ({error}) for the standard deviation"
                raise NegativeStdDevError(msg)

            caster = np.float64 if isinstance(value, np.number) else float
            if isinstance(value, int | np.integer):
                self._nom = cast(T, caster(value))
                self._err = cast(
                    T,
                    caster(error)
                    if (error is not None and np.isfinite(error))
                    else caster(0.0),
                )
            else:
                caster = (
                    (lambda x: x)
                    if isinstance(error, np.floating) and isinstance(value, np.floating)
                    else caster
                )
                self._nom = cast(T, value)
                self._err = cast(
                    T,
                    caster(error)
                    if (error is not None and np.isfinite(error))
                    else caster(0.0),
                )

        else:
            msg = f"Unsupported argument types (got type(value)={type(value)}, type(error)={type(error)})"
            raise TypeError(msg)

    def _init_seq(
        self, value: SupportedSequence, error: Sequence[ScalarT] | ScalarT | None
    ) -> None:
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
        if error is None:
            error = np.zeros_like(value)

        # Constant error.
        elif isinstance(error, ScalarT):
            error = np.ones_like(value) * error

        elif np.ndim(error) != np.ndim(value) or np.shape(error) != np.shape(value):
            msg = f"Error must have the same shape as value (got value.shape={np.shape(value)}, error.shape={np.shape(error)})"
            raise ValueError(msg)

        # Replace NaN with zero in errors
        error[~np.isfinite(error)] = 0

        if np.any(error < 0):
            msg = f"Found {np.count_nonzero(error < 0)} negative values for the standard deviation"
            raise NegativeStdDevError(msg)

        # Convert int data to float
        if issubclass(value.dtype.type, np.integer):
            value = value.astype(np.float64)
        if issubclass(error.dtype.type, np.integer):
            error = error.astype(np.float64)

        self._nom = cast(T, value)
        self._err = cast(T, error)

    @property
    def is_vector(self) -> bool:
        """Whether the current object is a vector uncertainty."""
        return isinstance(self._nom, np.ndarray) and isinstance(self._err, np.ndarray)

    @property
    def value(self) -> T:
        """The central value of the `Uncertainty` object."""
        return self._nom

    @property
    def error(self) -> T:
        """The uncertainty (error) of the `Uncertainty` object."""
        return self._err

    @property
    def relative(self) -> T:
        """The relative uncertainty of the `Uncertainty` object."""
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
            return cast(T, self._err / abs(self._nom))
        except OverflowError:
            return cast(T, float("inf") if isinstance(self._err, float) else np.inf)
        except ZeroDivisionError:
            return cast(T, float("nan") if isinstance(self._err, float) else np.nan)

    @property
    def rel(self) -> T:
        """Alias for `relative`."""
        return self.relative

    @property
    def rel2(self) -> T:
        """The square of the relative uncertainty of the `Uncertainty` object."""
        if self.is_vector:
            return self.relative**2
        try:
            return self.relative**2
        except OverflowError:
            return cast(T, float("inf") if isinstance(self._err, float) else np.inf)

    def plus_minus(self, error: UType) -> Uncertainty[T]:
        """
        Add an error to the `Uncertainty` object.

        Returns a new instance.

        :param error: Error value to add
        """
        old_err: T = self._err
        new_err: T
        result = np.sqrt(old_err**2 + error**2)

        # Check if float, NumPy float, or NumPy array.
        if isinstance(self._err, float):
            new_err = cast(T, float(result))
        else:
            new_err = cast(T, result)

        return self.__class__(self._nom, new_err)

    @classmethod
    def from_string(cls, string: str) -> Uncertainty:
        """
        Create an `Uncertainty` object from a string representation of the value and error.

        :param string: A string representation of the value and error. The error can be represented as
            "+/-" or "±". For instance, 5.0 +- 1.0 or 5.0 ± 1.0.
        """
        new_str = string.replace("+/-", "±")
        new_str = new_str.replace("+-", "±")
        if "±" not in new_str:
            return cls(float(string))
        else:
            u1, u2 = new_str.split("±")
            return cls(float(u1), float(u2))

    @classmethod
    def from_quantities(cls, value, error=None):
        """
        Create a `pint.Quantity` object with uncertainty from one or more `~pint.Quantity` objects.

        .. warning::

           Static type inference is hindered when using this method.
           Call ``Uncertainty(value, error)`` instead for full typing support.

        :param value: The central value(s) of the `Uncertainty` object
        :param error: The uncertainty value(s) of the `Uncertainty` object

        .. note::

           It is not necessary (and not advised) to call this method explicitly.
           Instantiating an `Uncertainty` object with ``Uncertainty(value, error)``
           will automatically use `from_quantities` if `~pint.Quantity` objects
           are supplied as parameters.

        .. note::

           * If **neither** argument is a `~pint.Quantity`, returns a regular
             `Uncertainty` object.

           * If **both** arguments are `~pint.Quantity` objects, returns a
             `~pint.Quantity` (wrapped `Uncertainty`) with the same units as
             ``value`` (attempts to convert ``error`` to ``value.units``).

           * If **only the** ``value`` argument is a `~pint.Quantity`, returns
             a `~pint.Quantity` (wrapped `Uncertainty`) object with the same units as ``value``.

           * If **only the** ``error`` argument is a `~pint.Quantity`, returns
             a `~pint.Quantity` (wrapped `Uncertainty`) object with the same units as ``error``.
        """
        value_, error_, units = _check_units(value, error)
        instance = cls(value_, error_)
        if units is not None:
            instance *= units
        return instance

    @classmethod
    def from_sequence(cls, value, error=None):
        """
        Creates either an `Uncertainty` object or a `pint.Quantity` object
        from a supported sequence.

        The primary purpose of this method is to intercept sequences containing
        `~pint.Quantity` objects, reformat them, and then continue the instantiation
        process.

        .. warning::

           Static type inference is hindered when using this method.
           Call ``Uncertainty(value, error)`` instead for full typing support.

        :param value: The central value(s)
        :param error: The uncertainty value(s). Zero if not provided.

        .. note::

           It is not necessary (and not advised) to call this method explicitly.
           Instantiating an `Uncertainty` object with ``Uncertainty(value, error)``
           will automatically use `from_sequence` if sequences are supplied as
           parameters.
        """
        if not isinstance(value, Sequence):
            return cls(value, error)

        mag_units = err_units = None
        if len(value) > 0 and hasattr(value[0], "units"):
            # Converts all values to the same unit.
            mag_units = value[0].units
            value = [
                (item.to(mag_units).m if hasattr(item, "units") else item)
                for item in value
            ]
        if (
            isinstance(error, Sequence)
            and len(error) > 0
            and hasattr(error[0], "units")
        ):
            # Convert all error units to value units, if possible. Otherwise, make sure all errors use same units.
            err_units = error[0].units
            to_units = err_units if mag_units is None else mag_units
            error = [
                (item.to(to_units).m if hasattr(item, "units") else item)
                for item in error
            ]

        instance = super().__new__(cls)
        instance.__init__(value, error, skip=False)
        units = err_units if mag_units is None else mag_units
        return instance if units is None else (instance * units)

    @classmethod
    @deprecated("call Uncertainty() directly instead")
    def from_list(
        cls, value: ValT, error: ErrT | None = None
    ) -> Uncertainty:  # pragma: no cover
        """
        Alias for `from_sequence`.

        .. warning::

           This method is deprecated.
        """
        return cls(value, error)

    def __getstate__(self) -> dict[str, T]:
        return {"nominal_value": self._nom, "std_devs": self._err}

    def __setstate__(self, state) -> None:
        self._nom = state["nominal_value"]
        self._err = state["std_devs"]

    def __getnewargs__(self) -> tuple[T, T]:
        return self._nom, self._err

    def __copy__(self) -> Uncertainty[T]:
        return self.__class__(copy.copy(self._nom), copy.copy(self._err))

    def __deepcopy__(self, memo) -> Uncertainty[T]:
        return self.__class__(
            copy.deepcopy(self._nom, memo), copy.deepcopy(self._err, memo)
        )

    # =====================================================
    # ------------------ MATH OPERATIONS ------------------
    # =====================================================

    _HANDLED_TYPES = (float, int, np.ndarray)

    def __add__(self, other):
        if isinstance(other, Uncertainty):
            new_mag = self._nom + other._nom
            new_err = np.sqrt(self._err**2 + other._err**2)
        elif isinstance(other, self._HANDLED_TYPES):
            new_mag = self._nom + other
            new_err = self._err
        else:
            return NotImplemented
        try:
            return self.__class__(new_mag, new_err)
        except NotImplementedError:
            return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Uncertainty):
            new_mag = self._nom - other._nom
            new_err = np.sqrt(self._err**2 + other._err**2)
        elif isinstance(other, self._HANDLED_TYPES):
            new_mag = self._nom - other
            new_err = self._err
        else:
            return NotImplemented
        try:
            return self.__class__(new_mag, new_err)
        except NotImplementedError:
            return NotImplemented

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, Uncertainty):
            new_mag = self._nom * other._nom
            new_err = np.sqrt(
                (other._nom * self._err) ** 2 + (self._nom * other._err) ** 2
            )
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
    def __truediv__(self, other):
        if isinstance(other, Uncertainty):
            new_mag = self._nom / other._nom
            new_err = np.sqrt(
                (self._err / other._nom) ** 2
                + (self._nom * other._err / other._nom**2) ** 2
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
    def __rtruediv__(self, other):
        # Other / Self
        if isinstance(other, Uncertainty):
            raise TypeError
        elif isinstance(other, self._HANDLED_TYPES):
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

    def __floordiv__(self, other):
        if isinstance(other, Uncertainty):
            new_mag = self._nom // other._nom
        elif isinstance(other, self._HANDLED_TYPES):
            new_mag = self._nom // other
        else:
            return NotImplemented
        new_err = self.__div__(other).error

        return self.__class__(new_mag, new_err)

    def __rfloordiv__(self, other):
        if isinstance(other, Uncertainty):
            return other.__floordiv__(self)
        elif isinstance(other, self._HANDLED_TYPES):
            new_mag = other // self._nom
            new_err = self.__rdiv__(other).error
            return self.__class__(new_mag, new_err)
        else:
            return NotImplemented

    def __mod__(self, other):
        if isinstance(other, Uncertainty):
            new_mag = self._nom % other._nom
        elif isinstance(other, self._HANDLED_TYPES):
            new_mag = self._nom % other
        else:
            return NotImplemented
        return self.__class__(new_mag, self._err)

    def __rmod__(self, other):
        if isinstance(other, self._HANDLED_TYPES):
            new_mag = other % self._nom
            return self.__class__(new_mag, self._err)
        else:
            return NotImplemented

    def __divmod__(self, other):  # pragma: no cover
        return self // other, self % other

    def __rdivmod__(self, other):  # pragma: no cover
        return other // self, other % self

    @ignore_runtime_warnings
    def __pow__(self, other):
        # Self ** other
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

        new_mag = A**B
        new_err = np.abs(new_mag) * np.sqrt(
            (B / A * sA) ** 2 + (np.log(np.abs(A)) * sB) ** 2
        )

        return self.__class__(new_mag, new_err)

    @ignore_runtime_warnings
    def __rpow__(self, other):
        # Other ** self
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
        new_mag = A**B
        new_err = np.abs(new_mag) * np.sqrt(
            (B / A * sA) ** 2 + (np.log(np.abs(A)) * sB) ** 2
        )

        return self.__class__(new_mag, new_err)

    def __abs__(self):
        return self.__class__(abs(self._nom), self._err)

    def __pos__(self):
        return self.__class__(operator.pos(self._nom), self._err)

    def __neg__(self):
        return self.__class__(operator.neg(self._nom), self._err)

    def _compare(self, other, op):
        if isinstance(other, Uncertainty):
            return op(self._nom, other._nom)
        else:
            return op(self._nom, other)

    __lt__ = lambda self, other: self._compare(  # noqa: E731
        other, op=operator.lt
    )
    __le__ = lambda self, other: self._compare(  # noqa: E731
        other, op=operator.le
    )
    __ge__ = lambda self, other: self._compare(  # noqa: E731
        other, op=operator.ge
    )
    __gt__ = lambda self, other: self._compare(  # noqa: E731
        other, op=operator.gt
    )

    def __bool__(self) -> bool:
        return bool(self._nom)

    __nonzero__ = __bool__

    def __ne__(self, other):
        out = self.__eq__(other)
        if self.is_vector:
            return np.logical_not(out)
        else:
            return not out

    def __eq__(self, other):
        if self.is_vector:
            # Compare vector Uncertainty with vector Uncertainty.
            if isinstance(other, Uncertainty):
                result = np.array_equal(self._nom, other._nom)
                if result and not np.array_equal(self._err, other._err):
                    msg = "Uncertainty objects have identical values but different standard deviations."
                    if ERROR_ON_EQ:
                        raise EqualityError(msg)
                    else:
                        warnings.warn(msg, EqualityWarning, stacklevel=2)
                return result

            # Compare vector Uncertainty with other object.
            result = np.array_equal(self._nom, other)
            if result:
                msg = "Compared Uncertainty object with non-Uncertainty object."
                if ERROR_ON_EQ:
                    raise EqualityError(msg)
                else:
                    warnings.warn(msg, EqualityWarning, stacklevel=2)
            return result

        else:
            # Compare scalar Uncertainty wth scalar Uncertainty.
            if isinstance(other, Uncertainty):
                try:
                    val_result = math.isclose(
                        self._nom, other._nom, rel_tol=COMPARE_RTOL
                    )
                    err_result = math.isclose(
                        self._err, other._err, rel_tol=COMPARE_RTOL
                    )
                except TypeError:
                    val_result = self._nom == other._nom
                    err_result = self._err == other._err
                if val_result and not err_result:
                    msg = "Uncertainty objects have identical values but different standard deviations."
                    if ERROR_ON_EQ:
                        raise EqualityError(msg)
                    else:
                        warnings.warn(msg, EqualityWarning, stacklevel=2)
                return val_result

            # Compare scalar Uncertainty with other object.
            try:
                result = math.isclose(self._nom, other, rel_tol=COMPARE_RTOL)
            except TypeError:
                result = self._nom == other
            if result:
                msg = "Compared Uncertainty object with non-Uncertainty object."
                if ERROR_ON_EQ:
                    raise EqualityError(msg)
                else:
                    warnings.warn(msg, EqualityWarning, stacklevel=2)
            return result

    def __round__(self, ndigits):
        if isinstance(self._nom, np.ndarray | np.number):
            return self.__class__(np.round(self._nom, decimals=ndigits), self._err)
        else:
            return self.__class__(float(round(self._nom, ndigits=ndigits)), self._err)

    def __hash__(self) -> int:
        if self.is_vector:
            digest = joblib.hash((self._nom, self._err), hash_name="sha1")
            digest = "" if digest is None else digest
            return int.from_bytes(bytes(digest, encoding="utf-8"), "big")
        else:
            return hash((self._nom, self._err))

    # ====================================================================
    # ------------------ NUMPY FUNCTION / UFUNC SUPPORT ------------------
    # ====================================================================

    @ignore_runtime_warnings
    def __array_function__(self, func, types, args, kwargs):
        if func.__name__ not in HANDLED_FUNCTIONS or not any(
            issubclass(t, self.__class__) for t in types
        ):
            return NotImplemented
        else:
            return wrap_numpy("function", func, args, kwargs)

    @ignore_runtime_warnings
    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if method != "__call__":
            raise NotImplementedError
        else:
            if ufunc.__name__ not in HANDLED_UFUNCS:
                msg = f"Ufunc {ufunc.__name__} is not implemented!"
                raise NotImplementedError(msg) from None
            else:
                return wrap_numpy("ufunc", ufunc, args, kwargs)

    def __array__(self, dtype=None, *, copy=None) -> np.ndarray:
        msg = "The uncertainty is stripped when downcasting to ndarray."
        if ERROR_ON_DOWNCAST:
            raise DowncastError(msg)
        else:
            warnings.warn(
                msg,
                DowncastWarning,
                stacklevel=2,
            )
            return np.asarray(self._nom, dtype=dtype, copy=copy)

    def __getattr__(self, item) -> Any:
        if item.startswith("__array_"):
            # Handle array protocol attributes other than `__array__`
            msg = f"Array protocol attribute {item} not available."
            raise AttributeError(msg)

        # Vector uncertainty
        if self.is_vector:
            if item in self.__ndarray_attributes__:
                return getattr(self._nom, item)
            elif item in self.__apply_to_both_ndarray__:
                val: npt.NDArray | np.number | ScalarT | Callable = getattr(
                    self._nom, item
                )
                err = getattr(self._err, item)
                if callable(val):
                    return lambda *args, **kwargs: self.__class__(
                        val(*args, **kwargs), err(*args, **kwargs)
                    )
                else:
                    return self.__class__(val, err)

        if item in HANDLED_UFUNCS:
            return lambda *args, **kwargs: wrap_numpy(
                "ufunc", item, [self, *list(args)], kwargs
            )
        elif item in HANDLED_FUNCTIONS:
            return lambda *args, **kwargs: wrap_numpy(
                "function", item, [self, *list(args)], kwargs
            )

        msg = f"Attribute {item} not available in Uncertainty, or as NumPy ufunc or function."
        raise AttributeError(msg) from None

    def __bytes__(self) -> bytes:
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
    def clip(self, *args, **kwargs) -> Uncertainty[T]:  # type: ignore
        """
        NumPy `~numpy.ndarray.clip` implementation.

        :param min:
        :param max:
        :param out:

        .. note::

           Implemented only for vector uncertainty objects.
        """
        if isinstance(self._nom, np.ndarray):
            return self.__class__(self._nom.clip(*args, **kwargs), self._err)

    @_unsupported_type("scal")
    def fill(self, value) -> None:
        """
        NumPy `~numpy.ndarray.fill` implementation.

        :param value:

        .. note::

           Implemented only for vector uncertainty objects.

        """
        if isinstance(self._nom, np.ndarray):
            return self._nom.fill(value)

    @_unsupported_type("scal")
    def put(
        self, indices, values, mode: Literal["raise", "wrap", "clip"] = "raise"
    ) -> None:
        """
        NumPy `~numpy.ndarray.put` implementation.

        :param indices:
        :param values:
        :param mode:

        .. note::

           Implemented only for vector uncertainty objects.

        """
        if isinstance(self._nom, np.ndarray) and isinstance(self._err, np.ndarray):
            if isinstance(values, Uncertainty):
                self._nom.put(indices, values._nom, mode)
                self._err.put(indices, values._err, mode)
            else:
                msg = "Can only 'put' Uncertainty objects into Uncertainty objects"
                raise TypeError(msg)

    @_unsupported_type("scal")
    def copy(self) -> Uncertainty[T]:  # type: ignore
        """
        Return a copy of the `Uncertainty` object.

        .. note::

           Implemented only for vector uncertainty objects.
        """
        if isinstance(self._nom, np.ndarray) and isinstance(self._err, np.ndarray):
            return self.__class__(self._nom.copy(), self._err.copy())

    # Special properties.
    @property
    @_unsupported_type("scal")
    def flat(self):
        """
        NumPy `~numpy.ndarray.flat` implementation.

        .. note::

           Implemented only for vector uncertainty objects.
        """
        if isinstance(self._nom, np.ndarray) and isinstance(self._err, np.ndarray):
            for u, v in zip(self._nom.flat, self._err.flat, strict=False):
                yield self.__class__(u, v)

    @property
    @_unsupported_type("scal")
    def shape(self):
        """
        NumPy `~numpy.ndarray.shape` implemenetation.


        .. note::

           Implemented only for vector uncertainty objects.
        """
        if isinstance(self._nom, np.ndarray):
            return self._nom.shape

    @shape.setter
    @_unsupported_type("scal")
    def shape(self, value):
        if isinstance(self._nom, np.ndarray) and isinstance(self._err, np.ndarray):
            self._nom.shape = value
            self._err.shape = value

    @property
    @_unsupported_type("scal")
    def nbytes(self):
        """
        NumPy `~numpy.ndarray.nbytes` implementation.

        .. note::

           Implemented only for vector uncertainty objects.
        """
        if isinstance(self._nom, np.ndarray) and isinstance(self._err, np.ndarray):
            return self._nom.nbytes + self._err.nbytes

    @_unsupported_type("scal")
    def searchsorted(self, v, side: Literal["left", "right"] = "left", sorter=None):
        """
        NumPy `~numpy.ndarray.searchsorted` implementation.

        .. note::

           Implemented only for vector uncertainty objects.
        """
        if isinstance(self._nom, np.ndarray):
            return self._nom.searchsorted(v, side)

    @_unsupported_type("scal")
    def tolist(self):
        """
        NumPy `~numpy.ndarray.tolist` implementation.

        .. note::

           Implemented only for vector uncertainty objects.
        """
        if isinstance(self._nom, np.ndarray) and isinstance(self._err, np.ndarray):
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
    def view(self):
        """
        NumPy `~numpy.ndarray.view` implementation.

        .. note::

           Implemented only for vector uncertainty objects.
        """
        if isinstance(self._nom, np.ndarray) and isinstance(self._err, np.ndarray):
            return self.__class__(self._nom.view(), self._err.view())

    @_unsupported_type("scal")
    def __iter__(self):
        if isinstance(self._nom, np.ndarray) and isinstance(self._err, np.ndarray):
            for v, e in zip(self._nom, self._err, strict=False):
                yield self.__class__(v, e)

    @_unsupported_type("scal")
    def __len__(self) -> int:  # type: ignore
        if isinstance(self._nom, np.ndarray):
            return len(self._nom)

    @_unsupported_type("scal")
    def __getitem__(self, key: int) -> Uncertainty:  # type: ignore
        if isinstance(self._nom, np.ndarray) and isinstance(self._err, np.ndarray):
            try:
                return self.__class__(self._nom[key], self._err[key])
            except IndexError as e:
                msg = f"Index '{key}' not supported"
                raise IndexError(msg) from e

    @_unsupported_type("scal")
    def __setitem__(self, key: int, value: Uncertainty) -> None:
        if isinstance(self._nom, np.ndarray) and isinstance(self._err, np.ndarray):
            # If value is nan, just set the value in those regions to nan and return. This is the only case where a scalar can be passed as an argument!
            if not isinstance(value, Uncertainty):
                if not np.isfinite(value):
                    self._nom[key] = value
                    self._err[key] = 0
                    return
                else:
                    msg = f"Can only pass Uncertainty type to __setitem__! Instead passed {type(value)}"
                    raise TypeError(msg)

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
    def __float__(self):
        msg = "The uncertainty is stripped when downcasting to float."
        if ERROR_ON_DOWNCAST:
            raise DowncastError(msg)
        else:
            warnings.warn(
                msg,
                DowncastWarning,
                stacklevel=2,
            )
        return float(self._nom)

    @_unsupported_type("vec")
    def __int__(self):
        msg = "The uncertainty is stripped when downcasting to int."
        if ERROR_ON_DOWNCAST:
            raise DowncastError(msg)
        else:
            warnings.warn(
                msg,
                DowncastWarning,
                stacklevel=2,
            )
        return int(self._nom)

    @_unsupported_type("vec")
    def __complex__(self):
        msg = "The uncertainty is stripped when downcasting to float."
        if ERROR_ON_DOWNCAST:
            raise DowncastError(msg)
        else:
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
def nominal_values(x: Uncertainty[UType]) -> UType : ... 
@overload 
def nominal_values(x: G) -> G:... 
    
def nominal_values(x: Any) -> UType | Any:
    """Return the central value of an `Uncertainty` object if it is one, otherwise returns the object."""
    if isinstance(x, Uncertainty):
        return x.value
    else:
        try:
            x2 = Uncertainty(x)
        except Exception:
            return x
        else:
            if isinstance(x2, ScalarT):
                return x2
            else:
                return x2.value

@overload 
def std_devs(x: Uncertainty[UType]) -> UType : ... 
@overload 
def std_devs(x: G) -> G:... 
    
def std_devs(x):
    """Return the uncertainty of an `Uncertainty` object if it is one, otherwise returns zero."""
    if isinstance(x, Uncertainty):
        return x.error
    else:
        try:
            x2 = Uncertainty(x)
        except Exception:
            return np.zeros_like(x) if np.ndim(x) > 0 else 0.0
        else:
            if isinstance(x2, ScalarT):
                return 0.0
            else:
                return x2.error


def _check_units(value, err) -> tuple[Any, Any, Any]:
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
        Q = err_units._REGISTRY.Quantity  # type: ignore
        ret_val = Q(value).to(err_units).m
        ret_err = Q(err.m, err.units).to(err_units).m
        ret_units = err_units
    else:
        ret_units = None
        ret_val = value
        ret_err = err

    return ret_val, ret_err, ret_units


def _type_error_msg(u_type: str, operation: str) -> str:
    return f"{u_type} Uncertainty objects do not support the '{operation}' operation"
