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

from auto_uncertainties.arithmetic import binary
from auto_uncertainties.config import get_settings
from auto_uncertainties.display_format import UncertaintyDisplay
from auto_uncertainties.exceptions import (
    DowncastError,
    DowncastWarning,
    EqualityError,
    EqualityWarning,
    NegativeStdDevError,
)
from auto_uncertainties.numpy.numpy_wrappers import HANDLED_FUNCTIONS, HANDLED_UFUNCS, wrap_numpy
from auto_uncertainties.units import split_units
from auto_uncertainties.util import ignore_runtime_warnings

__all__ = [
    "UType",
    "Uncertainty",
    "nominal_values",
    "std_devs",
]

F = TypeVar("F", bound=Callable[..., Any])
ValueT = TypeVar("ValueT", float, np.floating, npt.NDArray[np.floating])
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
def unsupported_type(t: str) -> Callable[[F], F]:
    """
    Restrict a method to scalar or vector uncertainties.

    :param t: Kind to reject: ``"scal"`` for scalars or ``"vec"`` for vectors.
    :return: A decorator that checks the receiver before calling the method.
    :raises TypeError: The decorated method is called on a rejected kind or the kind
        selector is invalid.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self: Uncertainty[Any], *args: Any, **kwargs: Any) -> Any:
            if t == "vec":
                if self.is_vector:
                    raise TypeError(type_error_msg("Vector", func.__qualname__))
            elif t == "scal":
                if not self.is_vector:
                    raise TypeError(type_error_msg("Scalar", func.__qualname__))
            else:
                msg = "Invalid arguments passed to unsupported_type decorator"
                raise TypeError(msg)
            return func(self, *args, **kwargs)

        return cast("F", wrapper)

    return decorator


class Uncertainty(UncertaintyDisplay[ValueT]):
    """
    Represent a central value and its associated uncertainty.

    Parameters can be numbers, sequences, `numpy` arrays, `pint.Quantity` objects,
    other `Uncertainty` objects, or lists / tuples of `Uncertainty` objects.

    Array methods have explicit signatures. Indexing, mutation, and storage-view
    methods require array inputs; NumPy operations also support scalar values.

    `Uncertainty` objects only support float-based data types. If integers or
    integer arrays are passed as parameters to the `Uncertainty` constructor,
    they will be cast to `float` (or `numpy.float64` if a `numpy.integer` subclass
    is detected).

    :param value: Central values or an existing uncertainty.
    :param error: Standard deviations; omitted errors are zero. Existing
        uncertainties in ``value`` supply their own errors.
    :return: An initialized uncertainty, wrapped in a quantity when units are supplied.
    :raises auto_uncertainties.exceptions.NegativeStdDevError: An error is negative under the configured
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

    nom: ValueT
    err: ValueT

    # Only quantity inputs need to replace the object returned by construction.
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
        """
        Construct an uncertainty, preserving quantity units where present.

        :param value: Central values or an existing uncertainty.
        :param error: Standard deviations, or ``None`` for exact values.
        :return: An initialized uncertainty, wrapped in a quantity when the inputs
            supply units.
        """
        magnitude, deviation, units = split_units(value, error)
        if units is not None:
            return units._REGISTRY.Quantity(cls(magnitude, deviation), units)
        return super().__new__(cls)

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
    def __init__(self: Self, value: ValueT, error: ErrT | None = ...) -> None: ...
    @overload
    def __init__(
        self,
        value: ValT | PlainQuantity | Sequence[PlainQuantity],
        error: ErrT | PlainQuantity | Sequence[PlainQuantity] | None = ...,
    ) -> None: ...
    def __init__(
        self,
        value: ValT | PlainQuantity | Sequence[PlainQuantity],
        error: ErrT | PlainQuantity | Sequence[PlainQuantity] | None = None,
    ) -> None:
        """
        Initialize and validate nominal values and standard deviations once.

        :param value: Central values or an uncertainty whose components are reused.
        :param error: Standard deviations; omitted errors are zero.
        :return: ``None``.
        :raises auto_uncertainties.exceptions.NegativeStdDevError: A standard deviation is negative.
        :raises TypeError: Input types are unsupported or incompatible.
        :raises ValueError: Value and error shapes or sequence lengths differ.
        """
        nominal, deviation = normalize(value, error)
        self.nom, self.err = cast("ValueT", nominal), cast("ValueT", deviation)

    @property
    def is_vector(self) -> bool:
        """
        Check whether the uncertainty stores arrays.

        :return: Whether both nominal values and errors are NumPy arrays.
        """
        return isinstance(self.nom, np.ndarray) and isinstance(self.err, np.ndarray)

    @property
    def value(self) -> ValueT:
        """
        Access the central values.

        :return: The stored nominal values.
        """
        return self.nom

    @property
    def error(self) -> ValueT:
        """
        Access the absolute uncertainties.

        :return: The stored standard deviations.
        """
        return self.err

    @property
    def relative(self) -> ValueT:
        """
        Compute uncertainty relative to the nominal magnitude.

        :return: The error divided by the absolute nominal value, with NaN at zero
            nominal values.
        """
        # Vector uncertainty
        if isinstance(self.nom, np.ndarray) and isinstance(self.err, np.ndarray):
            rel = np.zeros_like(self.nom)
            inf = np.isinf(self.nom)
            nan = self.nom == 0
            valid = ~inf & ~nan
            rel[valid] = self.err[valid] / np.abs(self.nom[valid])
            rel[inf] = np.inf
            rel[nan] = np.nan
            return rel

        # Scalar uncertainty
        try:
            return cast("ValueT", self.err / abs(self.nom))
        except OverflowError:
            return cast("ValueT", float("inf") if isinstance(self.err, float) else np.inf)
        except ZeroDivisionError:
            return cast("ValueT", float("nan") if isinstance(self.err, float) else np.nan)

    @property
    def rel(self) -> ValueT:
        """
        Access the relative uncertainty through its short alias.

        :return: The same result as :attr:`relative`.
        """
        return self.relative

    @property
    def rel2(self) -> ValueT:
        """
        Compute the squared relative uncertainty.

        :return: The square of :attr:`relative`.
        """
        if self.is_vector:
            return cast("ValueT", self.relative**2)
        try:
            return cast("ValueT", self.relative**2)
        except OverflowError:
            return cast("ValueT", float("inf") if isinstance(self.err, float) else np.inf)

    def plus_minus(self, error: UType) -> Uncertainty[ValueT]:
        """
        Combine an additional independent error with the existing uncertainty.

        :param error: Additional standard deviation to combine in quadrature.
        :return: A new uncertainty with unchanged nominal values and the combined error.
        """
        old_err: ValueT = self.err
        new_err: ValueT
        result = np.sqrt(old_err**2 + error**2)

        # Check if float, NumPy float, or NumPy array.
        new_err = (
            cast("ValueT", float(result))
            if isinstance(self.err, float)
            else cast("ValueT", result)
        )

        return self.__class__(self.nom, new_err)

    @classmethod
    def from_string(cls, string: str) -> Uncertainty:
        """
        Parse a nominal value and an optional error.

        :param string: Value optionally followed by ``+/-``, ``+-``, or ``±`` and an
            error, such as ``5.0 +/- 1.0``.
        :return: The parsed scalar uncertainty, with zero error if none is supplied.
        :raises ValueError: The components cannot be parsed as numbers.
        :raises auto_uncertainties.exceptions.NegativeStdDevError: The parsed error is negative under the configured
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
        return cls(value, error)

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
        return cls(value, error)

    def __getstate__(self) -> dict[str, ValueT]:
        """
        Collect the state used to serialize an uncertainty.

        :return: The nominal values and errors under ``nominal_value`` and ``std_devs``.
        """
        return {"nominal_value": self.nom, "std_devs": self.err}

    def __setstate__(self, state: dict[str, ValueT]) -> None:
        """
        Restore previously serialized nominal values and errors.

        :param state: State containing the ``nominal_value`` and ``std_devs`` entries.
        :return: ``None``.
        """
        self.nom = state["nominal_value"]
        self.err = state["std_devs"]

    def __getnewargs__(self) -> tuple[ValueT, ValueT]:
        """
        Provide constructor arguments for deserialization.

        :return: The nominal values and errors, in constructor order.
        """
        return self.nom, self.err

    def __copy__(self) -> Uncertainty[ValueT]:
        """
        Copy the nominal values and errors.

        :return: A new uncertainty with shallow copies of both components.
        """
        return self.__class__(copy.copy(self.nom), copy.copy(self.err))

    def __deepcopy__(self, memo: dict[int, Any]) -> Uncertainty[ValueT]:
        """
        Deep-copy the nominal values and errors.

        :param memo: Objects already copied during the current deep-copy operation.
        :return: A new uncertainty with recursively copied components.
        """
        return self.__class__(copy.deepcopy(self.nom, memo), copy.deepcopy(self.err, memo))

    # =====================================================
    # ------------------ MATH OPERATIONS ------------------
    # =====================================================

    HANDLED_TYPES = (float, int, np.ndarray)

    def __add__(self, other: Any) -> Uncertainty[Any] | NotImplementedType:
        """
        Evaluate ``self + other`` with uncertainty propagation.

        Independent errors are combined in quadrature.

        :param other: The other operand, which may be exact or uncertain.
        :return: The resulting uncertainty, or ``NotImplemented`` for unsupported
            operands.
        """
        if isinstance(other, Uncertainty):
            other, other_error = other.nom, other.err
        elif isinstance(other, (*self.HANDLED_TYPES, np.floating, np.integer)):
            other_error = 0
        else:
            return NotImplemented
        return self.__class__(*binary(np.add, self.nom, self.err, other, other_error))

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
            other, other_error = other.nom, other.err
        elif isinstance(other, (*self.HANDLED_TYPES, np.floating, np.integer)):
            other_error = 0
        else:
            return NotImplemented
        return self.__class__(*binary(np.subtract, self.nom, self.err, other, other_error))

    def __rsub__(self, other: Any) -> Uncertainty[Any] | NotImplementedType:
        """
        Evaluate ``other - self`` with uncertainty propagation.

        Independent errors are combined in quadrature.

        :param other: The other operand, which may be exact or uncertain.
        :return: The resulting uncertainty, or ``NotImplemented`` for unsupported
            operands.
        """
        if isinstance(other, Uncertainty):
            other, other_error = other.nom, other.err
        elif isinstance(other, (*self.HANDLED_TYPES, np.floating, np.integer)):
            other_error = 0
        else:
            return NotImplemented
        return self.__class__(*binary(np.subtract, other, other_error, self.nom, self.err))

    def __mul__(self, other: Any) -> Uncertainty[Any] | NotImplementedType:
        """
        Evaluate ``self * other`` with uncertainty propagation.

        Independent errors are propagated through the product.

        :param other: The other operand, which may be exact or uncertain.
        :return: The resulting uncertainty, or ``NotImplemented`` for unsupported
            operands.
        """
        if isinstance(other, Uncertainty):
            other, other_error = other.nom, other.err
        elif isinstance(other, (*self.HANDLED_TYPES, np.floating, np.integer)):
            other_error = 0
        else:
            return NotImplemented
        return self.__class__(*binary(np.multiply, self.nom, self.err, other, other_error))

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
            other, other_error = other.nom, other.err
        elif isinstance(other, (*self.HANDLED_TYPES, np.floating, np.integer)):
            other_error = 0
        else:
            return NotImplemented
        return self.__class__(*binary(np.divide, self.nom, self.err, other, other_error))

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
            other, other_error = other.nom, other.err
        elif isinstance(other, (*self.HANDLED_TYPES, np.floating, np.integer)):
            other_error = 0
        else:
            return NotImplemented
        return self.__class__(*binary(np.divide, other, other_error, self.nom, self.err))

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
            new_mag = self.nom // other.nom
        elif isinstance(other, self.HANDLED_TYPES):
            new_mag = self.nom // other
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
        elif isinstance(other, self.HANDLED_TYPES):
            new_mag = other // self.nom
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
            new_mag = self.nom % other.nom
            new_err = np.hypot(self.err, (self.nom // other.nom) * other.err)
        elif isinstance(other, self.HANDLED_TYPES):
            new_mag = self.nom % other
            new_err = (
                np.broadcast_to(self.err, np.shape(new_mag)).copy()
                if np.ndim(new_mag)
                else self.err
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
        if isinstance(other, self.HANDLED_TYPES):
            new_mag = other % self.nom
            new_err = np.abs((other // self.nom) * self.err)
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
        if isinstance(other, Uncertainty):
            other, other_error = other.nom, other.err
        elif isinstance(other, (*self.HANDLED_TYPES, np.floating, np.integer)):
            other_error = 0
        else:
            return NotImplemented
        return self.__class__(*binary(np.power, self.nom, self.err, other, other_error))

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
        if isinstance(other, Uncertainty):
            other, other_error = other.nom, other.err
        elif isinstance(other, (*self.HANDLED_TYPES, np.floating, np.integer)):
            other_error = 0
        else:
            return NotImplemented
        return self.__class__(*binary(np.power, other, other_error, self.nom, self.err))

    def __abs__(self) -> Uncertainty[ValueT]:
        """
        Evaluate ``abs(self)`` while preserving the error.

        :return: A new uncertainty with the transformed nominal values and unchanged
            errors.
        """
        return self.__class__(abs(self.nom), self.err)

    def __pos__(self) -> Uncertainty[ValueT]:
        """
        Evaluate ``+self`` while preserving the error.

        :return: A new uncertainty with the transformed nominal values and unchanged
            errors.
        """
        return self.__class__(operator.pos(self.nom), self.err)

    def __neg__(self) -> Uncertainty[ValueT]:
        """
        Evaluate ``-self`` while preserving the error.

        :return: A new uncertainty with the transformed nominal values and unchanged
            errors.
        """
        return self.__class__(operator.neg(self.nom), self.err)

    def compare(
        self, other: Any, op: Callable[[Any, Any], Any]
    ) -> bool | np.bool_ | npt.NDArray[np.bool_]:
        """
        Compare nominal values using an ordering operation.

        :param other: Operand to compare, using its nominal values when uncertain.
        :param op: Comparison operation applied to the nominal operands.
        :return: The comparison result, without considering either operand error.
        """
        if isinstance(other, Uncertainty):
            return op(self.nom, other.nom)
        else:
            return op(self.nom, other)

    def __lt__(self, other: Any) -> bool | np.bool_ | npt.NDArray[np.bool_]:
        """
        Compare whether nominal values are less than the other operand.

        :param other: Exact or uncertain operand to compare.
        :return: The nominal less-than comparison, elementwise for arrays.
        """
        return self.compare(other, op=operator.lt)

    def __le__(self, other: Any) -> bool | np.bool_ | npt.NDArray[np.bool_]:
        """
        Compare whether nominal values are at most the other operand.

        :param other: Exact or uncertain operand to compare.
        :return: The nominal less-than-or-equal comparison, elementwise for arrays.
        """
        return self.compare(other, op=operator.le)

    def __ge__(self, other: Any) -> bool | np.bool_ | npt.NDArray[np.bool_]:
        """
        Compare whether nominal values are at least the other operand.

        :param other: Exact or uncertain operand to compare.
        :return: The nominal greater-than-or-equal comparison, elementwise for arrays.
        """
        return self.compare(other, op=operator.ge)

    def __gt__(self, other: Any) -> bool | np.bool_ | npt.NDArray[np.bool_]:
        """
        Compare whether nominal values are greater than the other operand.

        :param other: Exact or uncertain operand to compare.
        :return: The nominal greater-than comparison, elementwise for arrays.
        """
        return self.compare(other, op=operator.gt)

    def __bool__(self) -> bool:
        """
        Evaluate the truth value of the nominal value.

        :return: The nominal truth value.
        :raises ValueError: An array has no unambiguous truth value.
        """
        return bool(self.nom)

    __nonzero__ = __bool__

    # NumPy comparisons can return arrays rather than object's scalar boolean.
    def __ne__(self, other: object) -> bool | np.bool_ | npt.NDArray[np.bool_]:  # pyrefly: ignore [bad-override]
        """
        Negate the nominal equality comparison.

        :param other: Operand to compare with this uncertainty.
        :return: The logical inverse of :meth:`__eq__`.
        :raises auto_uncertainties.exceptions.EqualityError: The equality check rejects incompatible errors under the
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
        :raises auto_uncertainties.exceptions.EqualityError: Nominal values match but error information differs and
            ``error_on_equality`` is enabled.
        """
        if self.is_vector:
            # Compare vector Uncertainty with vector Uncertainty.
            if isinstance(other, Uncertainty):
                result = np.array_equal(self.nom, other.nom)
                if result and not np.array_equal(self.err, other.err):
                    msg = "Uncertainty objects have identical values but different standard deviations."
                    if get_settings().error_on_equality:
                        raise EqualityError(msg)
                    warnings.warn(msg, EqualityWarning, stacklevel=2)
                return result

            # Compare vector Uncertainty with other object.
            result = np.array_equal(self.nom, cast("npt.ArrayLike", other))
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
                        self.nom, other.nom, rel_tol=get_settings().compare_rtol
                    )
                    err_result = math.isclose(
                        self.err, other.err, rel_tol=get_settings().compare_rtol
                    )
                except TypeError:
                    val_result = self.nom == other.nom
                    err_result = self.err == other.err
                if val_result and not err_result:
                    msg = "Uncertainty objects have identical values but different standard deviations."
                    if get_settings().error_on_equality:
                        raise EqualityError(msg)
                    warnings.warn(msg, EqualityWarning, stacklevel=2)
                return val_result

            # Compare scalar Uncertainty with other object.
            try:
                result = math.isclose(
                    self.nom, cast("SupportsFloat", other), rel_tol=get_settings().compare_rtol
                )
            except TypeError:
                result = self.nom == other
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
        if isinstance(self.nom, np.ndarray | np.number):
            return self.__class__(np.round(self.nom, decimals=ndigits), self.err)
        else:
            return self.__class__(float(round(self.nom, ndigits=ndigits)), self.err)

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
        :raises auto_uncertainties.exceptions.DowncastError: The settings prohibit discarding uncertainty.
        """
        msg = "The uncertainty is stripped when downcasting to ndarray."
        if get_settings().error_on_downcast:
            raise DowncastError(msg)
        warnings.warn(
            msg,
            DowncastWarning,
            stacklevel=2,
        )
        return np.asarray(self.nom, dtype=dtype, copy=copy)

    def apply_numpy(self, func: Callable[..., Any], **options: Any) -> Any:
        """
        Forward an explicit method call to the shared NumPy dispatcher.

        :param func: NumPy operation whose first argument is this uncertainty.
        :param options: Declared options; omitted NumPy defaults remain omitted.
        :return: The NumPy dispatch result.
        """
        return func(
            self, **{key: value for key, value in options.items() if value is not np._NoValue}
        )

    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: npt.DTypeLike | None = None,
        out: Uncertainty[Any] | None = None,
        keepdims: bool = False,  # noqa: FBT001, FBT002 - NumPy signature.
        *,
        initial: Any = np._NoValue,
        where: Any = np._NoValue,
    ) -> Uncertainty[Any]:
        """
        Apply :func:`numpy.sum` through the shared uncertainty handler.

        :param axis: Axes to reduce.
        :param dtype: Computation dtype.
        :param out: Optional destination.
        :param keepdims: Retain reduced axes.
        :param initial: Initial value.
        :param where: Elements included in the reduction.
        :return: The operation result.
        """
        return self.apply_numpy(
            np.sum,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def mean(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: npt.DTypeLike | None = None,
        out: Uncertainty[Any] | None = None,
        keepdims: bool = False,  # noqa: FBT001, FBT002 - NumPy signature.
        *,
        where: Any = np._NoValue,
    ) -> Uncertainty[Any]:
        """
        Apply :func:`numpy.mean` through the shared uncertainty handler.

        :param axis: Axes to reduce.
        :param dtype: Computation dtype.
        :param out: Optional destination.
        :param keepdims: Retain reduced axes.
        :param where: Elements included in the reduction.
        :return: The operation result.
        """
        return self.apply_numpy(
            np.mean, axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where
        )

    def std(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: npt.DTypeLike | None = None,
        out: Uncertainty[Any] | None = None,
        ddof: float = 0,
        keepdims: bool = False,  # noqa: FBT001, FBT002 - NumPy signature.
        *,
        where: Any = np._NoValue,
        mean: Any = np._NoValue,
        correction: Any = np._NoValue,
    ) -> Uncertainty[Any]:
        """
        Apply :func:`numpy.std` through the shared uncertainty handler.

        :param axis: Axes to reduce.
        :param dtype: Computation dtype.
        :param out: Optional destination.
        :param ddof: Degrees of freedom adjustment.
        :param keepdims: Retain reduced axes.
        :param where: Elements included in the reduction.
        :param mean: Precomputed mean.
        :param correction: Alternative degrees of freedom adjustment.
        :return: The operation result.
        """
        return self.apply_numpy(
            np.std,
            axis=axis,
            dtype=dtype,
            out=out,
            ddof=ddof,
            keepdims=keepdims,
            where=where,
            mean=mean,
            correction=correction,
        )

    def var(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: npt.DTypeLike | None = None,
        out: Uncertainty[Any] | None = None,
        ddof: float = 0,
        keepdims: bool = False,  # noqa: FBT001, FBT002 - NumPy signature.
        *,
        where: Any = np._NoValue,
        mean: Any = np._NoValue,
        correction: Any = np._NoValue,
    ) -> Uncertainty[Any]:
        """
        Apply :func:`numpy.var` through the shared uncertainty handler.

        :param axis: Axes to reduce.
        :param dtype: Computation dtype.
        :param out: Optional destination.
        :param ddof: Degrees of freedom adjustment.
        :param keepdims: Retain reduced axes.
        :param where: Elements included in the reduction.
        :param mean: Precomputed mean.
        :param correction: Alternative degrees of freedom adjustment.
        :return: The operation result.
        """
        return self.apply_numpy(
            np.var,
            axis=axis,
            dtype=dtype,
            out=out,
            ddof=ddof,
            keepdims=keepdims,
            where=where,
            mean=mean,
            correction=correction,
        )

    def min(
        self,
        axis: int | tuple[int, ...] | None = None,
        out: Uncertainty[Any] | None = None,
        keepdims: bool = False,  # noqa: FBT001, FBT002 - NumPy signature.
        *,
        initial: Any = np._NoValue,
        where: Any = np._NoValue,
    ) -> Uncertainty[Any]:
        """
        Apply :func:`numpy.min` through the shared uncertainty handler.

        :param axis: Axes to reduce.
        :param out: Optional destination.
        :param keepdims: Retain reduced axes.
        :param initial: Initial comparison value.
        :param where: Elements included in the reduction.
        :return: The operation result.
        """
        return self.apply_numpy(
            np.min, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where
        )

    def max(
        self,
        axis: int | tuple[int, ...] | None = None,
        out: Uncertainty[Any] | None = None,
        keepdims: bool = False,  # noqa: FBT001, FBT002 - NumPy signature.
        *,
        initial: Any = np._NoValue,
        where: Any = np._NoValue,
    ) -> Uncertainty[Any]:
        """
        Apply :func:`numpy.max` through the shared uncertainty handler.

        :param axis: Axes to reduce.
        :param out: Optional destination.
        :param keepdims: Retain reduced axes.
        :param initial: Initial comparison value.
        :param where: Elements included in the reduction.
        :return: The operation result.
        """
        return self.apply_numpy(
            np.max, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where
        )

    def argmin(
        self,
        axis: int | None = None,
        out: npt.NDArray[np.intp] | None = None,
        *,
        keepdims: bool = False,
    ) -> np.intp | npt.NDArray[np.intp]:
        """
        Apply :func:`numpy.argmin` through the shared uncertainty handler.

        :param axis: Axis to search.
        :param out: Optional index destination.
        :param keepdims: Retain the searched axis.
        :return: The operation result.
        """
        return self.apply_numpy(np.argmin, axis=axis, out=out, keepdims=keepdims)

    def argmax(
        self,
        axis: int | None = None,
        out: npt.NDArray[np.intp] | None = None,
        *,
        keepdims: bool = False,
    ) -> np.intp | npt.NDArray[np.intp]:
        """
        Apply :func:`numpy.argmax` through the shared uncertainty handler.

        :param axis: Axis to search.
        :param out: Optional index destination.
        :param keepdims: Retain the searched axis.
        :return: The operation result.
        """
        return self.apply_numpy(np.argmax, axis=axis, out=out, keepdims=keepdims)

    def ravel(self, order: Literal["C", "F", "A", "K"] = "C") -> Uncertainty[Any]:
        """
        Apply :func:`numpy.ravel` through the shared uncertainty handler.

        :param order: Traversal order.
        :return: The operation result.
        """
        return self.apply_numpy(np.ravel, order=order)

    def repeat(
        self, repeats: int | Sequence[int] | npt.NDArray[np.integer], axis: int | None = None
    ) -> Uncertainty[Any]:
        """
        Apply :func:`numpy.repeat` through the shared uncertainty handler.

        :param repeats: Repetition counts.
        :param axis: Axis to repeat.
        :return: The operation result.
        """
        return self.apply_numpy(np.repeat, repeats=repeats, axis=axis)

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> Uncertainty[Any]:
        """
        Apply :func:`numpy.squeeze` through the shared uncertainty handler.

        :param axis: Singleton axes to remove.
        :return: The operation result.
        """
        return self.apply_numpy(np.squeeze, axis=axis)

    def swapaxes(self, axis1: int, axis2: int) -> Uncertainty[Any]:
        """
        Apply :func:`numpy.swapaxes` through the shared uncertainty handler.

        :param axis1: First axis.
        :param axis2: Second axis.
        :return: The operation result.
        """
        return self.apply_numpy(np.swapaxes, axis1=axis1, axis2=axis2)

    def diagonal(self, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Uncertainty[Any]:
        """
        Apply :func:`numpy.diagonal` through the shared uncertainty handler.

        :param offset: Diagonal offset.
        :param axis1: First matrix axis.
        :param axis2: Second matrix axis.
        :return: The operation result.
        """
        return self.apply_numpy(np.diagonal, offset=offset, axis1=axis1, axis2=axis2)

    def take(
        self,
        indices: npt.ArrayLike,
        axis: int | None = None,
        out: Uncertainty[Any] | None = None,
        mode: str = "raise",
    ) -> Uncertainty[Any]:
        """
        Apply :func:`numpy.take` through the shared uncertainty handler.

        :param indices: Indices to select.
        :param axis: Axis to index.
        :param out: Optional destination.
        :param mode: Out-of-bounds handling.
        :return: The operation result.
        """
        return self.apply_numpy(np.take, indices=indices, axis=axis, out=out, mode=mode)

    def round(self, decimals: int = 0, out: Uncertainty[Any] | None = None) -> Uncertainty[Any]:
        """
        Apply :func:`numpy.round` through the shared uncertainty handler.

        :param decimals: Decimal places to retain.
        :param out: Optional destination.
        :return: The operation result.
        """
        return self.apply_numpy(np.round, decimals=decimals, out=out)

    def sort(
        self,
        axis: int | None = -1,
        kind: str | None = None,
        order: Any = None,
        *,
        stable: bool | None = None,
    ) -> Uncertainty[Any]:
        """
        Return a sorted copy, keeping errors paired with nominal values.

        :param axis: Axis to sort.
        :param kind: Sorting algorithm.
        :param order: Structured field order.
        :param stable: Request a stable sort.
        :return: The operation result.
        """
        return self.apply_numpy(np.sort, axis=axis, kind=kind, order=order, stable=stable)

    def conjugate(self) -> Uncertainty[Any]:
        """
        Apply :func:`numpy.conjugate` through the shared uncertainty handler.

        :return: The operation result.
        """
        return self.apply_numpy(np.conjugate)

    conj = conjugate

    def reshape(
        self,
        *shape: int | tuple[int, ...],
        order: Literal["C", "F", "A", "K"] = "C",
        copy: bool | None = None,
    ) -> Uncertainty[Any]:
        """
        Reshape values and errors together.

        :param shape: New dimensions, supplied separately or as one tuple.
        :param order: Element traversal order.
        :param copy: Whether to copy, share storage, or copy only when needed.
        :return: The reshaped uncertainty.
        """
        dimensions = shape[0] if len(shape) == 1 else shape
        return self.apply_numpy(np.reshape, shape=dimensions, order=order, copy=copy)

    def transpose(self, *axes: int | tuple[int, ...] | None) -> Uncertainty[Any]:
        """
        Permute the axes of both components.

        :param axes: Axis order as individual indices or one tuple; omitted to reverse.
        :return: The transposed uncertainty.
        """
        order = axes[0] if len(axes) == 1 else axes or None
        return self.apply_numpy(np.transpose, axes=order)

    def compress(
        self,
        condition: npt.ArrayLike,
        axis: int | None = None,
        out: Uncertainty[Any] | None = None,
    ) -> Uncertainty[Any]:
        """
        Select entries using a boolean condition.

        :param condition: Boolean selection mask.
        :param axis: Axis to select along, or ``None`` to flatten.
        :param out: Optional destination.
        :return: The selected uncertainty.
        """
        return wrap_numpy("function", np.compress, (condition, self), {"axis": axis, "out": out})

    @unsupported_type("scal")
    def flatten(self, order: Literal["C", "F", "A", "K"] = "C") -> Uncertainty[Any]:
        """
        Copy both components into one-dimensional arrays.

        :param order: Element traversal order.
        :return: A flattened copy.
        """
        assert isinstance(self.nom, np.ndarray) and isinstance(self.err, np.ndarray)
        return self.__class__(self.nom.flatten(order), self.err.flatten(order))

    @unsupported_type("scal")
    def astype(
        self,
        dtype: npt.DTypeLike,
        order: Literal["C", "F", "A", "K"] = "K",
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe", "same_value"] = "unsafe",
        subok: bool = True,  # noqa: FBT001, FBT002 - NumPy signature.
        copy: bool = True,  # noqa: FBT001, FBT002 - NumPy signature.
    ) -> Uncertainty[Any]:
        """
        Cast component arrays before applying normal uncertainty construction.

        :param dtype: Requested component dtype; integer results normalize to floats.
        :param order: Memory layout policy.
        :param casting: NumPy casting rule.
        :param subok: Preserve ndarray subclasses.
        :param copy: Whether to always copy component arrays.
        :return: An uncertainty with converted components.
        """
        assert isinstance(self.nom, np.ndarray) and isinstance(self.err, np.ndarray)
        return self.__class__(
            self.nom.astype(dtype, order, casting, subok, copy),
            self.err.astype(dtype, order, casting, subok, copy),
        )

    @property
    def dtype(self) -> np.dtype[Any]:
        """
        Access the ``dtype`` array property.

        :return: The nominal metadata or corresponding component transformation.
        """
        return np.asarray(self.nom).dtype

    @property
    def ndim(self) -> int:
        """
        Access the ``ndim`` array property.

        :return: The nominal metadata or corresponding component transformation.
        """
        return np.ndim(self.nom)

    @property
    def size(self) -> int:
        """
        Access the ``size`` array property.

        :return: The nominal metadata or corresponding component transformation.
        """
        return np.size(self.nom)

    @property
    def T(self) -> Uncertainty[Any]:
        """
        Access the ``T`` array property.

        :return: The nominal metadata or corresponding component transformation.
        """
        return self.transpose()

    @property
    def real(self) -> Uncertainty[Any]:
        """
        Access the ``real`` array property.

        :return: The nominal metadata or corresponding component transformation.
        """
        return self.__class__(np.real(self.nom), np.real(self.err))

    @property
    def imag(self) -> Uncertainty[Any]:
        """
        Access the ``imag`` array property.

        :return: The nominal metadata or corresponding component transformation.
        """
        return self.__class__(np.imag(self.nom), np.imag(self.err))

    def __bytes__(self) -> bytes:
        """
        Encode the uncertainty display using the preferred locale encoding.

        :return: The encoded text representation.
        """
        return str(self).encode(locale.getpreferredencoding())

    # ===================================================================
    # ------------------ VECTOR-SPECIFIC FUNCTIONALITY ------------------
    # ===================================================================

    __array_priority__ = 18

    @unsupported_type("scal")
    def clip(
        self,
        a_min: Any = np._NoValue,
        a_max: Any = np._NoValue,
        out: Uncertainty[Any] | None = None,
        *,
        min: Any = np._NoValue,  # noqa: A002 - NumPy keyword.
        max: Any = np._NoValue,  # noqa: A002 - NumPy keyword.
        **kwargs: Any,
    ) -> Uncertainty[Any]:
        """
        Clip nominal values while preserving their existing errors.

        :param a_min: Lower bound, or ``None`` for no lower bound.
        :param a_max: Upper bound, or ``None`` for no upper bound.
        :param out: Optional destination, updated after validating both components.
        :param min: Alternative keyword for the lower bound.
        :param max: Alternative keyword for the upper bound.
        :param kwargs: Additional NumPy clipping options, validated by NumPy.
        :return: The clipped uncertainty or the supplied destination.
        """
        return self.apply_numpy(
            np.clip, a_min=a_min, a_max=a_max, out=out, min=min, max=max, **kwargs
        )

    @unsupported_type("scal")
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
        assert isinstance(self.nom, np.ndarray) and isinstance(self.err, np.ndarray)
        nominal = self.nom.copy()
        error = self.err.copy()
        if isinstance(value, Uncertainty):
            if value.is_vector:
                msg = "fill requires a scalar value"
                raise TypeError(msg)
            nominal.fill(value.value)
            error.fill(value.error)
        else:
            nominal.fill(value)
        normalized = self.__class__(nominal, error)
        self.nom[...] = normalized.value
        self.err[...] = normalized.error

    @unsupported_type("scal")
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
        assert isinstance(self.nom, np.ndarray) and isinstance(self.err, np.ndarray)
        if isinstance(values, Uncertainty):
            nominal = self.nom.copy()
            error = self.err.copy()
            nominal.put(indices, values.nom, mode)
            error.put(indices, values.err, mode)
            normalized = self.__class__(nominal, error)
            self.nom[...] = normalized.value
            self.err[...] = normalized.error
        else:
            msg = "Can only 'put' Uncertainty objects into Uncertainty objects"
            raise TypeError(msg)

    @unsupported_type("scal")
    def copy(self) -> Uncertainty[ValueT]:
        """
        Copy a vector uncertainty and both of its arrays.

        :return: A new uncertainty with independent nominal and error storage.
        :raises TypeError: The uncertainty is scalar.
        """
        assert isinstance(self.nom, np.ndarray) and isinstance(self.err, np.ndarray)
        return cast("Uncertainty[ValueT]", self.__class__(self.nom.copy(), self.err.copy()))

    # Special properties.
    @property
    @unsupported_type("scal")
    def flat(self) -> Iterator[Uncertainty[np.floating]]:
        """
        Iterate over scalar uncertainties in row-major order.

        :yields: A scalar uncertainty for each nominal value and matching error.
        :raises TypeError: The uncertainty is scalar.
        """
        assert isinstance(self.nom, np.ndarray) and isinstance(self.err, np.ndarray)
        for u, v in zip(self.nom.flat, self.err.flat, strict=False):
            yield self.__class__(u, v)

    @property
    @unsupported_type("scal")
    def shape(self) -> tuple[int, ...]:
        """
        Access the shape of a vector uncertainty.

        :return: The shared shape of the nominal and error arrays.
        :raises TypeError: The uncertainty is scalar.
        """
        assert isinstance(self.nom, np.ndarray)
        return self.nom.shape

    @shape.setter
    @unsupported_type("scal")
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
        assert isinstance(self.nom, np.ndarray) and isinstance(self.err, np.ndarray)
        nominal_view = self.nom.view()
        error_view = self.err.view()
        nominal_view.shape = value
        error_view.shape = value
        self.nom.shape = value
        self.err.shape = value

    @property
    @unsupported_type("scal")
    def nbytes(self) -> int:
        """
        Count the storage occupied by a vector uncertainty.

        :return: The total bytes occupied by the nominal and error arrays.
        :raises TypeError: The uncertainty is scalar.
        """
        assert isinstance(self.nom, np.ndarray) and isinstance(self.err, np.ndarray)
        return self.nom.nbytes + self.err.nbytes

    @unsupported_type("scal")
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
        assert isinstance(self.nom, np.ndarray)
        return self.apply_numpy(np.searchsorted, v=v, side=side, sorter=sorter)

    @unsupported_type("scal")
    def tolist(self) -> list[Any] | Uncertainty[Any]:
        """
        Convert a vector uncertainty into nested lists of scalar uncertainties.

        :return: Nested lists preserving the array dimensions and each element error.
        :raises TypeError: The uncertainty is scalar.
        """
        assert isinstance(self.nom, np.ndarray) and isinstance(self.err, np.ndarray)
        try:
            nom = self.nom.tolist()
            err = self.err.tolist()
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
            msg = f"{type(self.nom).__name__}' does not support tolist."
            raise AttributeError(msg) from None

    @unsupported_type("scal")
    def view(self) -> Self:
        """
        Create an uncertainty sharing the current vector storage.

        :return: A new uncertainty whose arrays are views of the nominal and error
            arrays.
        :raises TypeError: The uncertainty is scalar.
        """
        assert isinstance(self.nom, np.ndarray) and isinstance(self.err, np.ndarray)
        return self.__class__(self.nom.view(), self.err.view())

    @unsupported_type("scal")
    def __iter__(self) -> Iterator[Uncertainty[Any]]:
        """
        Iterate over a vector uncertainty along its first dimension.

        :yields: An uncertainty for each nominal slice and its matching errors.
        :raises TypeError: The uncertainty is scalar.
        """
        assert isinstance(self.nom, np.ndarray) and isinstance(self.err, np.ndarray)
        for v, e in zip(self.nom, self.err, strict=False):
            yield self.__class__(v, e)

    @unsupported_type("scal")
    def __len__(self) -> int:
        """
        Count entries along the first dimension of a vector uncertainty.

        :return: The length of the nominal array.
        :raises TypeError: The uncertainty is scalar.
        """
        assert isinstance(self.nom, np.ndarray)
        return len(self.nom)

    @unsupported_type("scal")
    def __getitem__(self, key: Any) -> Uncertainty[Any]:
        """
        Select nominal values and their matching errors from a vector uncertainty.

        :param key: Index or slice selecting the desired entries.
        :return: An uncertainty containing the selected values and errors.
        :raises TypeError: The uncertainty is scalar.
        :raises IndexError: The index is invalid for the array.
        """
        assert isinstance(self.nom, np.ndarray) and isinstance(self.err, np.ndarray)
        try:
            return self.__class__(self.nom[key], self.err[key])
        except IndexError as e:
            msg = f"Index '{key}' not supported"
            raise IndexError(msg) from e

    @unsupported_type("scal")
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
        assert isinstance(self.nom, np.ndarray) and isinstance(self.err, np.ndarray)
        if not isinstance(value, Uncertainty) and np.isfinite(value):
            msg = f"Can only pass Uncertainty type to __setitem__! Instead passed {type(value)}"
            raise TypeError(msg)

        value = self.__class__(value)
        if np.size(value.nom) == 1 and np.ndim(value.nom) > 0:
            self.nom[key] = value.nom[0]
            self.err[key] = value.err[0]
        else:
            self.nom[key] = value.nom
            self.err[key] = value.err

    # ===================================================================
    # ------------------ SCALAR-SPECIFIC FUNCTIONALITY ------------------
    # ===================================================================

    @unsupported_type("vec")
    def __float__(self) -> float:
        """
        Apply ``float`` to the scalar nominal value, discarding its error.

        The conversion emits a downcast warning unless configured to raise.

        :return: The nominal value converted by ``float``.
        :raises TypeError: The uncertainty is vector-valued.
        :raises auto_uncertainties.exceptions.DowncastError: The settings prohibit discarding uncertainty.
        """
        msg = "The uncertainty is stripped when downcasting to float."
        if get_settings().error_on_downcast:
            raise DowncastError(msg)
        warnings.warn(
            msg,
            DowncastWarning,
            stacklevel=2,
        )
        return float(self.nom)

    @unsupported_type("vec")
    def __int__(self) -> int:
        """
        Apply ``int`` to the scalar nominal value, discarding its error.

        The conversion emits a downcast warning unless configured to raise.

        :return: The nominal value converted by ``int``.
        :raises TypeError: The uncertainty is vector-valued.
        :raises auto_uncertainties.exceptions.DowncastError: The settings prohibit discarding uncertainty.
        """
        msg = "The uncertainty is stripped when downcasting to int."
        if get_settings().error_on_downcast:
            raise DowncastError(msg)
        warnings.warn(
            msg,
            DowncastWarning,
            stacklevel=2,
        )
        return int(self.nom)

    @unsupported_type("vec")
    def __complex__(self) -> complex:
        """
        Apply ``complex`` to the scalar nominal value, discarding its error.

        The conversion emits a downcast warning unless configured to raise.

        :return: The nominal value converted by ``complex``.
        :raises TypeError: The uncertainty is vector-valued.
        :raises auto_uncertainties.exceptions.DowncastError: The settings prohibit discarding uncertainty.
        """
        msg = "The uncertainty is stripped when downcasting to float."
        if get_settings().error_on_downcast:
            raise DowncastError(msg)
        warnings.warn(
            msg,
            DowncastWarning,
            stacklevel=2,
        )
        return complex(self.nom)


@overload
def nominal_values(x: Uncertainty[ValueT]) -> ValueT: ...
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
def std_devs(x: Uncertainty[ValueT]) -> ValueT: ...
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


def normalize(value: Any, error: Any) -> tuple[Any, Any]:
    """
    Normalize unitless inputs without constructing another uncertainty.

    :param value: Scalar, array, sequence, or existing uncertainty.
    :param error: Matching standard deviations or ``None``.
    :return: Validated nominal values and standard deviations.
    :raises TypeError: Input types are unsupported or incompatible.
    :raises auto_uncertainties.exceptions.NegativeStdDevError: A standard deviation is negative.
    :raises ValueError: Component shapes differ.
    """
    if isinstance(value, Uncertainty):
        value, error = value.value, value.error
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value[()]
    if isinstance(error, np.ndarray) and error.ndim == 0:
        error = error[()]
    if isinstance(value, Sequence):
        return normalize_sequence(value, error)
    if isinstance(value, np.ndarray):
        return normalize_array(value, error)
    if not isinstance(value, ScalarT) or (error is not None and not isinstance(error, ScalarT)):
        msg = f"Unsupported argument types (type(value)={type(value)}, type(error)={type(error)})"
        raise TypeError(msg)
    propagate = get_settings().nonfinite_error_policy == "propagate"
    if error is not None and error < 0 and (np.isfinite(error) or propagate):
        msg = f"Got negative value ({error}) for the standard deviation"
        raise NegativeStdDevError(msg)
    caster = np.float64 if isinstance(value, np.number) else float
    if isinstance(value, int | np.integer):
        value = caster(value)
        error = (
            caster(error) if error is not None and (np.isfinite(error) or propagate) else caster(0)
        )
    elif error is None or (not propagate and not np.isfinite(error)):
        error = caster(0)
    elif not (isinstance(value, np.floating) and isinstance(error, np.floating)):
        error = caster(error)
    if propagate and (np.isnan(value) or np.isnan(error)):
        value, error = type(cast("Any", value))(np.nan), type(error)(np.nan)
    return value, error


def normalize_sequence(value: Sequence[Any], error: Any) -> tuple[Any, Any]:
    """
    Convert a sequence into floating-point component arrays.

    :param value: Sequence of numbers or scalar uncertainties.
    :param error: A scalar, a matching sequence, or ``None``.
    :return: Normalized component arrays; uncertain elements supply their own errors.
    :raises TypeError: A component has an unsupported type.
    :raises ValueError: Sequence lengths differ.
    """
    if error is not None and not isinstance(error, Sequence | ScalarT):
        msg = "Error must be a sequence or scalar when value is a sequence"
        raise TypeError(msg)
    if isinstance(error, Sequence) and len(error) != len(value):
        msg = "Error must be the same length as the value sequence"
        raise ValueError(msg)
    nominal = np.empty(len(value), dtype=np.float64)
    deviation = np.empty(len(value), dtype=np.float64)
    for index, element in enumerate(value):
        if isinstance(element, Uncertainty):
            item, item_error = element.value, element.error
        else:
            item = element
            item_error = error[index] if isinstance(error, Sequence) else error
        if not isinstance(item, ScalarT) or (
            item_error is not None and not isinstance(item_error, ScalarT)
        ):
            msg = "Value and error sequences must contain scalars or scalar uncertainties"
            raise TypeError(msg)
        nominal[index], deviation[index] = item, 0 if item_error is None else item_error
    return normalize_array(nominal, deviation)


def normalize_array(value: npt.NDArray[Any], error: Any) -> tuple[Any, Any]:
    """
    Validate array components, copying only when normalization requires it.

    :param value: Nominal array.
    :param error: A matching array, scalar error, or ``None``.
    :return: Normalized nominal and error arrays.
    :raises TypeError: The error type is unsupported.
    :raises ValueError: Component shapes differ.
    :raises auto_uncertainties.exceptions.NegativeStdDevError: A standard deviation is negative.
    """
    if error is None:
        error = np.zeros_like(value)
    elif isinstance(error, ScalarT):
        error = np.ones_like(value) * error
    elif not isinstance(error, np.ndarray):
        msg = "Error must be a NumPy array or scalar when value is a NumPy array"
        raise TypeError(msg)
    elif error.shape != value.shape:
        msg = f"Error must have the same shape as value (got {value.shape} and {error.shape})"
        raise ValueError(msg)
    if get_settings().nonfinite_error_policy == "zero":
        invalid = ~np.isfinite(error)
        if np.any(invalid):
            error = np.where(invalid, 0, error)
    if np.any(error < 0):
        msg = "Found negative values for the standard deviation"
        raise NegativeStdDevError(msg)
    if np.issubdtype(value.dtype, np.integer):
        value = value.astype(np.float64)
    if np.issubdtype(error.dtype, np.integer):
        error = error.astype(np.float64)
    if get_settings().nonfinite_error_policy == "propagate":
        invalid = np.isnan(value) | np.isnan(error)
        if np.any(invalid):
            value, error = np.where(invalid, np.nan, value), np.where(invalid, np.nan, error)
    return value, error


def type_error_msg(u_type: str, operation: str) -> str:
    """
    Describe an operation unsupported by an uncertainty kind.

    :param u_type: Uncertainty kind to display in the message.
    :param operation: Name of the unsupported operation.
    :return: The formatted error message.
    """
    return f"{u_type} Uncertainty objects do not support the '{operation}' operation"
