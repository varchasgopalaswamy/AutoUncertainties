# Based heavily on the implementation of pint's Quantity object
from __future__ import annotations

import copy
import locale
import math
import operator
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar
import warnings

import joblib
import numpy as np

from auto_uncertainties import (
    DowncastError,
    DowncastWarning,
    NegativeStdDevError,
)
from auto_uncertainties.display_format import ScalarDisplay, VectorDisplay
from auto_uncertainties.numpy import HANDLED_FUNCTIONS, HANDLED_UFUNCS, wrap_numpy
from auto_uncertainties.util import (
    ignore_numpy_downcast_warnings,
    ignore_runtime_warnings,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pint import Unit
    from pint.facets.plain import PlainQuantity as Quantity

    from auto_uncertainties.pint.extensions import UncertaintyQuantity


ERROR_ON_DOWNCAST = False
COMPARE_RTOL = 1e-9

__all__ = [
    "Uncertainty",
    "VectorUncertainty",
    "ScalarUncertainty",
    "set_downcast_error",
    "set_compare_error",
    "nominal_values",
    "std_devs",
    "UType",
    "SType",
]


UType: type(TypeVar) = TypeVar("UType", np.ndarray, float, int)
"""`TypeVar` specifying the supported underlying types wrapped by `Uncertainty` objects."""

SType: type(TypeVar) = TypeVar("SType", float, int)
"""`TypeVar` specifying the scalar types used by `ScalarUncertainty` objects."""


class Uncertainty(Generic[UType]):
    """
    Base class for `Uncertainty` objects.

    Parameters can be numbers, `numpy` arrays, `pint.Quantity` objects,
    other `Uncertainty` objects, or lists / tuples of `Uncertainty` objects.

    Generally, it is sipmler to let AutoUncertainties determine whether to
    instantiate a `VectorUncertainty` or a `ScalarUncertainty` based on the
    arguments passed to `Uncertainty`:

    .. code-block:: python
       :caption: Example

       # Creates a ScalarUncertainty
       s = Uncertainty(10, 1.5)

       # Creats a VectorUncertainty
       v = Uncertainty(np.array([1, 2, 3]), np.array([1.5, 1.2, 1.1])

    However, users can also directly instantiate `ScalarUncertainty` or
    `VectorUncertainty` objects if necessary:

    .. code-block:: python
       :caption: Example

       s = ScalarUncertainty(10, 1.5)
       v = VectorUncertainty(np.array([1, 2, 3]), np.array([1.5, 1.2, 1.1])

    :param value: The central value(s).
    :param err: The uncertainty value(s). Zero if not provided.

    :raise NegativeStdDevError: If ``err`` is negative, or contains negative values.

    .. note::

       * If `pint.Quantity` objects are supplied for either parameter, the behavior
         is exactly as described in the `from_quantities` method.

       * If an `Uncertainty` is supplied for ``value``, its ``error`` attribute will
         override any ``err`` argument (if it is supplied).

    .. seealso::

        * `from_quantities`
    """

    _nom: UType
    _err: UType

    def __getstate__(self) -> dict[str, UType]:
        return {"nominal_value": self._nom, "std_devs": self._err}

    def __setstate__(self, state) -> None:
        self._nom = state["nominal_value"]
        self._err = state["std_devs"]

    def __getnewargs__(self) -> tuple[UType, UType]:
        return self._nom, self._err

    @ignore_numpy_downcast_warnings
    def __new__(
        cls: type[Uncertainty],
        value: UType | Uncertainty | Sequence[Uncertainty],
        err: UType | None = None,
    ):
        # If instantiated with Quantity objects, call from_quantities
        if hasattr(value, "units") or hasattr(err, "units"):
            return cls.from_quantities(value, err)

        # If instantiated with an Uncertainty subclass
        if isinstance(value, ScalarUncertainty | VectorUncertainty):
            err = value.error
            value = value.value

        # If instantiated with a list or tuple of uncertainties
        elif isinstance(value, list | tuple):
            inst = cls.from_sequence(value)
            value = inst.value
            err = inst.error

        nan = False
        # Numpy arrays
        if np.ndim(value) > 0:
            vector = True
            # Zero error
            if err is None:
                err = np.zeros_like(value)
            else:
                # Constant error
                if np.ndim(err) == 0:
                    err = np.ones_like(value) * err
                else:
                    assert np.ndim(value) == np.ndim(err)
                    assert np.shape(value) == np.shape(err)
            # replace NaN with zero in errors
            err[~np.isfinite(err)] = 0

            if np.any(err < 0):
                msg = f"Found {np.count_nonzero(err < 0)} negative values for the standard deviation!"
                raise NegativeStdDevError(msg)
        else:
            vector = False
            # Zero error
            if err is None:
                err = 0.0
            if np.isfinite(value):
                nan = False
                if np.isfinite(err) and err < 0:
                    msg = f"Found negative value ({err}) for the standard deviation!"
                    raise NegativeStdDevError(msg)
                elif err is None or not np.isfinite(err):
                    err = 0.0
            else:
                nan = True

        if nan:
            inst = np.nan
        else:
            if vector:
                inst = object.__new__(VectorUncertainty)
            else:
                inst = object.__new__(ScalarUncertainty)

            inst.__init__(value, err, trigger=True)

        return inst

    def __init__(
        self,
        value: UType | Uncertainty | Sequence[Uncertainty],
        err: UType | None = None,
        *,
        trigger=False,
    ):
        if trigger:
            if hasattr(value, "units") or hasattr(err, "units"):
                msg = "Parameters 'value' or 'err' should not have the 'units' attribute at this point."
                raise ValueError(msg)

            self._nom = value
            self._err = err

    def __copy__(self) -> Uncertainty[UType]:
        return self.__class__(copy.copy(self._nom), copy.copy(self._err))

    def __deepcopy__(self, memo) -> Uncertainty[UType]:
        return self.__class__(
            copy.deepcopy(self._nom, memo), copy.deepcopy(self._err, memo)
        )

    @property
    def value(self) -> UType:
        """The central value of the `Uncertainty` object."""
        return self._nom

    @property
    def error(self) -> UType:
        """The uncertainty (error) value of the `Uncertainty` object."""
        return self._err

    @property
    def relative(self) -> UType:  # pragma: no cover
        """The relative uncertainty of the `Uncertainty` object."""
        raise NotImplementedError

    @property
    def rel(self) -> UType:
        """Alias for relative property."""
        return self.relative

    @property
    def rel2(self) -> UType:  # pragma: no cover
        """The square of the relative uncertainty of the `Uncertainty` object."""
        raise NotImplementedError

    def plus_minus(self, err: UType):
        """
        Add an error to the `Uncertainty` object.

        Returns a new instance.

        :param err: Error value to add
        """

        val = self._nom
        old_err = self._err
        new_err = np.sqrt(old_err**2 + err**2)

        return self.__class__(val, new_err)

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
    def from_quantities(
        cls, value: Quantity[UType] | UType, err: Quantity[UType] | UType
    ) -> UncertaintyQuantity:
        """
        Create an `Uncertainty` object from one or more `pint.Quantity` objects.

        .. important:: The `pint` package must be installed for this to work.

        :param value: The central value of the `Uncertainty` object
        :param err: The uncertainty value of the `Uncertainty` object

        .. note::

           * If **neither** argument is a `~pint.Quantity`, returns an
             `Uncertainty` object.

           * If **both** arguments are `~pint.Quantity` objects, returns an
             `UncertaintyQuantity` with the same units as ``value`` (attempts
             to convert ``err`` to ``value.units``).

           * If **only the** ``value`` argument is a `~pint.Quantity`, returns
             an `UncertaintyQuantity` object with the same units as ``value``.

           * If **only the** ``err`` argument is a `~pint.Quantity`, returns
             an `UncertaintyQuantity` object with the same units as ``err``.
        """

        value_, err_, units = _check_units(value, err)
        inst = cls(value_, err_)

        from auto_uncertainties.pint.extensions import UncertaintyQuantity

        if units is not None:
            inst = UncertaintyQuantity(inst, units)

        return inst

    def as_quantity(self, unit: str | Unit | None = None) -> UncertaintyQuantity:
        """
        Returns the current object as an `UncertaintyQuantity`.

        This is an alternative to calling `UncertaintyQuantity()` directly.

        .. attention::

           This will **not** create a copy of the underlying `Uncertainty` object.
           It simply returns the current object wrapped in `UncertaintyQuantity`.
           Any changes to the underlying object (such as to the `numpy` arrays of a
           `VectorUncertainty`) will be reflected in the `UncertaintyQuantity`, and vice versa.

        .. important:: The `pint` package must be installed for this to work.

        :param unit: The Pint unit to apply. Can be a string, or a `pint.Unit` object. (Optional)
        """

        from auto_uncertainties.pint.extensions import UncertaintyQuantity

        return UncertaintyQuantity(self, unit)

    @classmethod
    def from_list(cls, u_list: Sequence[Uncertainty]):  # pragma: no cover
        """
        Alias for `from_sequence`.

        :param u_list: A list of `Uncertainty` objects.
        """
        return cls.from_sequence(u_list)

    @classmethod
    def from_sequence(cls, seq: Sequence[Uncertainty]):
        """
        Create an `Uncertainty` object from a sequence of `Uncertainty` objects.

        :param seq: A sequence of `Uncertainty` objects.
        """
        _ = iter(seq)

        len_seq = len(seq)
        val = np.empty(len_seq)
        err = np.empty(len_seq)
        if len_seq > 0:
            first_item = seq[0]
            try:
                first_item + 1
            except TypeError:
                msg = f"Sequence elements of type {type(first_item)} don't support math operations!"
                raise TypeError(msg) from None
            if hasattr(first_item, "units"):
                val *= first_item.units
                err *= first_item.units
            for i, seq_i in enumerate(seq):
                try:
                    val[i] = float(seq_i._nom)
                    err[i] = float(seq_i._err)
                except AttributeError:
                    val[i] = float(seq_i)
                    err[i] = 0

        return cls(val, err)

    _HANDLED_TYPES = (np.ndarray, float, int)

    # Math Operators
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
            new_err = np.abs(new_mag) * np.sqrt(self.rel2 + other.rel2)
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
            new_err = np.abs(new_mag) * np.sqrt(self.rel2 + other.rel2)
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
        new_err = 0.0 if np.ndim(new_mag) == 0 else np.zeros_like(new_mag)
        return self.__class__(new_mag, new_err)

    def __rmod__(self, other):
        if isinstance(other, self._HANDLED_TYPES):
            new_mag = other % self._nom
            if np.ndim(new_mag) == 0:
                new_err = 0.0
            else:
                new_err = np.zeros_like(new_mag)
            return self.__class__(new_mag, new_err)
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

    # NumPy function/ufunc support
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

    def __getattr__(self, item):
        if item.startswith("__array_"):
            # Handle array protocol attributes other than `__array__`
            msg = f"Array protocol attribute {item} not available."
            raise AttributeError(msg)
        elif item in HANDLED_UFUNCS:
            return lambda *args, **kwargs: wrap_numpy(
                "ufunc", item, [self, *list(args)], kwargs
            )
        elif item in HANDLED_FUNCTIONS:
            return lambda *args, **kwargs: wrap_numpy(
                "function", item, [self, *list(args)], kwargs
            )
        else:
            msg = f"Attribute {item} not available in Uncertainty, or as NumPy ufunc or function."
            raise AttributeError(msg) from None


class VectorUncertainty(VectorDisplay, Uncertainty[np.ndarray]):
    """Vector `Uncertainty` class."""

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

    # More numpy capabilities exposed here
    def __getattr__(self, item):
        if item.startswith("__array_"):
            # Handle array protocol attributes other than `__array__`
            msg = f"Array protocol attribute {item} not available."
            raise AttributeError(msg)
        elif item in self.__apply_to_both_ndarray__:
            val = getattr(self._nom, item)
            err = getattr(self._err, item)

            if callable(val):
                return lambda *args, **kwargs: self.__class__(
                    val(*args, **kwargs), err(*args, **kwargs)
                )
            else:
                return self.__class__(val, err)
        elif item in HANDLED_UFUNCS:
            return lambda *args, **kwargs: wrap_numpy(
                "ufunc", item, [self, *list(args)], kwargs
            )
        elif item in HANDLED_FUNCTIONS:
            return lambda *args, **kwargs: wrap_numpy(
                "function", item, [self, *list(args)], kwargs
            )
        elif item in self.__ndarray_attributes__:
            return getattr(self._nom, item)
        else:
            msg = f"Attribute {item} not available in Uncertainty, or as NumPy ufunc or function."
            raise AttributeError(msg) from None

    def __init__(
        self,
        value: UType | Uncertainty | Sequence[Uncertainty],
        err: UType | None = None,
        *,
        trigger=False,
    ):
        if trigger:
            super().__init__(value=value, err=err, trigger=trigger)

            # This should not be executed, as the parent class should account for this
            if np.ndim(self._nom) == 0:  # pragma: no cover
                msg = "VectorUncertainty must have a dimension greater than 0!"
                raise ValueError(msg)

    def __ne__(self, other):
        out = self.__eq__(other)
        return np.logical_not(out)

    def __bytes__(self) -> bytes:
        return str(self).encode(locale.getpreferredencoding())

    def __iter__(self):
        for v, e in zip(self._nom, self._err, strict=False):
            yield self.__class__(v, e)

    def __eq__(self, other):
        if isinstance(other, Uncertainty):
            ret = self._nom == other._nom
        else:
            ret = self._nom == other
        return ret

    @property
    def relative(self):
        rel = np.zeros_like(self._nom)
        valid = np.isfinite(self._nom) & (self._nom > 0)
        rel[valid] = self._err[valid] / self._nom[valid]
        return rel

    @property
    def rel2(self):
        return self.relative**2

    def __round__(self, ndigits):
        return self.__class__(np.round(self._nom, decimals=ndigits), self._err)

    def __array__(self, t=None) -> np.ndarray:
        if ERROR_ON_DOWNCAST:
            msg = "The uncertainty is stripped when downcasting to ndarray."
            raise DowncastError(msg)
        else:
            warnings.warn(
                "The uncertainty is stripped when downcasting to ndarray.",
                DowncastWarning,
                stacklevel=2,
            )
            return np.asarray(self._nom)

    def clip(self, min=None, max=None, out=None, **kwargs) -> Uncertainty:  # noqa: A002
        """NumPy `~numpy.ndarray.clip` implementation."""
        return self.__class__(self._nom.clip(min, max, out, **kwargs), self._err)

    def fill(self, value) -> None:
        """NumPy `~numpy.ndarray.fill` implementation."""
        return self._nom.fill(value)

    def put(
        self, indices, values, mode: Literal["raise", "wrap", "clip"] = "raise"
    ) -> None:
        """NumPy `~numpy.ndarray.put` implementation."""
        if isinstance(values, self.__class__):
            self._nom.put(indices, values._nom, mode)
            self._err.put(indices, values._err, mode)
        else:
            msg = "Can only 'put' Uncertainties into uncertainties!"
            raise TypeError(msg)

    def copy(self):
        """Return a copy of the `Uncertainty` object."""
        return Uncertainty(self._nom.copy(), self._err.copy())

    # Special properties
    @property
    def flat(self):
        """NumPy `~numpy.ndarray.flat` implementation."""
        for u, v in zip(self._nom.flat, self._err.flat, strict=False):
            yield self.__class__(u, v)

    @property
    def shape(self):
        """NumPy `~numpy.ndarray.shape` implemenetation."""
        return self._nom.shape

    @shape.setter
    def shape(self, value):
        self._nom.shape = value
        self._err.shape = value

    @property
    def nbytes(self):
        """NumPy `~numpy.ndarray.nbytes` implementation."""
        return self._nom.nbytes + self._err.nbytes

    def searchsorted(self, v, side: Literal["left", "right"] = "left", sorter=None):
        """NumPy `~numpy.ndarray.searchsorted` implementation."""
        return self._nom.searchsorted(v, side)

    def __len__(self) -> int:
        return len(self._nom)

    def __getitem__(self, key):
        try:
            return Uncertainty(self._nom[key], self._err[key])
        except IndexError as e:
            msg = f"Index {key} not supported!"
            raise IndexError(msg) from e

    def __setitem__(self, key, value):
        # If value is nan, just set the value in those regions to nan and return. This is the only case where a scalar can be passed as an argument!
        if not isinstance(value, Uncertainty):
            if not np.isfinite(value):
                self._nom[key] = value
                self._err[key] = 0
                return
            else:
                msg = f"Can only pass Uncertainty type to __setitem__! Instead passed {type(value)}"
                raise ValueError(msg)

        try:
            _ = self._nom[key]
        except TypeError as exc:
            msg = f"Object {type(self._nom)} does not support indexing"
            raise ValueError(msg) from exc

        if np.size(value._nom) == 1 and np.ndim(value._nom) > 0:
            self._nom[key] = value._nom[0]
            self._err[key] = value._err[0]
        else:
            self._nom[key] = value._nom
            self._err[key] = value._err

    def tolist(self):
        """NumPy `~numpy.ndarray.tolist` implementation."""
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

    @property
    def ndim(self):
        """NumPy `~numpy.ndarray.ndim` implementation."""
        return np.ndim(self._nom)

    def view(self):
        """NumPy `~numpy.ndarray.view` implementation."""
        return self.__class__(self._nom.view(), self._err.view())

    def __hash__(self) -> int:
        digest = joblib.hash((self._nom, self._err), hash_name="sha1")
        return int.from_bytes(bytes(digest, encoding="utf-8"), "big")


class ScalarUncertainty(ScalarDisplay, Uncertainty[SType]):
    """Scalar `Uncertainty` class."""

    @property
    def relative(self):
        try:
            return self._err / self._nom
        except OverflowError:
            return np.inf
        except ZeroDivisionError:
            return np.nan

    def __float__(self):
        if ERROR_ON_DOWNCAST:
            msg = "The uncertainty is stripped when downcasting to float."
            raise DowncastError(msg)
        else:
            warnings.warn(
                "The uncertainty is stripped when downcasting to float.",
                DowncastWarning,
                stacklevel=2,
            )

        return float(self._nom)

    def __int__(self):
        if ERROR_ON_DOWNCAST:
            msg = "The uncertainty is stripped when downcasting to int."
            raise DowncastError(msg)
        else:
            warnings.warn(
                "The uncertainty is stripped when downcasting to int.",
                DowncastWarning,
                stacklevel=2,
            )
        return int(self._nom)

    def __complex__(self):
        if ERROR_ON_DOWNCAST:
            msg = "The uncertainty is stripped when downcasting to float."
            raise DowncastError(msg)
        else:
            warnings.warn(
                "The uncertainty is stripped when downcasting to float.",
                DowncastWarning,
                stacklevel=2,
            )
        return complex(self._nom)

    def __round__(self, ndigits):
        return self.__class__(round(self._nom, ndigits=ndigits), self._err)

    @property
    def rel2(self):
        try:
            return self.relative**2
        except OverflowError:
            return np.inf

    def __ne__(self, other):
        out = self.__eq__(other)
        return not out

    def __eq__(self, other):
        if isinstance(other, Uncertainty):
            try:
                ret = math.isclose(self._nom, other._nom, rel_tol=COMPARE_RTOL)
            except TypeError:
                ret = self._nom == other._nom
        else:
            try:
                ret = math.isclose(self._nom, other, rel_tol=COMPARE_RTOL)
            except TypeError:
                ret = self._nom == other
        return ret

    def __hash__(self) -> int:
        return hash((self._nom, self._err))


def set_downcast_error(val: bool) -> None:
    """Set whether errors occur when uncertainty is stripped."""
    global ERROR_ON_DOWNCAST
    ERROR_ON_DOWNCAST = val


def set_compare_error(val: float) -> None:  # pragma: no cover
    global COMPARE_RTOL
    COMPARE_RTOL = val


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


def nominal_values(x) -> UType:
    """Return the central value of an `Uncertainty` object if it is one, otherwise returns the object."""
    # Is an Uncertainty
    if hasattr(x, "_nom"):
        return x.value
    else:
        if np.ndim(x) > 0:
            try:
                x2 = Uncertainty.from_sequence(x)
            except Exception:
                return x
            else:
                return x2.value
        else:
            try:
                x2 = Uncertainty(x)
            except Exception:
                return x
            else:
                if isinstance(x2, float):
                    return x2
                else:
                    return x2.value


def std_devs(x) -> UType:
    """Return the uncertainty of an `Uncertainty` object if it is one, otherwise returns zero."""
    # Is an Uncertainty
    if hasattr(x, "_err"):
        return x.error
    else:
        if np.ndim(x) > 0:
            try:
                x2 = Uncertainty.from_sequence(x)
            except Exception:
                return np.zeros_like(x)
            else:
                return x2.error
        else:
            try:
                x2 = Uncertainty(x)
            except Exception:
                return 0
            else:
                if isinstance(x2, float):
                    return 0
                else:
                    return x2.error
