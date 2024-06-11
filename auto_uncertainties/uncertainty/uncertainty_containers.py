# Based heavily on the implementation of pint's Quantity object
from __future__ import annotations

from collections.abc import Sequence
import copy
import locale
import math
import operator
from typing import Generic, TypeVar
import warnings

import joblib
import numpy as np
from numpy.typing import NDArray

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
]


def set_downcast_error(val: bool):
    """Set whether errors occur when uncertainty is stripped"""
    global ERROR_ON_DOWNCAST
    ERROR_ON_DOWNCAST = val


def set_compare_error(val: float):
    global COMPARE_RTOL
    COMPARE_RTOL = val


def _check_units(value, err):
    mag_has_units = hasattr(value, "units")
    mag_units = getattr(value, "units", None)
    err_has_units = hasattr(err, "units")
    err_units = getattr(err, "units", None)

    if mag_has_units and mag_units is not None:
        Q = mag_units._REGISTRY.Quantity
        ret_val = Q(value).to(mag_units).m
        ret_err = Q(err).to(mag_units).m if err is not None else None
        ret_units = mag_units
    # This branch will never actually work, but its here
    # to raise a Dimensionality error without needing to import pint
    elif err_has_units:
        Q = err_units._REGISTRY.Quantity  # type: ignore
        ret_val = Q(value).to(err_units).m
        ret_err = Q(err).to(err_units).m
        ret_units = err_units
    else:
        ret_units = None
        ret_val = value
        ret_err = err

    return ret_val, ret_err, ret_units


def nominal_values(x) -> T:
    """Return the central value of an Uncertainty object if it is one, otherwise returns the object"""
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


def std_devs(x) -> T:
    """Return the uncertainty of an Uncertainty object if it is one, otherwise returns zero"""
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


ST = TypeVar("ST", float, int)
T = TypeVar("T", NDArray, float, int)


class Uncertainty(Generic[T]):
    """Base class for Uncertainty objects


    Parameters
    ----------

    value :
        The central value(s)

    err:
        The uncertainty value(s). Zero if not provided. Negative numbers raise a RuntimeError.

    """

    _nom: T
    _err: T

    def __getstate__(self):
        return {"nominal_value": self._nom, "std_devs": self._err}

    def __setstate__(self, state):
        self._nom = state["nominal_value"]
        self._err = state["std_devs"]

    def __getnewargs__(self):
        return (self._nom, self._err)

    @ignore_numpy_downcast_warnings
    def __new__(cls: type[Uncertainty], value: T | Uncertainty, err=None):
        # If instantiated with an Uncertainty subclass
        if isinstance(value, ScalarUncertainty | VectorUncertainty):
            err = value.error
            value = value.value
        # If instantiated with a list or tuple of uncertainties
        elif isinstance(value, list | tuple):
            inst = cls.from_list(value)
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

    def __init__(self, value: T, err: T | None, *, trigger=False):
        if trigger:
            if hasattr(value, "units") or hasattr(err, "units"):
                msg = "Uncertainty cannot have units! Call Uncertainty.from_quantities instead."
                raise NotImplementedError(msg)

            self._nom = value
            self._err = err

    def __copy__(self) -> Uncertainty[T]:
        return self.__class__(copy.copy(self._nom), copy.copy(self._err))

    def __deepcopy__(self, memo) -> Uncertainty[T]:
        return self.__class__(
            copy.deepcopy(self._nom, memo), copy.deepcopy(self._err, memo)
        )

    @property
    def value(self) -> T:
        """The central value of the Uncertainty object"""
        return self._nom

    @property
    def error(self) -> T:
        """The uncertainty value of the Uncertainty object"""
        return self._err

    @property
    def relative(self) -> T:
        """The relative uncertainty of the Uncertainty object"""
        raise NotImplementedError

    @property
    def rel(self) -> T:
        """Alias for relative property"""
        return self.relative

    @property
    def rel2(self) -> T:
        """The square of the relative uncertainty of the Uncertainty object"""
        raise NotImplementedError

    def plus_minus(self, err: T):
        """Add an error to the Uncertainty object"""
        val = self._nom
        old_err = self._err
        new_err = np.sqrt(old_err**2 + err**2)

        return self.__class__(val, new_err)

    @classmethod
    def from_string(cls, string: str):
        """Create an Uncertainty object from a string representation of the value and error.

        Parameters
        ----------
        string : str
            A string representation of the value and error. The error can be represented as "+/-" or "±". For instance, 5.0 +- 1.0 or 5.0 ± 1.0.
        """
        new_str = string.replace("+/-", "±")
        new_str = new_str.replace("+-", "±")
        if "±" not in new_str:
            return Uncertainty(float(string))
        else:
            u1, u2 = new_str.split("±")
            return Uncertainty(float(u1), float(u2))

    @classmethod
    def from_quantities(cls, value, err):
        """Create an Uncertainty object from two `Pint` quantities

        Parameters
        ----------
        value : pint.Quantity
            The central value of the Uncertainty object
        err : pint.Quantity
            The uncertainty value of the Uncertainty object
        """
        value_, err_, units = _check_units(value, err)
        inst = cls(value_, err_)
        if units is not None:
            inst *= units
        return inst

    @classmethod
    def from_list(cls, u_list: Sequence[Uncertainty]):
        """Create an Uncertainty object from a list of Uncertainty objects

        Parameters
        ----------
        u_list :
            A list of Uncertainty objects
        """
        return cls.from_sequence(u_list)

    @classmethod
    def from_sequence(cls, seq: Sequence[Uncertainty]):
        """Create an Uncertainty object from a sequence of Uncertainty objects

        Parameters
        ----------
        seq :
            A list of Uncertainty objects
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
                msg = f"Sequence elements of type {type(first_item)} dont support math operations!"
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
        new_mag = other % self._nom
        if np.ndim(new_mag) == 0:
            new_err = 0.0
        elif isinstance(other, self._HANDLED_TYPES):
            new_err = np.zeros_like(new_mag)
        else:
            return NotImplemented
        return self.__class__(new_mag, new_err)

    def __divmod__(self, other):
        return self // other, self % other

    def __rdivmod__(self, other):
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
        new_err = new_err = np.abs(new_mag) * np.sqrt(
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

    def __init__(self, value: T, err: T | None = None, *, trigger=False):
        if trigger:
            super().__init__(value=value, err=err, trigger=trigger)
            if np.ndim(self._nom) == 0:
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

    def clip(self, min=None, max=None, out=None, **kwargs):  # noqa: A002
        """Numpy clip implementation"""
        return self.__class__(self._nom.clip(min, max, out, **kwargs), self._err)

    def fill(self, value) -> None:
        """Numpy fill implementation"""
        return self._nom.fill(value)

    def put(self, indices, values, mode="raise") -> None:
        """Numpy put implementation"""
        if isinstance(values, self.__class__):
            self._nom.put(indices, values._nom, mode)
            self._err.put(indices, values._err, mode)
        else:
            msg = "Can only 'put' Uncertainties into uncertainties!"
            raise TypeError(msg)

    def copy(self):
        """Return a copy of the Uncertainty object"""
        return Uncertainty(self._nom.copy(), self._err.copy())

    # Special properties
    @property
    def flat(self):
        """ "numpy flat implementation"""
        for u, v in (self._nom.flat, self._err.flat):
            yield self.__class__(u, v)

    @property
    def shape(self):
        """Numpy shape implemenetation"""
        return self._nom.shape

    @shape.setter
    def shape(self, value):
        self._nom.shape = value
        self._err.shape = value

    @property
    def nbytes(self):
        """Numpy nbytes implementation"""
        return self._nom.nbytes + self._err.nbytes

    def searchsorted(self, v, side="left", sorter=None):
        """numpy searchsorted implementation"""
        return self._nom.searchsorted(v, side)

    def __len__(self) -> int:
        return len(self._nom)

    def __getitem__(self, key):
        try:
            return Uncertainty(self._nom[key], self._err[key])
        except TypeError:
            msg = f"Index {key} not supported!"
            raise TypeError(msg) from None

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
        except ValueError as exc:
            msg = f"Object {type(self._nom)} does not support indexing"
            raise ValueError(msg) from exc

        if np.size(value._nom) == 1 and np.ndim(value._nom) > 0:
            self._nom[key] = value._nom[0]
            self._err[key] = value._err[0]
        else:
            self._nom[key] = value._nom
            self._err[key] = value._err

    def tolist(self):
        """numpy tolist implementation"""
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
                    for n, e in (nom, err)
                ]
        except AttributeError:
            msg = f"{type(self._nom).__name__}' does not support tolist."
            raise AttributeError(msg) from None

    @property
    def ndim(self):
        """numpy ndim implementation"""
        return np.ndim(self._nom)

    def view(self):
        """numpy view implementation"""
        return self.__class__(self._nom.view(), self._err.view())

    def __hash__(self) -> int:
        digest = joblib.hash((self._nom, self._err), hash_name="sha1")
        return int.from_bytes(bytes(digest, encoding="utf-8"), "big")


class ScalarUncertainty(ScalarDisplay, Uncertainty[ST]):
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
            msg = "The uncertainty is stripped when downcasting to float."
            raise DowncastError(msg)
        else:
            warnings.warn(
                "The uncertainty is stripped when downcasting to float.",
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
