# -*- coding: utf-8 -*-
# Based heavily on the implementation of pint's Quantity object
from __future__ import annotations

import copy
import locale
import operator
import warnings

import joblib
import numpy as np
from numpy.typing import NDArray
from typing_extensions import Generic, Type, TypeVar

from . import NegativeStdDevError, NumpyDowncastWarning
from .display_format import ScalarDisplay, VectorDisplay
from .util import (
    ignore_numpy_downcast_warnings,
    ignore_runtime_warnings,
    strip_device_array,
)
from .wrap_numpy import HANDLED_FUNCTIONS, HANDLED_UFUNCS, wrap_numpy

ERROR_ON_DOWNCAST = False


def set_downcast_error(val: bool):
    global ERROR_ON_DOWNCAST
    ERROR_ON_DOWNCAST = val


def _check_units(value, err):
    mag_has_units = hasattr(value, "units")
    mag_units = getattr(value, "units", None)
    err_has_units = hasattr(err, "units")
    err_units = getattr(err, "units", None)

    if mag_has_units and mag_units is not None:
        Q = mag_units._REGISTRY.Quantity
        ret_val = Q(value).to(mag_units).m
        if err is not None:
            ret_err = Q(err).to(mag_units).m
        else:
            ret_err = None
        ret_units = mag_units
    # This branch will never actually work, but its here
    # to raise a Dimensionality error without needing to import pint
    elif err_has_units:
        Q = err_units._REGISTRY.Quantity
        ret_val = Q(value).to(err_units).m
        ret_err = Q(err).to(err_units).m
        ret_units = err_units
    else:
        ret_units = None
        ret_val = value
        ret_err = err

    return ret_val, ret_err, ret_units


ST = TypeVar("ST", float, int)
T = TypeVar("T", NDArray, float, int)


class Uncertainty(Generic[T]):
    _nom: T
    _err: T

    @ignore_numpy_downcast_warnings
    def __new__(cls: Type[Uncertainty], value: T | Uncertainty, err=None):
        # If instantiated with an Uncertainty subclass
        if isinstance(value, cls):
            value = value.value
            err = value.error
        # If instantiated with a list or tuple of uncertainties
        elif isinstance(value, (list, tuple)):
            inst = cls.from_list(value)
            value = inst.value
            err = inst.error

        value = strip_device_array(value)
        if err is not None:
            err = strip_device_array(err)

        # Numpy arrays
        if np.ndim(value) > 0:
            vector = True
            value = np.asarray(value)
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
                    err = np.asarray(err)

            # replace NaN with zero in errors
            err[~np.isfinite(err)] = 0

            if np.any(err < 0):
                raise NegativeStdDevError(
                    f"Found {np.count_nonzero(err < 0)} negative values for the standard deviation!"
                )
        else:
            vector = False
            if err is None or not np.isfinite(err):
                err = 0.0
            elif err < 0:
                raise NegativeStdDevError(
                    f"Found negative value ({err}) for the standard deviation!"
                )

        if vector:
            inst = object.__new__(VectorUncertainty)
        else:
            inst = object.__new__(ScalarUncertainty)
        inst.__init__(value, err)
        return inst

    def __init__(self, value: T, err: T | None):
        if hasattr(value, "units") or hasattr(err, "units"):
            raise NotImplementedError(
                "Uncertainty cannot have units! Call Uncertainty.from_quantities instead."
            )
        self._nom = value
        self._err = err

    def __copy__(self) -> Uncertainty[T]:
        ret = self.__class__(copy.copy(self._nom), copy.copy(self._err))

        return ret  # type: ignore

    def __deepcopy__(self, memo) -> Uncertainty[T]:
        ret = self.__class__(
            copy.deepcopy(self._nom, memo), copy.deepcopy(self._err, memo)
        )
        return ret  # type: ignore

    @property
    def value(self):
        return self._nom

    @property
    def error(self):
        return self._err

    def __hash__(self) -> int:
        digest = joblib.hash((self._nom, self._err), hash_name="sha1")
        return int.from_bytes(bytes(digest, encoding="utf-8"), "big")

    @property
    def relative(self) -> T:
        raise NotImplementedError

    @property
    def rel(self):
        return self.relative

    @property
    def rel2(self) -> T:
        raise NotImplementedError

    def plus_minus(self, err: T):
        val = self._nom
        old_err = self._err
        new_err = np.sqrt(old_err**2 + err**2)

        return self.__class__(val, new_err)

    @classmethod
    def from_string(cls, string: str):
        new_str = string.replace("+/-", "±")
        new_str = new_str.replace("+-", "±")
        if "±" not in new_str:
            return Uncertainty(float(string))
        else:
            u1, u2 = new_str.split("±")
            return cls(float(u1), float(u2))

    @classmethod
    def from_quantities(cls, value, err):
        value_, err_, units = _check_units(value, err)
        inst = cls(value_, err_)
        if units is not None:
            inst *= units
        return inst

    @classmethod
    def from_list(cls, u_list):
        return cls.from_sequence(u_list)

    @classmethod
    def from_sequence(cls, seq):
        _ = iter(seq)

        len_seq = len(seq)
        val = np.empty(len_seq)
        err = np.empty(len_seq)
        if len_seq > 0:
            first_item = seq[0]
            try:
                first_item + 1
            except TypeError:
                raise TypeError(
                    f"Sequence elements of type {type(first_item)} dont support math operations!"
                )
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

    # Math Operators
    def __add__(self, other):
        if isinstance(other, Uncertainty):
            new_mag = self._nom + other._nom
            new_err = np.sqrt(self._err**2 + other._err**2)
        else:
            new_mag = self._nom + other
            new_err = self._err
        try:
            return self.__class__(new_mag, new_err)
        except NotImplementedError:
            return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Uncertainty):
            new_mag = self._nom - other._nom
            new_err = np.sqrt(self._err**2 + other._err**2)
        else:
            new_mag = self._nom - other
            new_err = self._err
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
        else:
            new_mag = self._nom * other
            new_err = np.abs(self._err * other)
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
        else:
            new_mag = self._nom / other
            new_err = np.abs(self._err / other)
        try:
            return self.__class__(new_mag, new_err)
        except NotImplementedError:
            return NotImplemented

    @ignore_runtime_warnings
    def __rtruediv__(self, other):
        # Other / Self
        if isinstance(other, Uncertainty):
            raise Exception
        else:
            new_mag = other / self._nom
            new_err = np.abs(new_mag) * np.abs(self.rel)
        try:
            return self.__class__(new_mag, new_err)
        except NotImplementedError:
            return NotImplemented

    __div__ = __truediv__
    __rdiv__ = __rtruediv__

    def __floordiv__(self, other):
        if isinstance(other, Uncertainty):
            new_mag = self._nom // other._nom
            new_err = 0.0
        else:
            new_mag = self._nom // other
            new_err = 0.0
        return self.__class__(new_mag, new_err)

    def __rfloordiv__(self, other):
        if isinstance(other, Uncertainty):
            return other.__truediv__(self)
        else:
            new_mag = other // self._nom
            new_err = 0.0
            return self.__class__(new_mag, new_err)

    def __mod__(self, other):
        if isinstance(other, Uncertainty):
            new_mag = self._nom % other._nom
        else:
            new_mag = self._nom % other
        if np.ndim(new_mag) == 0:
            new_err = 0.0
        else:
            new_err = np.zeros_like(new_mag)
        return self.__class__(new_mag, new_err)

    def __rmod__(self, other):
        new_mag = other % self._nom
        if np.ndim(new_mag) == 0:
            new_err = 0.0
        else:
            new_err = np.zeros_like(new_mag)
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
        else:
            B = other
        new_mag = A**B
        new_err = np.abs(new_mag) * np.sqrt((B / A * sA) ** 2)

        return self.__class__(new_mag, new_err)

    @ignore_runtime_warnings
    def __rpow__(self, other):
        # Other ** self
        B = self._nom
        sB = self._err
        if isinstance(other, Uncertainty):
            A = other._nom
            sA = other._err
        else:
            A = other
            sA = 0

        new_mag = A**B
        new_err = np.abs(new_mag) * np.sqrt(
            (B / A * sA) ** 2 + (np.log(A) * sB) ** 2
        )

        return self.__class__(new_mag, new_err)

    def __abs__(self):
        return self.__class__(abs(self._nom), self._err)

    def __pos__(self):
        return self.__class__(operator.pos(self._nom), self._err)

    def __neg__(self):
        return self.__class__(operator.neg(self._nom), self._err)

    def __eq__(self, other):
        if isinstance(other, Uncertainty):
            return self._nom == other._nom
        else:
            return self._nom == other

    def compare(self, other, op):
        if isinstance(other, Uncertainty):
            return op(self._nom, other._nom)
        else:
            return op(self._nom, other)

    __lt__ = lambda self, other: self.compare(  # noqa: E731
        other, op=operator.lt
    )
    __le__ = lambda self, other: self.compare(  # noqa: E731
        other, op=operator.le
    )
    __ge__ = lambda self, other: self.compare(  # noqa: E731
        other, op=operator.ge
    )
    __gt__ = lambda self, other: self.compare(  # noqa: E731
        other, op=operator.gt
    )

    def __bool__(self) -> bool:
        return bool(self._nom)

    __nonzero__ = __bool__


class VectorUncertainty(VectorDisplay, Uncertainty[np.ndarray]):
    __apply_to_both_ndarray__ = [
        "flatten",
        "real",
        "imag",
        "astype",
        "T",
        "reshape",
    ]
    __ndarray_attributes__ = ["dtype", "ndim", "size"]

    __array_priority__ = 18

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if np.ndim(self._nom) == 0:
            raise ValueError(
                "VectorUncertainty must have a dimension greater than 0!"
            )

    def __ne__(self, other):
        out = self.__eq__(other)
        return np.logical_not(out)

    def __bytes__(self) -> bytes:
        return str(self).encode(locale.getpreferredencoding())

    def __iter__(self):
        for v, e in zip(self._nom, self._err):
            yield self.__class__(v, e)

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

    # NumPy function/ufunc support
    @ignore_runtime_warnings
    def __array_function__(self, func, types, args, kwargs):
        if func.__name__ not in HANDLED_FUNCTIONS:
            return NotImplemented
        elif not any(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        else:
            return wrap_numpy("function", func, args, kwargs)

    @ignore_runtime_warnings
    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if method != "__call__":
            raise NotImplementedError
        else:
            if ufunc.__name__ not in HANDLED_UFUNCS:
                raise NotImplementedError(
                    f"Ufunc {ufunc.__name__} is not implemented!"
                ) from None
            else:
                return wrap_numpy("ufunc", ufunc, args, kwargs)

    def __getattr__(self, item):
        if item.startswith("__array_"):
            # Handle array protocol attributes other than `__array__`
            raise AttributeError(
                f"Array protocol attribute {item} not available."
            )
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
                "ufunc", item, [self] + list(args), kwargs
            )
        elif item in HANDLED_FUNCTIONS:
            return lambda *args, **kwargs: wrap_numpy(
                "function", item, [self] + list(args), kwargs
            )
        elif item in self.__ndarray_attributes__:
            return getattr(self._nom, item)
        else:
            raise AttributeError(
                f"Attribute {item} not available in Uncertainty, or as NumPy ufunc or function."
            ) from None

    def __array__(self, t=None) -> np.ndarray:
        if ERROR_ON_DOWNCAST:
            raise Exception(
                "The uncertainty is stripped when downcasting to ndarray."
            )
        else:
            warnings.warn(
                "The uncertainty is stripped when downcasting to ndarray.",
                NumpyDowncastWarning,
                stacklevel=2,
            )
            return np.asarray(self._nom)

    def clip(self, min=None, max=None, out=None, **kwargs):
        return self.__class__(
            self._nom.clip(min, max, out, **kwargs), self._err
        )

    def fill(self, value) -> None:
        return self._nom.fill(value)

    def put(self, indices, values, mode="raise") -> None:
        if isinstance(values, self.__class__):
            self._nom.put(indices, values._nom, mode)
            self._err.put(indices, values._err, mode)
        else:
            raise ValueError(
                "Can only 'put' Uncertainties into uncertainties!"
            )

    def copy(self):
        return Uncertainty(self._nom.copy(), self._err.copy())

    # Special properties
    @property
    def flat(self):
        for u, v in (self._nom.flat, self._err.flat):
            yield self.__class__(u, v)

    @property
    def shape(self):
        return self._nom.shape

    @shape.setter
    def shape(self, value):
        self._nom.shape = value
        self._err.shape = value

    @property
    def nbytes(self):
        return self._nom.nbytes + self._err.nbytes

    def searchsorted(self, v, side="left", sorter=None):
        return self._nom.searchsorted(v, side)

    def __len__(self) -> int:
        return len(self._nom)

    def __getitem__(self, key):
        try:
            return Uncertainty(self._nom[key], self._err[key])
        except TypeError:
            raise TypeError(f"Index {key} not supported!")

    def __setitem__(self, key, value):
        if not isinstance(value, Uncertainty):
            raise ValueError(
                f"Can only pass Uncertainty type to __setitem__! Instead passed {type(value)}"
            )
        try:
            _ = self._nom[key]
        except ValueError as exc:
            raise ValueError(
                f"Object {type(self._nom)} does not support indexing"
            ) from exc

        if np.size(value._nom) == 1 and np.ndim(value._nom) > 0:
            self._nom[key] = value._nom[0]
            self._err[key] = value._err[0]
        else:
            self._nom[key] = value._nom
            self._err[key] = value._err

    def tolist(self):
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
            raise AttributeError(
                f"{type(self._nom).__name__}' does not support tolist."
            )

    @property
    def ndim(self):
        return np.ndim(self._nom)

    def view(self):
        return self.__class__(self._nom.view(), self._err.view())


class ScalarUncertainty(ScalarDisplay, Uncertainty[ST]):
    @property
    def relative(self):
        try:
            return self._err / self._nom
        except OverflowError:
            return np.inf
        except ZeroDivisionError:
            return np.NaN

    def __float__(self) -> ScalarUncertainty:
        return ScalarUncertainty(float(self.value), float(self.error))

    def __int__(self) -> ScalarUncertainty[int]:
        return ScalarUncertainty(int(self.value), int(self.error))

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
