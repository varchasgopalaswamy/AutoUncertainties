# -*- coding: utf-8 -*-
# Based heavily on the implementation of pint's Quantity object
from __future__ import annotations
from types import MethodType

import numpy as np
import locale
import copy
import operator
import warnings

from .wrap_numpy import wrap_numpy, HANDLED_FUNCTIONS, HANDLED_UFUNCS
from . import NegativeStdDevError
from .util import is_np_duck_array


class Uncertainty(object):
    __apply_to_both_ndarray__ = ["flatten", "real", "imag", "astype", "T"]
    __ndarray_attributes__ = ["dtype", "ndim", "size"]

    def __init__(self, value, err=None):
        if isinstance(value, self.__class__):
            magnitude_nom = value.value
            magnitude_err = value.error
        elif np.ndim(value) > 0:
            magnitude_nom = np.asarray(value)
            if err is None:
                magnitude_err = np.zeros_like(value)
            else:
                magnitude_err = np.asarray(err)
        else:
            magnitude_nom = value
            if err is None:
                magnitude_err = 0.0
            else:
                magnitude_err = err

        # Basic sanity checks
        if is_np_duck_array(type(magnitude_nom)):
            for item in self.__ndarray_attributes__ + ["shape"]:
                if not getattr(magnitude_nom, item) == getattr(magnitude_err, item):
                    raise ValueError(
                        f"Attribute {item} does not match for value and error! ({getattr(magnitude_nom,item)} and {getattr(magnitude_err,item)})"
                    )
        if np.any(np.atleast_1d(magnitude_err) < 0):
            raise NegativeStdDevError

        self._nom = magnitude_nom
        self._err = magnitude_err

    def __str__(self) -> str:
        if self._nom is not None:
            if self._err is not None:
                return f"{self._nom} +/- {self._err}"
            else:
                return f"{self._nom}"

    def __repr__(self) -> str:
        return str(self)

    def __bytes__(self) -> bytes:
        return str(self).encode(locale.getpreferredencoding())

    def __iter__(self):
        for v, e in zip(self._nom, self._err):
            yield self.__class__(v, e)

    def __copy__(self) -> Uncertainty:
        ret = self.__class__(copy.copy(self._nom), copy.copy(self._err))

        return ret

    def __deepcopy__(self, memo) -> Uncertainty:
        ret = self.__class__(copy.deepcopy(self._nom, memo), copy.deepcopy(self._err, memo))
        return ret

    def __hash__(self) -> int:
        return hash((self.__class__, self._nom, self._err))

    @property
    def value(self):
        return self._nom

    @property
    def nominal_value(self):
        return self.value

    @property
    def error(self):
        return self._err

    @property
    def std_dev(self):
        return self.error

    @property
    def relative(self):
        if self._nom != 0:
            return self._err / self._nom
        else:
            return np.NaN

    @property
    def rel(self):
        return self.relative

    @classmethod
    def from_list(cls, u_list):
        return cls.from_sequence(u_list)

    @classmethod
    def from_sequence(cls, seq):

        len_seq = len(seq)
        val = np.empty(len_seq)
        err = np.empty(len_seq)

        for i, seq_i in enumerate(seq):
            val[i] = seq_i._nom
            err[i] = seq_i._err

        return cls(val, err)

    def __float__(self) -> Uncertainty:
        return float(self._nom)

    def __complex__(self) -> Uncertainty:
        return complex(self._nom)

    def __int__(self) -> Uncertainty:
        return int(self._nom)

    # Math Operators
    def __iadd__(self, other):
        new = self + other
        if is_np_duck_array(type(self._nom)):
            self._err = new._err
            self._nom = new._nom
            return self
        else:
            return new

    def __add__(self, other):
        if isinstance(other, Uncertainty):
            new_mag = self._nom + other._nom
            new_err = np.sqrt(self._err ** 2 + other._err ** 2)
        else:
            new_mag = self._nom + other
            new_err = self._err
        return self.__class__(new_mag, new_err)

    __radd__ = __add__

    def __isub__(self, other):
        new = self - other
        if is_np_duck_array(type(self._nom)):
            self._err = new._err
            self._nom = new._nom
            return self
        else:
            return new

    def __sub__(self, other):
        if isinstance(other, Uncertainty):
            new_mag = self._nom - other._nom
            new_err = np.sqrt(self._err ** 2 + other._err ** 2)
        else:
            new_mag = self._nom - other
            new_err = self._err
        return self.__class__(new_mag, new_err)

    def __rsub__(self, other):
        -self.__sub__(other)

    def __imul__(self, other):
        new = self * other
        if is_np_duck_array(type(self._nom)):
            self._err = new._err
            self._nom = new._nom
            return self
        else:
            return new

    def __mul__(self, other):
        if isinstance(other, Uncertainty):
            new_mag = self._nom * other._nom
            new_err = new_mag * np.sqrt(self.rel ** 2 + other.rel ** 2)
        else:
            new_mag = self._nom * other
            new_err = self._err * other
        return self.__class__(new_mag, new_err)

    __rmul__ = __mul__

    def __itruediv__(self, other):
        new = self / other
        if is_np_duck_array(type(self._nom)):
            self._err = new._err
            self._nom = new._nom
            return self
        else:
            return new

    def __truediv__(self, other):
        # Self / Other
        if isinstance(other, Uncertainty):
            new_mag = self._nom / other._nom
            new_err = new_mag * np.sqrt(self.rel ** 2 + other.rel ** 2)
        else:
            new_mag = self._nom / other
            new_err = self._err / other
        return self.__class__(new_mag, new_err)

    def __rtruediv__(self, other):
        # Other / Self
        if isinstance(other, Uncertainty):
            raise Exception
        else:
            new_mag = other / self._nom
            new_err = new_mag * self.rel
            return self.__class__(new_mag, new_err)

    __div__ = __truediv__
    __rdiv__ = __rtruediv__
    __idiv__ = __itruediv__

    def __ifloordiv__(self, other):
        new = self // other
        if is_np_duck_array(type(self._nom)):
            self._err = new._err
            self._nom = new._nom
            return self
        else:
            return new

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

    def __imod__(self, other):
        new = self % other
        if is_np_duck_array(type(self._nom)):
            self._err = new._err
            self._nom = new._nom
            return self
        else:
            return new

    def __mod__(self, other):
        if isinstance(other, Uncertainty):
            new_mag = self._nom % other._nom
            new_err = 0.0
        else:
            new_mag = self._nom % other
            new_err = 0.0
        return self.__class__(new_mag, new_err)

    def __rmod__(self, other):
        new_mag = other % self._nom
        new_err = 0.0
        return self.__class__(new_mag, new_err)

    def __divmod__(self, other):
        return self // other, self % other

    def __rdivmod__(self, other):
        return other // self, other % self

    def __ipow__(self, other):
        new = self ** other
        if is_np_duck_array(type(self._nom)):
            self._err = new._err
            self._nom = new._nom
            return self
        else:
            return new

    def __pow__(self, other):
        # Self ** other
        if isinstance(other, Uncertainty):
            new_mag = self._nom % other._nom
            new_err = 0.0
        else:
            new_mag = self._nom % other
            new_err = 0.0
        return self.__class__(new_mag, new_err)

    def __rpow__(self, other):
        # Other ** self
        new_mag = other % self._nom
        new_err = 0.0
        return self.__class__(new_mag, new_err)

    def __abs__(self):
        return self.__class__(abs(self._nom), self._err)

    def __round__(self, ndigits):
        return self.__class__(round(self._nom, ndigits=ndigits), self._err)

    def __pos__(self):
        return self.__class__(operator.pos(self._nom), self._err)

    def __neg__(self):
        return self.__class__(operator.neg(self._nom), self._err)
        return self.__class__(operator.neg(self._nom), self._err)

    def __eq__(self, other):
        if isinstance(other, Uncertainty):
            return self._nom == other._nom
        else:
            return self._nom == other

    def __ne__(self, other):
        out = self.__eq__(other)
        if is_np_duck_array(type(out)):
            return np.logical_not(out)
        else:
            return not out

    def compare(self, other, op):
        if isinstance(other, Uncertainty):
            return op(self._nom, other._nom)
        else:
            return op(self._nom, other)

    __lt__ = lambda self, other: self.compare(other, op=operator.lt)
    __le__ = lambda self, other: self.compare(other, op=operator.le)
    __ge__ = lambda self, other: self.compare(other, op=operator.ge)
    __gt__ = lambda self, other: self.compare(other, op=operator.gt)

    def __bool__(self) -> bool:
        return bool(self._nom)

    __nonzero__ = __bool__

    # NumPy function/ufunc support
    __array_priority__ = 17

    def __array_function__(self, func, types, args, kwargs):
        # print(func)
        if func.__name__ not in HANDLED_FUNCTIONS:
            return NotImplemented
        elif not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        else:
            return wrap_numpy("function", func, args, kwargs)

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        # print(method,ufunc.__name__)
        if method != "__call__":
            raise NotImplementedError
        else:
            if ufunc.__name__ not in HANDLED_UFUNCS:
                raise NotImplementedError
            else:
                return wrap_numpy("ufunc", ufunc, args, kwargs)

    def __getattr__(self, item):
        if item.startswith("__array_"):
            # Handle array protocol attributes other than `__array__`
            raise AttributeError(f"Array protocol attribute {item} not available.")
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
            return lambda *args, **kwargs: wrap_numpy("ufunc", item, [self] + list(args), kwargs)
        elif item in HANDLED_FUNCTIONS:
            return lambda *args, **kwargs: wrap_numpy("function", item, [self] + list(args), kwargs)
        elif item in self.__ndarray_attributes__:
            return getattr(self._nom, item)
        else:
            raise AttributeError

    def __array__(self, t=None) -> np.ndarray:
        warnings.warn(
            "The uncertainty is stripped when downcasting to ndarray.", UserWarning, stacklevel=2
        )
        return np.asarray(self._nom)

    def clip(self, min=None, max=None, out=None, **kwargs):

        return self.__class__(self._nom.clip(min, max, out, **kwargs), self._err)

    def fill(self, value) -> None:
        return self._nom.fill(value)

    def put(self, indices, values, mode="raise") -> None:
        if isinstance(values, self.__class__):
            self._nom.put(indices, values._nom, mode)
            self._err.put(indices, values._err, mode)
        else:
            raise ValueError("Can only 'put' Uncertainties into uncertainties!")

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

    def searchsorted(self, v, side="left", sorter=None):
        return self._nom.searchsorted(v, side)

    def __len__(self) -> int:
        return len(self._nom)

    def __getitem__(self, key):
        try:
            return self.__class__(self._nom[key], self._err[key])
        except TypeError:
            raise TypeError(f"Index {key} not supported!")

    def __setitem__(self, key, value):
        if not isinstance(value, self.__class__):
            raise ValueError(
                f"Can only pass Uncertainty type to __setitem__! Instead passed {type(value)}"
            )
        try:
            _ = self._nom[key]
        except ValueError as exc:
            raise ValueError(f"Object {type(self._nom)} does not support indexing") from exc

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
                    self.__class__(n, e).tolist() if isinstance(n, list) else self.__class__(n, e)
                    for n, e in (nom, err)
                ]
        except AttributeError:
            raise AttributeError(f"{type(self._nom).__name__}' does not support tolist.")
