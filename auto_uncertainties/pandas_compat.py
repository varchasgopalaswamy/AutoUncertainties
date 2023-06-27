# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import re
from typing import Any, Sequence, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
    register_dataframe_accessor,
    register_extension_dtype,
)
from pandas.api.types import (
    is_integer,
    is_list_like,
    is_object_dtype,
    is_string_dtype,
)
from pandas.compat import set_function_name
from pandas.core.arrays.base import ExtensionOpsMixin
from pandas.core.indexers import check_array_indexer

from . import nominal_values, std_devs, Uncertainty


class UncertaintyType(ExtensionDtype):
    """
    An Uncertainty duck-typed class, suitable for holding an uncertainty (i.e. value and error pair) dtype. Closely follows the implementation in pint-pandas.
    """

    type = Uncertainty
    _metadata = ("value_dtype",)
    _match = re.compile(r"[U|u]ncertainty\[([a-zA-Z0-9]+)\]")
    _cache = {}
    value_dtype: Any

    @property
    def _is_numeric(self):
        # type: () -> bool
        return True

    def __new__(cls, value_dtype):
        """
        Parameters
        ----------
        units : Pint units or string
        """

        if isinstance(value_dtype, UncertaintyType):
            return value_dtype

        elif value_dtype is None:
            # empty constructor for pickle compat
            return object.__new__(cls)

        else:
            u = object.__new__(cls)
            u.value_dtype = np.dtype(value_dtype)

        return u

    @classmethod
    def construct_from_string(cls, string):
        """
        Strict construction from a string, raise a TypeError if not
        possible
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        try:
            dtype = np.dtype(str)
            return cls(value_dtype=dtype)
        except Exception:
            ...

        if cls._match.match(string):
            dtype = cls._match.match(string).group(1)
            return cls(value_dtype=dtype)

        if isinstance(string, str) and (
            string.startswith("uncertainty[") or string.startswith("Pint[")
        ):
            # do not parse string like U as pint[U]
            # avoid tuple to be regarded as unit
            try:
                return cls(units=string)
            except ValueError:
                pass
        raise TypeError(
            f"Cannot construct a 'UncertaintyType' from '{string}'"
        )

    @property
    def name(self):
        return f"Uncertainty[{self.value_dtype}]"

    @property
    def na_value(self):
        return Uncertainty(np.nan, np.nan)

    def __hash__(self):
        # make myself hashable
        return hash(str(self))

    def __eq__(self, other):
        try:
            other = UncertaintyType(other)
        except ValueError:
            return False
        return self.value_dtype == other.value_dtype

    @classmethod
    def is_dtype(cls, dtype):
        """
        Return a boolean if we if the passed type is an actual dtype that we
        can match (via string or type)
        """
        if isinstance(dtype, str):
            return cls._match.match(dtype)
        return super(UncertaintyType, cls).is_dtype(dtype)

    @classmethod
    def construct_array_type(cls):
        return UncertaintyArray

    def __repr__(self):
        """
        Return a string representation for this object.

        Invoked by unicode(df) in py2 only. Yields a Unicode String in both
        py2/py3.
        """

        return self.name


dtypemap = {
    int: pd.Int64Dtype(),
    np.int64: pd.Int64Dtype(),
    np.int32: pd.Int32Dtype(),
    np.int16: pd.Int16Dtype(),
    np.int8: pd.Int8Dtype(),
    # np.float128: pd.Float128Dtype(),
    float: pd.Float64Dtype(),
    np.float64: pd.Float64Dtype(),
    np.float32: pd.Float32Dtype(),
    np.complex128: pd.core.dtypes.dtypes.PandasDtype("complex128"),
    np.complex64: pd.core.dtypes.dtypes.PandasDtype("complex64"),
    # np.float16: pd.Float16Dtype(),
}
dtypeunmap = {v: k for k, v in dtypemap.items()}


class UncertaintyArray(ExtensionArray, ExtensionOpsMixin):
    """Implements a class to describe an array of physical quantities:
    the product of an array of numerical values and a unit of measurement.

    Parameters
    ----------
    values : scalar or array-like
        Array of mean/nominal values
    errors: scalar or array-like
        Array of standard deviations/uncertainties
    dtype : UncertaintyType or str
        Datatype of the underlying values
    copy: bool
        Whether to copy the values.
    Returns
    -------

    """

    _data = np.array([])
    context_name = None

    def __init__(
        self,
        values,
        errors=None,
        dtype=None,
        copy=False,
    ):
        if dtype is None:
            if isinstance(values, np.ndarray):
                dtype = UncertaintyType(value_dtype=values.dtype)
        if isinstance(dtype, str):
            dtype = UncertaintyType.construct_from_string(dtype)
        if not isinstance(dtype, UncertaintyType):
            raise NotImplementedError

        self._dtype = dtype

        if isinstance(values, (UncertaintyArray)):
            values = values._nom
            errors = values._err
        elif isinstance(values, (Uncertainty)):
            values = values.value
            errors = values.error
        else:
            assert (
                errors is not None
            ), "errors must be specified for non UncertaintyArray values"

        if isinstance(values, np.ndarray):
            dtype = values.dtype
            if dtype in dtypemap:
                dtype = dtypemap[dtype]
            values = pd.array(values, copy=copy, dtype=dtype)
            errors = pd.array(errors, copy=copy, dtype=dtype)
            copy = False
        elif not isinstance(values, pd.core.arrays.numeric.NumericArray):
            values = pd.array(values, copy=copy)
            errors = pd.array(errors, copy=copy)
            copy = False
        if copy:
            values = values.copy()
            errors = errors.copy()

        self._nom = values
        self._err = errors

    def __getstate__(self):
        # we need to discard the cached _Q, which is not pickleable
        ret = dict(self.__dict__)
        return ret

    def __setstate__(self, dct):
        self.__dict__.update(dct)

    @property
    def dtype(self):
        # type: () -> ExtensionDtype
        """An instance of 'ExtensionDtype'."""
        return self._dtype

    def __len__(self):
        # type: () -> int
        """Length of this array

        Returns
        -------
        length : int
        """
        return len(self._nom)

    def __getitem__(self, item):
        # type (Any) -> Any
        """Select a subset of self.
        Parameters
        ----------
        item : int, slice, or ndarray
            * int: The position in 'self' to get.
            * slice: A slice object, where 'start', 'stop', and 'step' are
              integers or None
            * ndarray: A 1-d boolean NumPy ndarray the same length as 'self'
        Returns
        -------
        item : scalar Uncertainty or UncertaintyArray
        """

        if is_integer(item):
            return Uncertainty(self._nom[item], self._err[item])

        item = check_array_indexer(self, item)

        return self.__class__(
            self._nom[item],
            self._err[item],
        )

    def __setitem__(
        self,
        key,
        value: Union[Uncertainty, UncertaintyArray, Sequence[Uncertainty]],
    ):
        # need to not use `not value` on numpy arrays
        if isinstance(value, (list, tuple)) and (not value):
            # doing nothing here seems to be ok
            return

        if isinstance(value, self._dtype.type):
            val = value.value
            err = value.error

        elif (
            is_list_like(value)
            and len(value) > 0
            and isinstance(value[0], self._dtype.type)
        ):
            val = [item.value for item in value]
            err = [item.error for item in value]

        key = check_array_indexer(self, key)
        try:
            self._nom[key] = val
            self._err[key] = err
        except IndexError as e:
            msg = "Mask is wrong length. {}".format(e)
            raise IndexError(msg)

    def _formatter(self, boxed=False):
        """Formatting function for scalar values.
        This is used in the default '__repr__'. The returned formatting
        function receives scalar Uncertainties.

        # type: (bool) -> Callable[[Any], Optional[str]]

        Parameters
        ----------
        boxed: bool, default False
            An indicated for whether or not your array is being printed
            within a Series, DataFrame, or Index (True), or just by
            itself (False). This may be useful if you want scalar values
            to appear differently within a Series versus on its own (e.g.
            quoted or not).

        Returns
        -------
        Callable[[Any], str]
            A callable that gets instances of the scalar type and
            returns a string. By default, :func:`repr` is used
            when ``boxed=False`` and :func:`str` is used when
            ``boxed=True``.
        """

        def formatting_function(uncertainty: Uncertainty):
            return str(uncertainty)

        return formatting_function

    def isna(self):
        # type: () -> np.ndarray
        """Return a Boolean NumPy array indicating if each value is missing.

        Returns
        -------
        missing : np.array
        """
        return ~np.isfinite(self._nom)

    def astype(self, dtype, copy=True):
        """Cast to a NumPy array with 'dtype'.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        array : ndarray
            NumPy ndarray with 'dtype' for its dtype.
        """
        if isinstance(dtype, str) and self._dtype._match(dtype):
            dtype = self._dtype.construct_from_string(dtype)

        if isinstance(dtype, self._dtype):
            if dtype == self._dtype and not copy:
                return self
            else:
                return UncertaintyArray(
                    self.uncertainty.to(dtype.value_dtype), dtype
                )

        # do *not* delegate to __array__ -> is required to return a numpy array,
        # but somebody may be requesting another pandas array
        # examples are e.g. PyArrow arrays as requested by "string[pyarrow]"
        if is_object_dtype(dtype):
            return self._to_array_of_quantity(copy=copy)
        if is_string_dtype(dtype):
            return pd.array([str(x) for x in self.uncertainty], dtype=dtype)
        return pd.array(self.uncertainty, dtype, copy)

    @property
    def uncertainty(self):
        return Uncertainty(self._nom, self._err)

    def take(
        self,
        indices: Sequence[int],
        allow_fill: bool = False,
        fill_value: Uncertainty = None,
    ):
        """Take elements from an array.

        # type: (Sequence[int], bool, Optional[Any]) -> UncertaintyArray

        Parameters
        ----------
        indices : sequence of integers
            Indices to be taken.
        allow_fill : bool, default False
            How to handle negative values in `indices`.
            * False: negative values in `indices` indicate positional indices
              from the right (the default). This is similar to
              :func:`numpy.take`.
            * True: negative values in `indices` indicate
              missing values. These values are set to `fill_value`. Any other
              other negative values raise a ``ValueError``.
        fill_value : any, optional
            Fill value to use for NA-indices when `allow_fill` is True.
            This may be ``None``, in which case the default NA value for
            the type, ``self.dtype.na_value``, is used.

        Returns
        -------
        UncertaintyArray

        Raises
        ------
        IndexError
            When the indices are out of bounds for the array.
        ValueError
            When `indices` contains negative values other than ``-1``
            and `allow_fill` is True.
        Notes
        -----
        UncertaintyArray.take is called by ``Series.__getitem__``, ``.loc``,
        ``iloc``, when `indices` is a sequence of values. Additionally,
        it's called by :meth:`Series.reindex`, or any other method
        that causes realignemnt, with a `fill_value`.
        See Also
        --------
        numpy.take
        pandas.api.extensions.take
        Examples
        --------
        """
        from pandas.core.algorithms import take

        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value
        else:
            fill_value = Uncertainty(0, 0)

        value = take(
            self._nom,
            indices,
            fill_value=float(fill_value.value),
            allow_fill=allow_fill,
        )
        error = take(
            self._err,
            indices,
            fill_value=float(fill_value.error),
            allow_fill=allow_fill,
        )

        return UncertaintyArray(value, error, dtype=self.dtype)

    def copy(self, deep: bool = False):
        data = (self._nom, self._err)
        if deep:
            data = copy.deepcopy(data)
        else:
            data = copy.copy(data)

        return type(self)(*data, dtype=self.dtype)

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[UncertaintyArray]):
        v = []
        e = []
        dtype = None
        for a in to_concat:
            if dtype is None:
                dtype = a.dtype
            else:
                a = a.astype(dtype)
            v.append(a._nom)
            e.append(a._err)

        return cls(np.concatenate(v), np.concatenate(e), dtype)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """
        Initialises a UncertaintyArray from a list like of Uncertainty scalars or a list like of value/error and dtype
        -----
        Usage
        UncertaintyArray._from_sequence([Uncertainty(1,0),Uncertainty(1,1)])
        """

        list_of_scalars = []
        for s in scalars:
            if isinstance(s, Uncertainty):
                list_of_scalars.append(s)
            elif isinstance(s, tuple):
                list_of_scalars.append(Uncertainty(*s))
            else:
                list_of_scalars.append(Uncertainty(s, 0))

        values = np.asarray([x.value for x in list_of_scalars])
        errors = np.asarray([x.error for x in list_of_scalars])

        return cls(values, errors, dtype=dtype, copy=copy)

    @classmethod
    def _from_sequence_of_strings(cls, scalars, dtype=None, copy=False):
        list_of_uncs = [Uncertainty.from_string(x) for x in scalars]
        return cls._from_sequence(list_of_uncs, dtype=dtype, copy=copy)

    def value_counts(self, dropna=True):
        """
        Returns a Series containing counts of each category.

        Every category will have an entry, even those with a count of 0.

        Parameters
        ----------
        dropna : boolean, default True
            Don't include counts of NaN.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.value_counts
        """

        from pandas import Series

        # compute counts on the data with no nans
        data = Uncertainty(self._nom, self._err)

        nafilt = np.isnan(data)
        data = data[~nafilt]

        data_list = [Uncertainty(x.value, x.error) for x in data]
        index = list(set(data))
        array = [data_list.count(item) for item in index]

        if not dropna:
            index.append(np.nan)
            array.append(nafilt.sum())

        return Series(array, index=index)

    def unique(self):
        """Compute the UncertaintyArray of unique values.

        Returns
        -------
        uniques : UncertaintyArray
        """
        from pandas import unique

        data = Uncertainty(self._nom, self._err)

        return self._from_sequence(unique(data), dtype=self.dtype)

    def __contains__(self, item) -> bool:
        if not isinstance(item, Uncertainty):
            return False
        elif pd.isna(item):
            return self.isna().any()
        else:
            return super().__contains__(item)

    @property
    def nbytes(self):
        return self._nom.nbytes * 2

    # The _can_hold_na attribute is set to True so that pandas internals
    # will use the ExtensionDtype.na_value as the NA value in operations
    # such as take(), reindex(), shift(), etc.  In addition, those results
    # will then be of the ExtensionArray subclass rather than an array
    # of objects
    _can_hold_na = True

    @property
    def _ndarray_values(self):
        # type: () -> np.ndarray
        """Internal pandas method for lossy conversion to a NumPy ndarray.
        This method is not part of the pandas interface.
        The expectation is that this is cheap to compute, and is primarily
        used for interacting with our indexers.
        """
        return np.array(self)

    @classmethod
    def _create_method(cls, op, coerce_to_dtype=True):
        """
        A class method that returns a method that will correspond to an
        operator for an ExtensionArray subclass, by dispatching to the
        relevant operator defined on the individual elements of the
        ExtensionArray.
        Parameters
        ----------
        op : function
            An operator that takes arguments op(a, b)
        coerce_to_dtype :  bool
            boolean indicating whether to attempt to convert
            the result to the underlying ExtensionArray dtype
            (default True)
        Returns
        -------
        A method that can be bound to a method of a class
        Example
        -------
        Given an ExtensionArray subclass called MyExtensionArray, use
        >>> __add__ = cls._create_method(operator.add)
        in the class definition of MyExtensionArray to create the operator
        for addition, that will be based on the operator implementation
        of the underlying elements of the ExtensionArray
        """

        def _binop(self: UncertaintyArray, other):
            def validate_length(obj1, obj2):
                # validates length
                try:
                    if len(obj1) != len(obj2):
                        raise ValueError("Lengths must match")
                except TypeError:
                    pass

            def convert_values(param):
                # convert to a quantity or listlike
                if isinstance(param, cls):
                    return param.uncertainty
                elif isinstance(param, Uncertainty):
                    return param
                elif (
                    is_list_like(param)
                    and len(param) > 0
                    and isinstance(param[0], Uncertainty)
                ):
                    return param[0]
                else:
                    return param

            if isinstance(other, (Series, DataFrame)):
                return NotImplemented
            lvalues = self.uncertainty
            validate_length(lvalues, other)
            rvalues = convert_values(other)

            # If the operator is not defined for the underlying objects,
            # a TypeError should be raised
            res = op(lvalues, rvalues)

            if coerce_to_dtype:
                try:
                    val = nominal_values(res)
                    err = std_devs(res)
                    res = cls(val, err, dtype=self.dtype)
                except TypeError:
                    pass

            return res

        op_name = f"__{op}__"
        return set_function_name(_binop, op_name, cls)

    @classmethod
    def _create_arithmetic_method(cls, op):
        return cls._create_method(op)

    @classmethod
    def _create_comparison_method(cls, op):
        return cls._create_method(op, coerce_to_dtype=False)

    def __array__(self, dtype=None, copy=False):
        if dtype is None or is_object_dtype(dtype):
            return self.uncertainty
        if is_string_dtype(dtype):
            return np.array([str(x) for x in self.uncertainty], dtype=str)
        return Uncertainty(self._nom.astype(dtype), self._err.astype(dtype))

    def searchsorted(self, value, side="left", sorter=None):
        """
        Find indices where elements should be inserted to maintain order.

        .. versionadded:: 0.24.0

        Find the indices into a sorted array `self` (a) such that, if the
        corresponding elements in `v` were inserted before the indices, the
        order of `self` would be preserved.

        Assuming that `a` is sorted:

        ======  ============================
        `side`  returned index `i` satisfies
        ======  ============================
        left    ``self[i-1] < v <= self[i]``
        right   ``self[i-1] <= v < self[i]``
        ======  ============================

        Parameters
        ----------
        value : array_like
            Values to insert into `self`.
        side : {'left', 'right'}, optional
            If 'left', the index of the first suitable location found is given.
            If 'right', return the last such index.  If there is no suitable
            index, return either 0 or N (where N is the length of `self`).
        sorter : 1-D array_like, optional
            Optional array of integer indices that sort array a into ascending
            order. They are typically the result of argsort.

        Returns
        -------
        indices : array of ints
            Array of insertion points with the same shape as `value`.

        See Also
        --------
        numpy.searchsorted : Similar method from NumPy.
        """
        # Note: the base tests provided by pandas only test the basics.
        # We do not test
        # 1. Values outside the range of the `data_for_sorting` fixture
        # 2. Values between the values in the `data_for_sorting` fixture
        # 3. Missing values.

        arr = self._nom
        if isinstance(value, Uncertainty):
            val = value.value
        elif (
            is_list_like(value)
            and len(value) > 0
            and isinstance(value[0], Uncertainty)
        ):
            val = [item.value for item in value]
        return arr.searchsorted(val, side=side, sorter=sorter)

    def _reduce(self, name, **kwds):
        """
        Return a scalar result of performing the reduction operation.

        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            { any, all, min, max, sum, mean, median, prod,
            std, var, sem, kurt, skew }.
        skipna : bool, default True
            If True, skip NaN values.
        **kwargs
            Additional keyword arguments passed to the reduction function.
            Currently, `ddof` is the only supported kwarg.

        Returns
        -------
        scalar

        Raises
        ------
        TypeError : subclass does not define reductions
        """

        functions = {
            "min": np.nanmin,
            "max": np.nanmax,
            "sum": np.nansum,
            "mean": np.nanmean,
            "median": np.nanmedian,
            "std": np.nanstd,
            "var": np.nanvar,
        }
        if name not in functions:
            raise TypeError(f"cannot perform {name} with type {self.dtype}")

        result = functions[name](self.uncertainty, **kwds)
        return result


UncertaintyArray._add_arithmetic_ops()
UncertaintyArray._add_comparison_ops()
register_extension_dtype(UncertaintyType)


@register_dataframe_accessor("uncertainty")
class UncertaintyDataFrameAccessor(object):
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj


class UncertaintySeriesAccessor(object):
    def __init__(self, pandas_obj: pd.Series):
        self._validate(pandas_obj)
        self.pandas_obj = pandas_obj
        self.uncertainty = pandas_obj.values
        self._index = pandas_obj.index
        self._name = pandas_obj.name

    @staticmethod
    def _validate(obj):
        if not is_uncert_type(obj):
            raise AttributeError(
                "Cannot use 'uncertainty' accessor on objects of "
                "dtype '{}'.".format(obj.dtype)
            )


class Delegated:
    # Descriptor for delegating attribute access to from
    # a Series to an underlying array
    to_series = True

    def __init__(self, name):
        self.name = name


class DelegatedProperty(Delegated):
    def __get__(self, obj, type=None):
        index = object.__getattribute__(obj, "_index")
        name = object.__getattribute__(obj, "_name")
        result = getattr(
            object.__getattribute__(obj, "uncertainty"), self.name
        )
        if self.to_series:
            if isinstance(result, Uncertainty):
                result = UncertaintyArray(result)
            return Series(result, index, name=name)
        else:
            return result


class DelegatedScalarProperty(DelegatedProperty):
    to_series = False


class DelegatedMethod(Delegated):
    def __get__(self, obj, type=None):
        index = object.__getattribute__(obj, "_index")
        name = object.__getattribute__(obj, "_name")
        method = getattr(
            object.__getattribute__(obj, "uncertainty"), self.name
        )

        def delegated_method(*args, **kwargs):
            result = method(*args, **kwargs)
            if self.to_series:
                if isinstance(result, Uncertainty):
                    result = UncertaintyArray(result)
                result = Series(result, index, name=name)
            return result

        return delegated_method


class DelegatedScalarMethod(DelegatedMethod):
    to_series = False


for attr in [
    "debug_used",
    "default_format",
    "dimensionality",
    "dimensionless",
    "force_ndarray",
    "shape",
    "u",
    "unitless",
    "units",
]:
    setattr(UncertaintySeriesAccessor, attr, DelegatedScalarProperty(attr))
for attr in ["imag", "m", "magnitude", "real"]:
    setattr(UncertaintySeriesAccessor, attr, DelegatedProperty(attr))

for attr in [
    "check",
    "compatible_units",
    "format_babel",
    "ito",
    "ito_base_units",
    "ito_reduced_units",
    "ito_root_units",
    "plus_minus",
    "put",
    "to_tuple",
    "tolist",
]:
    setattr(UncertaintySeriesAccessor, attr, DelegatedScalarMethod(attr))
for attr in [
    "clip",
    "from_tuple",
    "m_as",
    "searchsorted",
    "to",
    "to_base_units",
    "to_compact",
    "to_reduced_units",
    "to_root_units",
    "to_timedelta",
]:
    setattr(UncertaintySeriesAccessor, attr, DelegatedMethod(attr))


def is_uncert_type(obj):
    t = getattr(obj, "dtype", obj)
    try:
        return isinstance(t, UncertaintyType) or issubclass(t, UncertaintyType)
    except Exception:
        return False


# try:
#     # for pint < 0.21 we need to explicitly register
#     compat.upcast_types.append(UncertaintyArray)
# except AttributeError:
#     # for pint = 0.21 we need to add the full names of UncertaintyArray and DataFrame,
#     # which is to be added in pint > 0.21
#     compat.upcast_type_map.setdefault("pint_pandas.pint_array.UncertaintyArray", UncertaintyArray)
#     compat.upcast_type_map.setdefault("pandas.core.frame.DataFrame", DataFrame)
