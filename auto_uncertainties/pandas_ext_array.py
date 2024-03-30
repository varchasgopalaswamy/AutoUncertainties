# -*- coding: utf-8 -*-
"""An implementation of Decimal as a DType.

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.extensions.ExtensionDtype.html#pandas.api.extensions.ExtensionDtype
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.extensions.ExtensionArray.html#pandas.api.extensions.ExtensionArray

https://github.com/pandas-dev/pandas/tree/e246c3b05924ac1fe083565a765ce847fcad3d91/pandas/tests/extension/decimal

"""

from __future__ import annotations

import sys
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.stats
from pandas.api.extensions import register_extension_dtype
from pandas.api.types import is_list_like
from pandas.compat import set_function_name
from pandas.core.arrays import ExtensionArray, ExtensionScalarOpsMixin
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import is_integer
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndex, ABCSeries
from pandas.core.indexers import check_array_indexer

from .uncertainty import ScalarUncertainty, Uncertainty, VectorUncertainty

if TYPE_CHECKING:
    from pandas._typing import type_t


@register_extension_dtype
class UncertaintyDtype(ExtensionDtype):
    type = Uncertainty
    name = "Uncertainty"

    def __init__(self, dtype):
        self.value_dtype = dtype

    @property
    def na_value(self):
        return ScalarUncertainty(np.nan, 0)

    def __repr__(self) -> str:
        return f"Uncertainty[{self.value_dtype}]"

    @classmethod
    def construct_array_type(cls) -> type_t[Uncertainty]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return VectorUncertainty

    @property
    def _is_numeric(self) -> bool:
        return True


class UncertaintyArray(ExtensionArray, ExtensionScalarOpsMixin):
    """Abstract base class for custom 1-D array types."""

    __array_priority__ = VectorUncertainty.__array_priority__

    def __init__(
        self,
        values,
        errors=None,
        dtype=None,
        copy=False,
    ):
        if errors is not None:
            assert len(values) == len(
                errors
            ), "values and errors must have the same length"
        else:
            # Passed a UncertaintyArray
            if isinstance(values, (UncertaintyArray)):
                errors = values._data._err
                values = values._data._nom
            # Passed an Uncertainty
            elif isinstance(values, (Uncertainty)):
                errors = values._err
                values = values._nom
            # Passed some kind of list-like
            elif is_list_like(values):
                # If its got anything in it
                if len(values) > 0:
                    # If its a sequence of Uarrays
                    if all(isinstance(x, UncertaintyArray) for x in values):
                        errors = np.concatenate([x._data._err for x in values])
                        values = np.concatenate([x._data._nom for x in values])
                    # If its a sequence of Uncertainties
                    elif all(isinstance(x, Uncertainty) for x in values):
                        errors = [x._err for x in values]
                        values = [x._nom for x in values]
                    # If its a sequence of tuples of value/error paris
                    elif all(len(x) == 2 for x in values):
                        errors = [x[1] for x in values]
                        values = [x[0] for x in values]
                    else:
                        unique_types = set(type(x) for x in values)
                        raise ValueError(
                            f"values must be only UncertaintyArray or Uncertainty. Instead got {unique_types}"
                        )
                else:
                    errors = np.array([])
                    values = np.array([])
            else:
                raise ValueError(
                    f"values must be only UncertaintyArray, Uncertainty or a list of them. Instead got {type(values)}"
                )
        if copy:
            values = deepcopy(values)
            errors = deepcopy(errors)
        values = np.atleast_1d((values))
        errors = np.atleast_1d((errors))

        if dtype is None:
            dtype = UncertaintyDtype(values.dtype)

        self._dtype = dtype

        self._data = VectorUncertainty(values, errors)
        self._items = self.data = self._data

        assert self._data.ndim == 1, "Data must be 1-dimensional"

    #############################################
    ############# Attributes ####################
    #############################################
    @property
    def nbytes(self):
        """The byte size of the data."""
        return sys.getsizeof(self._data._nom[0]) * len(self) * 2

    @property
    def dtype(self):
        """An instance of 'ExtensionDtype'."""
        return self._dtype

    def array(self):
        return self._data

    def copy(self):
        """
        Return a copy of the array.

        Returns
        -------
        ExtensionArray

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr2 = arr.copy()
        >>> arr[0] = 2
        >>> arr2
        <IntegerArray>
        [1, 2, 3]
        Length: 3, dtype: Int64
        """

        return self.__class__(self._data, copy=True)

    _HANDLED_TYPES = (np.ndarray, Uncertainty)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        #
        if not all(
            isinstance(t, self._HANDLED_TYPES + (UncertaintyArray,))
            for t in inputs
        ):
            return NotImplemented
        if method != "__call__":
            raise NotImplementedError
        inputs = tuple(
            x._data if isinstance(x, UncertaintyArray) else x for x in inputs
        )
        raise ValueError
        return getattr(ufunc, method)(*inputs, **kwargs)

        # def reconstruct(x):
        #     if isinstance(x, (decimal.Decimal, numbers.Number)):
        #         return x
        #     else:
        #         return type(self)._from_sequence(x, dtype=self.dtype)

        # if ufunc.nout > 1:
        #     return tuple(reconstruct(x) for x in result)
        # else:
        #     return reconstruct(result)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """Construct a new ExtensionArray from a sequence of scalars."""
        return cls(scalars, dtype=dtype)

    @classmethod
    def _from_factorized(cls, values, original):
        """Reconstruct an ExtensionArray after factorization."""
        return cls(values)

    def __getitem__(self, item):
        """Select a subset of self."""
        if is_integer(item):
            return self._data[item]
        return UncertaintyArray(self._data[item])

    def __setitem__(self, key, value):
        """Set the value of a subset of self."""
        if isinstance(value, UncertaintyArray):
            v = value._data
        elif is_list_like(value) and len(value) > 0:
            if all(isinstance(x, UncertaintyArray) for x in value):
                v = UncertaintyArray._from_sequence(value)._data
            else:
                v = Uncertainty.from_sequence(value)
            if len(v) == 1:
                v = v[0]
        elif (is_list_like(value) and len(value) == 0) or (not value):
            return
        elif isinstance(value, Uncertainty):
            v = value
        else:
            raise ValueError

        key = check_array_indexer(self, key)
        self._data[key] = v

    def __len__(self) -> int:
        """Length of this array."""
        if np.ndim(self._data) == 0:
            return 0
        else:
            return len(self._data)

    def __eq__(
        self, other: pd.DataFrame | pd.Series | pd.Index | UncertaintyArray
    ):
        """
        Return for `self == other` (element-wise equality).
        """
        # Implementer note: this should return a boolean numpy ndarray or
        # a boolean ExtensionArray.
        # When `other` is one of Series, Index, or DataFrame, this method should
        # return NotImplemented (to ensure that those objects are responsible for
        # first unpacking the arrays, and then dispatch the operation to the
        # underlying arrays)
        if (
            isinstance(other, pd.DataFrame)
            or isinstance(other, pd.Series)
            or isinstance(other, pd.Index)
        ):
            return NotImplemented
        # rely on Quantity comparison that will return a boolean array
        return self._data == other._data

    def isna(self):
        """A 1-D array indicating if each value is missing."""
        return np.isnan(self._data._nom)

    def take(self, indexer, allow_fill=False, fill_value=None):
        """Take elements from an array.

        Relies on the take method defined in pandas:
        https://github.com/pandas-dev/pandas/blob/e246c3b05924ac1fe083565a765ce847fcad3d91/pandas/core/algorithms.py#L1483
        """
        from pandas.api.extensions import take

        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value

        v = take(
            self._data._nom,
            indexer,
            fill_value=fill_value,
            allow_fill=allow_fill,
        )
        e = take(
            self._data._err,
            indexer,
            fill_value=fill_value,
            allow_fill=allow_fill,
        )
        return self._from_sequence(list(zip(v, e)))

    def _formatter(self, boxed=False):
        def formatter(x):
            return f"{x}"

        return formatter

    @classmethod
    def _concat_same_type(cls, to_concat):
        """Concatenate multiple arrays."""
        return cls(
            np.concatenate([x._data for x in to_concat]),
        )

    @property
    def _na_value(self):
        return self.dtype.na_value

    def dropna(self):
        return self[~self.isna()]

    def unique(self):
        return self.__class__(np.unique(self._data))

    def searchsorted(self, value, side="left", sorter=None):
        return np.searchsorted(self._data, value, side=side, sorter=sorter)

    def _values_for_argsort(self):
        """
        Return values for sorting.
        Returns
        -------
        ndarray
            The transformed values should maintain the ordering between values
            within the array.
        See Also
        --------
        ExtensionArray.argsort : Return the indices that would sort this array.
        """
        # Note: this is used in `ExtensionArray.argsort`.
        return self._data._nom

    def _reduce(
        self,
        name: str,
        *,
        skipna: bool = True,
        keepdims: bool = False,
        **kwargs,
    ):
        functions = {
            "all": np.all,
            "any": np.any,
            "min": np.min,
            "max": np.max,
            "sum": np.sum,
            "mean": np.mean,
            "median": np.median,
            "prod": np.prod,
            "std": lambda x: np.std(x, ddof=1),
            "var": lambda x: np.var(x, ddof=1),
            "sem": lambda x: np.std(x, ddof=0),
            "kurt": lambda x: scipy.stats.kurtosis(x, bias=False),
            "skew": lambda x: scipy.stats.skew(x, bias=False),
        }
        if name not in functions:
            raise TypeError(f"cannot perform {name} with type {self.dtype}")

        if skipna:
            quantity = self.dropna()._data
        else:
            quantity = self._data

        result = functions[name](quantity)

        if keepdims:
            return self.__class__(result)
        else:
            return result

    def _cmp_method(self, other, op):
        # For use with OpsMixin
        def convert_values(param):
            if isinstance(param, ExtensionArray):
                ovalues = param
            else:
                # Assume it's an object
                ovalues = [param] * len(self)
            return ovalues

        lvalues = self
        rvalues = convert_values(other)

        # If the operator is not defined for the underlying objects,
        # a TypeError should be raised
        res = [op(a, b) for (a, b) in zip(lvalues, rvalues)]

        return np.asarray(res, dtype=bool)

    def value_counts(self, dropna: bool = True):
        from pandas.core.algorithms import (
            value_counts_internal as value_counts,
        )

        return value_counts(self._data._nom, dropna=dropna)

    # We override fillna here to simulate a 3rd party EA that has done so. This
    #  lets us test a 3rd-party EA that has not yet updated to include a "copy"
    #  keyword in its fillna method.
    def fillna(self, value=None, limit=None):
        return super().fillna(value=value, limit=limit)

    @classmethod
    def _create_method(
        cls, op, coerce_to_dtype: bool = True, result_dtype=None
    ):
        """
        A class method that returns a method that will correspond to an
        operator for an ExtensionArray subclass, by dispatching to the
        relevant operator defined on the individual elements of the
        ExtensionArray.

        Parameters
        ----------
        op : function
            An operator that takes arguments op(a, b)
        coerce_to_dtype : bool, default True
            boolean indicating whether to attempt to convert
            the result to the underlying ExtensionArray dtype.
            If it's not possible to create a new ExtensionArray with the
            values, an ndarray is returned instead.

        Returns
        -------
        Callable[[Any, Any], Union[ndarray, ExtensionArray]]
            A method that can be bound to a class. When used, the method
            receives the two arguments, one of which is the instance of
            this class, and should return an ExtensionArray or an ndarray.

            Returning an ndarray may be necessary when the result of the
            `op` cannot be stored in the ExtensionArray. The dtype of the
            ndarray uses NumPy's normal inference rules.

        Examples
        --------
        Given an ExtensionArray subclass called MyExtensionArray, use

            __add__ = cls._create_method(operator.add)

        in the class definition of MyExtensionArray to create the operator
        for addition, that will be based on the operator implementation
        of the underlying elements of the ExtensionArray
        """

        def _binop(self, other):
            def convert_values(param):
                if isinstance(param, ExtensionArray) or is_list_like(param):
                    ovalues = param
                else:  # Assume its an object
                    ovalues = [param] * len(self)
                return ovalues

            if isinstance(other, (ABCSeries, ABCIndex, ABCDataFrame)):
                # rely on pandas to unbox and dispatch to us
                return NotImplemented

            lvalues = self
            rvalues = convert_values(other)

            # If the operator is not defined for the underlying objects,
            # a TypeError should be raised
            if isinstance(rvalues, UncertaintyArray):
                res = op(lvalues._data, rvalues._data)
            else:
                res = op(lvalues._data, rvalues)

            return res

        op_name = f"__{op.__name__}__"
        return set_function_name(_binop, op_name, cls)


UncertaintyArray._add_arithmetic_ops()
UncertaintyArray._add_comparison_ops()
# UncertaintyArray._add_logical_ops()
