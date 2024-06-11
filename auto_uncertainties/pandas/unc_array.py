"""An implementation of Decimal as a DType.

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.extensions.ExtensionDtype.html#pandas.api.extensions.ExtensionDtype
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.extensions.ExtensionArray.html#pandas.api.extensions.ExtensionArray

https://github.com/pandas-dev/pandas/tree/e246c3b05924ac1fe083565a765ce847fcad3d91/pandas/tests/extension/decimal

"""

from __future__ import annotations

from copy import deepcopy
import sys
from typing import TYPE_CHECKING, Self, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_list_like
from pandas.compat import set_function_name
from pandas.core.arrays import ExtensionArray, ExtensionScalarOpsMixin
from pandas.core.dtypes.common import is_integer
from pandas.core.indexers import check_array_indexer

from auto_uncertainties.uncertainty import (
    ScalarUncertainty,
    Uncertainty,
    VectorUncertainty,
)

from .unc_dtype import UncertaintyDtype

if TYPE_CHECKING:
    pass

__all__ = ["UncertaintyArray"]


class UncertaintyArray(ExtensionArray, ExtensionScalarOpsMixin):
    """Abstract base class for custom 1-D array types."""

    __array_priority__ = VectorUncertainty.__array_priority__
    __pandas_priority__ = 1999

    ####################################################
    #### Construction ##################################
    ####################################################
    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """Construct a new ExtensionArray from a sequence of scalars."""
        return cls(scalars, dtype=dtype)

    @classmethod
    def _from_sequence_of_strings(
        cls,
        strings,
        *,
        dtype: UncertaintyDtype | None = None,
        copy: bool = False,
    ):
        vals = []
        for s in strings:
            if not isinstance(s, str):
                msg = "not all strings are of dtype str"
                raise TypeError(msg)
            vals.append(Uncertainty.from_string(s))

        return cls(vals, dtype=dtype, copy=copy)

    @classmethod
    def _from_factorized(cls, values, original):
        """Reconstruct an ExtensionArray after factorization."""
        return cls(values)

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
                # The only valid kinds of objects are
                # 1. All a seq of UArrays
                # Or any combination of
                # 2. Uncertainties
                # 3. tuples of value/error pairs
                # 4. floats (in which case error will be zero)

                if len(values) > 0:
                    # If its a sequence of Uarrays
                    if all(isinstance(x, UncertaintyArray) for x in values):
                        errors = np.concatenate([x._data._err for x in values])
                        values = np.concatenate([x._data._nom for x in values])
                    else:
                        vals = []
                        errs = []
                        for x in values:
                            if isinstance(x, VectorUncertainty):
                                errs += x._err.tolist()
                                vals += x._nom.tolist()
                            elif isinstance(x, ScalarUncertainty):
                                errs.append(x._err)
                                vals.append(x._nom)
                            elif hasattr(x, "__len__") and len(x) == 2:
                                errs.append(x[1])
                                vals.append(x[0])
                            elif isinstance(x, float):
                                errs.append(0.0)
                                vals.append(x)
                            else:
                                msg = f"values must be only UncertaintyArray, Uncertainty, (float,float), float or sequences of these. Instead got {type(x)}"
                                raise ValueError(msg)
                        values = vals
                        errors = errs
                    # If its a sequence of Uncertainties
                else:
                    errors = np.array([])
                    values = np.array([])
            else:
                msg = f"values must be only UncertaintyArray, Uncertainty or a list of them. Instead got {type(values)}"
                raise ValueError(msg)
        if copy:
            values = deepcopy(values)
            errors = deepcopy(errors)
        values = np.atleast_1d(values)
        errors = np.atleast_1d(errors)

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
    def value(self):
        return self._data._nom

    @property
    def error(self):
        return self._data._err

    @property
    def nbytes(self):
        """The byte size of the data."""
        return sys.getsizeof(self._data._nom[0]) * len(self) * 2

    @property
    def dtype(self):
        """An instance of 'ExtensionDtype'."""
        return self._dtype

    @property
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

        return self.__class__(self._data, dtype=self.dtype, copy=True)

    ##########################
    ###### NaN handling ######
    ##########################
    def __contains__(self, item: Uncertainty | float) -> bool | np.bool_:
        if isinstance(item, float) and pd.isna(item):
            return cast(np.ndarray, self.isna()).any()
        elif not isinstance(item, Uncertainty):
            return False
        else:
            return super().__contains__(item)

    ##########################
    ######## Numpy ###########
    ##########################

    _HANDLED_TYPES = (np.ndarray, Uncertainty, float, int)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        #
        if not all(
            isinstance(t, (*self._HANDLED_TYPES, UncertaintyArray)) for t in inputs
        ):
            return NotImplemented
        # Extract the underlying Uncertainty from all UArray objects
        inputs = tuple(
            x._data if isinstance(x, UncertaintyArray) else x for x in inputs
        )
        # Perform the operation
        result = getattr(ufunc, method)(*inputs, **kwargs)
        # Deal with boolean ops, otherwise return a new UArray
        if all(isinstance(x, bool | np.bool_) for x in result):
            retval = result[0] if len(result) == 1 else np.asarray(result, dtype=bool)
        elif ufunc.nout > 1:
            retval = tuple(self.__class__(x) for x in result)
        else:
            retval = self.__class__(result)

        return retval

    def __pos__(self):
        return self.__class__(+self._data)

    def __neg__(self):
        return self.__class__(-self._data)

    def __abs__(self):
        return self.__class__(abs(self._data))

    def __invert__(self):
        raise TypeError

    ##############################
    #### List-like ###############
    ##############################

    def __getitem__(self, item):
        """Select a subset of self."""
        if is_integer(item):
            return self._data[item]

        key = check_array_indexer(self, item)
        return UncertaintyArray(self._data[key])

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
        elif not np.any(np.isfinite(value)):
            v = self.dtype.na_value
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

    @classmethod
    def _concat_same_type(cls, to_concat):
        """Concatenate multiple arrays."""
        return cls(
            np.concatenate([x._data for x in to_concat]),
        )

    def take(
        self,
        indexer,
        allow_fill=False,
        fill_value: (float | tuple[float, float] | ScalarUncertainty | None) = None,
    ):
        """Take elements from an array.

        Relies on the take method defined in pandas:
        https://github.com/pandas-dev/pandas/blob/e246c3b05924ac1fe083565a765ce847fcad3d91/pandas/core/algorithms.py#L1483
        """
        from pandas.api.extensions import take

        if allow_fill:
            if fill_value is None:
                fill_value = self.dtype.na_value
                fval = fill_value
                ferr = 0

            elif isinstance(fill_value, tuple):
                fval = fill_value[0]
                ferr = fill_value[1]
            elif isinstance(fill_value, ScalarUncertainty):
                fval = fill_value.value
                ferr = fill_value.error
            else:
                fval = fill_value
                ferr = 0
        else:
            fval = ferr = None
        v = take(
            self._data._nom,
            indexer,
            fill_value=fval,
            allow_fill=allow_fill,
        )
        e = take(
            self._data._err,
            indexer,
            fill_value=ferr,
            allow_fill=allow_fill,
        )
        return self._from_sequence(list(zip(v, e, strict=False)))

    def __eq__(self, other: pd.DataFrame | pd.Series | pd.Index | UncertaintyArray):
        """
        Return for `self == other` (element-wise equality).
        """
        # Implementer note: this should return a boolean numpy ndarray or
        # a boolean ExtensionArray.
        # When `other` is one of Series, Index, or DataFrame, this method should
        # return NotImplemented (to ensure that those objects are responsible for
        # first unpacking the arrays, and then dispatch the operation to the
        # underlying arrays)
        if isinstance(other, pd.DataFrame | pd.Series | pd.Index):
            return NotImplemented

        return self._data == other._data

    def isna(self):
        """A 1-D array indicating if each value is missing."""
        return np.isnan(self._data._nom)

    def _formatter(self, boxed=False):
        def formatter(x):
            return f"{x}"

        return formatter

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

    _supported_reductions = (
        "min",
        "max",
        "sum",
        "mean",
        "median",
        "prod",
        "std",
        "var",
        "sem",
    )

    def _reduce(
        self,
        name: str,
        *,
        skipna: bool = True,
        keepdims: bool = False,
        **kwargs,
    ):
        functions = {
            "min": np.min,
            "max": np.max,
            "sum": np.sum,
            "mean": np.mean,
            "median": np.median,
            "prod": np.prod,
            "std": lambda x: np.std(x, ddof=1),
            "var": lambda x: np.var(x, ddof=1),
            "sem": lambda x: np.std(x, ddof=0),
        }
        if name not in functions:
            msg = f"cannot perform {name} with type {self.dtype}"
            raise TypeError(msg)
        quantity = self.dropna().array if skipna else self.array
        result = cast(Uncertainty, functions[name](quantity))

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
        res = [op(a, b) for (a, b) in zip(lvalues, rvalues, strict=False)]

        return np.asarray(res, dtype=bool)

    def value_counts(self, dropna: bool = True):
        from pandas.core.algorithms import (
            value_counts_internal as value_counts,
        )

        return value_counts(self._data._nom, dropna=dropna)

    @classmethod
    def _create_method(cls, op, coerce_to_dtype: bool = True, result_dtype=None):
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

        def _binop(self: Self, other):
            if isinstance(other, pd.DataFrame | pd.Series | pd.Index):
                # rely on pandas to unbox and dispatch to us
                return NotImplemented

            def convert_values(param):
                if isinstance(param, UncertaintyArray):
                    return param
                elif is_list_like(param):
                    ovalues = UncertaintyArray._from_sequence(param)
                else:
                    ovalues = param
                return ovalues

            lvalues = self._data
            rvalues = convert_values(other)

            real_op = op
            # If the operator is not defined for the underlying objects,
            # a TypeError should be raised
            if op.__name__ in ["divmod", "rdivmod", "__invert__"]:
                raise TypeError

            if isinstance(rvalues, UncertaintyArray):
                res = real_op(lvalues, rvalues._data)
            else:
                res = real_op(lvalues, rvalues)

            if all(isinstance(x, bool | np.bool_) for x in res):
                return res

            return UncertaintyArray._from_sequence(res)

        op_name = f"__{op.__name__}__"
        return set_function_name(_binop, op_name, cls)


UncertaintyArray._add_arithmetic_ops()
UncertaintyArray._add_comparison_ops()
