# -*- coding: utf-8 -*-
from collections.abc import Iterable
import pandas as pd
import numpy as np
from pandas.api.extensions import ExtensionArray, ExtensionDtype, register_extension_dtype
from pandas.core.arrays.base import ExtensionOpsMixin
from pandas.compat import set_function_name
import loguru
import numbers

from .uncertainty import Uncertainty
from .util import is_np_duck_array, ignore_numpy_downcast_warnings


@register_extension_dtype
class UncertaintyType(ExtensionDtype):
    type = Uncertainty
    name = "uncertainty"

    def __init__(self):
        pass

    @property
    def _is_numeric(self):
        return True

    @property
    def _is_boolean(self):
        return False

    @classmethod
    def construct_array_type(cls):
        return UncertaintyArray

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(str(self))

    @property
    def na_value(self):
        return Uncertainty(np.nan, np.nan)

    @classmethod
    def construct_from_string(cls, string):
        if not isinstance(string, str):
            raise TypeError("'construct_from_string' expects a string, got {}".format(type(string)))
        elif string == cls.name:
            return cls()
        else:
            raise TypeError("Cannot construct a '{}' from '{}'".format(cls.__name__, string))


class UncertaintyArray(ExtensionArray, ExtensionOpsMixin):
    # Required attributes
    @property
    def dtype(self):
        return UncertaintyType()

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def shape(self):
        return self.data.shape

    @property
    def nbytes(self):
        return self.data.nbytes

    # Required methods
    def __len__(self):
        return len(self.data)

    def __setitem__(self, key, value):
        self.data[key] = value

    def __eq__(self, other):
        return self.data == other

    def isna(self):
        return np.isnan(self.data)

    def copy(self):
        inst = self.__class__(np.copy(self.data))
        return inst

    @ignore_numpy_downcast_warnings
    def take(self, indices, allow_fill=False, fill_value=None):
        from pandas.api.extensions import take

        if allow_fill:
            if fill_value is None or pd.isna(fill_value):
                fill_value = None
            elif not (isinstance(fill_value, Uncertainty) or fill_value is None):
                raise TypeError("Fill value must be Uncertainty or None")

        result = take(self.data, indices, allow_fill=allow_fill, fill_value=fill_value)
        if allow_fill and fill_value is None:
            result[pd.isna(result)] = None
        return UncertaintyArray(result)

    @classmethod
    def _concat_same_type(cls, to_concat):
        """
        Concatenate multiple array
        Parameters
        ----------
        to_concat : sequence of this type
        Returns
        -------
        ExtensionArray
        """
        data = np.concatenate([ga.data for ga in to_concat])
        return UncertaintyArray(data)

    def __getitem__(self, idx):
        if isinstance(idx, numbers.Integral):
            return self.data[idx]
        # array-like, slice
        # for pandas >= 1.0, validate and convert IntegerArray/BooleanArray
        # to numpy array, pass-through non-array-like indexers
        idx = pd.api.indexers.check_array_indexer(self, idx)
        if isinstance(idx, (Iterable, slice)):
            return UncertaintyArray(self.data[idx])
        else:
            raise TypeError("Index type not supported", idx)

    def _formatter(self, boxed=False):
        """Formatting function for scalar values.
        This is used in the default '__repr__'. The returned formatting
        function receives instances of your scalar type.
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
        return repr

    # def __repr__(self):
    #     print("B")
    #     return ""

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """
        Construct a new ExtensionArray from a sequence of scalars.
        Parameters
        ----------
        scalars : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type``.
        dtype : dtype, optional
            Construct for this particular dtype. This should be a Dtype
            compatible with the ExtensionArray.
        copy : boolean, default False
            If True, copy the underlying data.
        Returns
        -------
        ExtensionArray
        """
        return Uncertainty(scalars)

    @classmethod
    def _from_factorized(cls, values, original):
        """
        Reconstruct an ExtensionArray after factorization.
        Parameters
        ----------
        values : ndarray
            An integer ndarray with the factorized values.
        original : ExtensionArray
            The original ExtensionArray that factorize was called on.
        See Also
        --------
        pandas.factorize
        ExtensionArray.factorize
        """
        raise NotImplementedError

    def fillna(self, value=None, method=None, limit=None):
        """Fill NA/NaN values using the specified method.
        Parameters
        ----------
        value : scalar, array-like
            If a scalar value is passed it is used to fill all missing values.
            Alternatively, an array-like 'value' can be given. It's expected
            that the array-like have the same length as 'self'.
        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
            Method to use for filling holes in reindexed Series
            pad / ffill: propagate last valid observation forward to next valid
            backfill / bfill: use NEXT valid observation to fill gap
        limit : int, default None
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled.
        Returns
        -------
        filled : ExtensionArray with NA/NaN filled
        """
        if method is not None:
            raise NotImplementedError("fillna with a method is not yet supported")

        mask = self.isna()
        new_values = self.copy()

        if mask.any():

            # fill with value
            if not isinstance(value, Uncertainty):
                raise NotImplementedError(
                    "fillna currently only supports filling with an Uncertainty"
                )
            new_values.data[mask] = value

        return new_values

    def __init__(self, value, error=None):

        if isinstance(value, Uncertainty):
            self.data = value
        else:
            value = np.asarray(value)

            if not is_np_duck_array(type(value)):
                raise TypeError("Values provided must be ndarrays.")
            if not value.ndim == 1:
                raise ValueError("Values provided must be a 1D array")
            if error is None:
                error = np.zeros_like(value)
            else:
                error = np.asarray(error)
            self.data = Uncertainty(value, error)

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

        def _binop(self, other):
            if isinstance(other, (pd.Series, pd.DataFrame)):
                return NotImplemented

            lvalues = self.data
            rvalues = other
            res = op(lvalues, rvalues)

            if op.__name__ == "divmod":
                return (cls(res[0]), cls(res[1]))

            if coerce_to_dtype:
                try:
                    res = cls(res)
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


UncertaintyArray._add_arithmetic_ops()
UncertaintyArray._add_comparison_ops()
