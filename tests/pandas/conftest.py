from __future__ import annotations

import operator

import numpy as np
from pandas import Series
import pytest

from auto_uncertainties import Uncertainty, UncertaintyArray, UncertaintyDtype


@pytest.fixture()
def dtype():
    """A fixture providing the ExtensionDtype to validate."""
    return UncertaintyDtype(np.float64)


@pytest.fixture()
def data():
    """
    Length-100 array for this type.

    * data[0] and data[1] should both be non missing
    * data[0] and data[1] should not be equal
    """

    v = np.random.random(size=100)
    e = np.random.random(size=100)

    return UncertaintyArray(v, e)


@pytest.fixture()
def data_for_twos(dtype):
    """
    Length-100 array in which all the elements are two.

    Call pytest.skip in your fixture if the dtype does not support divmod.
    """
    if not (dtype._is_numeric or dtype.kind == "m"):
        # Object-dtypes may want to allow this, but for the most part
        #  only numeric and timedelta-like dtypes will need to implement this.
        pytest.skip(f"{dtype} is not a numeric dtype")

    v = np.ones(100) * 2
    e = np.random.random(size=100)
    return UncertaintyArray(v, e)


@pytest.fixture()
def data_missing():
    """Length-2 array with [NA, Valid]"""
    return UncertaintyArray([np.nan, 1], [0, 0.1])


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """Parametrized fixture giving 'data' and 'data_missing'"""
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


@pytest.fixture()
def data_repeated(data):
    """
    Generate many datasets.

    Parameters
    ----------
    data : fixture implementing `data`

    Returns
    -------
    Callable[[int], Generator]:
        A callable that takes a `count` argument and
        returns a generator yielding `count` datasets.
    """

    def gen(count):
        for _ in range(count):
            yield data

    return gen


@pytest.fixture()
def data_for_sorting():
    """
    Length-3 array with a known sort order.

    This should be three items [B, C, A] with
    A < B < C

    For boolean dtypes (for which there are only 2 values available),
    set B=C=True
    """

    return UncertaintyArray([1, 2, 0], [0.1, 0.2, 0.3])


@pytest.fixture()
def data_missing_for_sorting():
    """
    Length-3 array with a known sort order.

    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    return UncertaintyArray([1, np.nan, 0], [0.1, np.nan, 0.3])


@pytest.fixture()
def na_cmp():
    """
    Binary operator for comparing NA values.

    Should return a function of two arguments that returns
    True if both arguments are (scalar) NA for your type.

    By default, uses ``operator.is_``
    """

    def na_compare(x, y):
        if isinstance(x, Uncertainty):
            xval = x.value
        else:
            xval = x
        if isinstance(y, Uncertainty):
            yval = y.value
        else:
            yval = y
        if np.isnan(xval) and np.isnan(yval):
            return True
        return operator.is_(xval, yval)

    return na_compare


@pytest.fixture()
def na_value(dtype):
    """
    The scalar missing value for this type. Default dtype.na_value.

    TODO: can be removed in 3.x (see https://github.com/pandas-dev/pandas/pull/54930)
    """
    return dtype.na_value


@pytest.fixture()
def data_for_grouping():
    """
    Data for factorization, grouping, and unique tests.

    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing.

    If a dtype has _is_boolean = True, i.e. only 2 unique non-NA entries,
    then set C=B.
    """
    return UncertaintyArray([1, 1, np.nan, np.nan, 0, 0, 1, 2], [0.1] * 8)


@pytest.fixture(params=[True, False])
def box_in_series(request):
    """Whether to box the data in a Series"""
    return request.param


_all_arithmetic_operators = [
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__floordiv__",
    "__rfloordiv__",
    "__truediv__",
    "__rtruediv__",
    "__pow__",
    "__rpow__",
    "__mod__",
    "__rmod__",
]


@pytest.fixture(params=_all_arithmetic_operators)
def all_arithmetic_operators(request):
    """
    Fixture for dunder names for common arithmetic operations
    """
    return request.param


@pytest.fixture(params=["__eq__", "__ne__", "__le__", "__lt__", "__ge__", "__gt__"])
def all_compare_operators(request):
    """
    Fixture for dunder names for common compare operations

    * >=
    * >
    * ==
    * !=
    * <
    * <=
    """
    return request.param


# commented functions aren't implemented in numpy/pandas
_all_numeric_reductions = [
    "sum",
    "max",
    "min",
    "mean",
    # "prod",
    "std",
    "var",
    "median",
    "sem",
    "kurt",
    "skew",
]


@pytest.fixture(params=_all_numeric_reductions)
def all_numeric_reductions(request):
    """
    Fixture for numeric reduction names.
    """
    return request.param


_all_boolean_reductions = []


@pytest.fixture(params=_all_boolean_reductions)
def all_boolean_reductions(request):
    """
    Fixture for boolean reduction names.
    """
    return request.param


_all_numeric_accumulations = ["cumsum", "cumprod", "cummin", "cummax"]


@pytest.fixture(params=_all_numeric_accumulations)
def all_numeric_accumulations(request):
    """
    Fixture for numeric accumulation names
    """
    return request.param


@pytest.fixture(
    params=[
        operator.ge,
        operator.gt,
        operator.le,
        operator.lt,
        operator.eq,
        operator.ne,
    ],
)
def comparison_op(request):
    """
    Functions to test groupby.apply().
    """
    return request.param


@pytest.fixture(
    params=[
        lambda x: 1,
        lambda x: [1] * len(x),
        lambda x: Series([1] * len(x)),
        lambda x: x,
    ],
    ids=["scalar", "list", "series", "object"],
)
def groupby_apply_op(request):
    """
    Functions to test groupby.apply().
    """
    return request.param


@pytest.fixture(params=[None, lambda x: x])
def sort_by_key(request):
    """
    Simple fixture for testing keys in sorting methods.
    Tests None (no key) and the identity key.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_frame(request):
    """
    Boolean fixture to support Series and Series.to_frame() comparison testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_series(request):
    """
    Boolean fixture to support arr and Series(arr) comparison testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def use_numpy(request):
    """
    Boolean fixture to support comparison testing of ExtensionDtype array
    and numpy array.
    """
    return request.param


@pytest.fixture(params=["ffill", "bfill"])
def fillna_method(request):
    """
    Parametrized fixture giving method parameters 'ffill' and 'bfill' for
    Series.<method> testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_array(request):
    """
    Boolean fixture to support ExtensionDtype _from_sequence method testing.
    """
    return request.param


@pytest.fixture()
def invalid_scalar(data):
    """
    A scalar that *cannot* be held by this ExtensionArray.

    The default should work for most subclasses, but is not guaranteed.

    If the array can hold any item (i.e. object dtype), then use pytest.skip.
    """
    return object.__new__(object)
