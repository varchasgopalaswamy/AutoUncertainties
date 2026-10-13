from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
import pytest

from auto_uncertainties.exceptions import DowncastWarning
from auto_uncertainties.util import (
    deprecated,
    has_length,
    ignore_numpy_downcast_warnings,
    ignore_runtime_warnings,
    is_iterable,
    ndarray_to_scalar,
    strip_device_array,
)


@pytest.mark.parametrize(
    "warning",
    (RuntimeWarning,),
)
def test_ignore_runtime_warnings(warning):
    @ignore_runtime_warnings
    def test_func_silent():
        warnings.warn("warning!", warning, stacklevel=2)

    def test_func():
        warnings.warn("warning!", warning, stacklevel=2)

    warnings.simplefilter("error")  # treat warnings as errors

    with pytest.warns(warning):
        test_func()

    test_func_silent()


@pytest.mark.parametrize("warning", (DowncastWarning,))
def test_ignore_numpy_downcast_warnings(warning):
    @ignore_numpy_downcast_warnings
    def test_func_silent():
        warnings.warn("warning!", warning, stacklevel=2)

    def test_func():
        warnings.warn("warning!", warning, stacklevel=2)

    warnings.simplefilter("error")  # treat warnings as errors

    with pytest.warns(warning):
        test_func()

    test_func_silent()


@pytest.mark.parametrize(
    ("decorator", "suppressed", "retained"),
    (
        (ignore_runtime_warnings, RuntimeWarning, UserWarning),
        (ignore_runtime_warnings, RuntimeWarning, DeprecationWarning),
        (ignore_numpy_downcast_warnings, DowncastWarning, RuntimeWarning),
    ),
)
def test_warning_decorators_preserve_other_categories(decorator, suppressed, retained):
    @decorator
    def example():
        warnings.warn("suppressed", suppressed, stacklevel=2)
        warnings.warn("retained", retained, stacklevel=2)
        return 42

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert example() == 42
        warnings.warn("outside", suppressed, stacklevel=2)
    assert [(warning.category, str(warning.message)) for warning in seen] == [
        (retained, "retained"),
        (suppressed, "outside"),
    ]


@pytest.mark.parametrize("reason", ("", "use replacement instead"))
def test_deprecated_includes_function_name(reason):
    @deprecated(reason)
    def example():
        return 42

    with pytest.warns(DeprecationWarning, match="example is deprecated") as seen:
        assert example() == 42
    suffix = f" ({reason})." if reason else "."
    assert str(seen[0].message) == f"example is deprecated{suffix}"


@pytest.mark.parametrize(
    ("obj", "iterable"),
    (
        ([1, 2, 3], True),
        (2.4, False),
        ({"a": 1, "b": 2}, True),
        ("string", True),
        (np.array([1, 2]), True),
        (Exception(), False),
    ),
)
def test_is_iterable(obj, iterable):
    assert is_iterable(obj) is iterable


@pytest.mark.parametrize(
    ("obj", "length"),
    (
        ("string", True),
        (2.4, False),
        (np.array([1, 2, 3]), True),
        (Exception(), False),
    ),
)
def test_has_length(obj, length):
    assert has_length(obj) is length


@pytest.mark.parametrize(
    ("val", "expected"),
    (
        (np.array([1]), 1),
        (np.array(["test"]), "test"),
    ),
)
def test_ndarray_to_scalar(val: NDArray, expected):
    assert ndarray_to_scalar(val) == expected


@pytest.mark.parametrize(
    ("val", "expected"),
    (
        ([1, 2, 3], np.array([1, 2, 3])),
        (np.array([1, 2]), np.array([1, 2])),
        (2.5, np.array(2.5)),
    ),
)
def test_strip_device_array(val, expected):
    assert np.array_equal(strip_device_array(val), expected)
