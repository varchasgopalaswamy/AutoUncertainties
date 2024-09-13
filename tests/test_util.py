from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
import pytest

from auto_uncertainties.exceptions import DowncastWarning
from auto_uncertainties.util import (
    has_length,
    ignore_numpy_downcast_warnings,
    ignore_runtime_warnings,
    is_iterable,
    ndarray_to_scalar,
    strip_device_array,
)


@pytest.mark.parametrize(
    "warning, should_raise",
    [
        (RuntimeWarning, False),
    ],
)
def test_ignore_runtime_warnings(warning, should_raise):
    @ignore_runtime_warnings
    def test_func():
        warnings.warn("warning!", warning)

    if should_raise:
        with pytest.warns(warning):
            test_func()
    else:
        test_func()

    # TODO: Improve this (decorator seems to block all warnings?)


@pytest.mark.parametrize("warning, should_raise", [(DowncastWarning, False)])
def test_ignore_numpy_downcast_warnings(warning, should_raise):
    @ignore_numpy_downcast_warnings
    def test_func():
        warnings.warn("warning!", warning)

    if should_raise:
        with pytest.warns(warning):
            test_func()
    else:
        test_func()

    # TODO: Improve this (decorator seems to block all warnings?)


@pytest.mark.parametrize(
    "obj, iterable",
    [
        ([1, 2, 3], True),
        (2.4, False),
        ({"a": 1, "b": 2}, True),
        ("string", True),
        (np.array([1, 2]), True),
        (Exception(), False),
    ],
)
def test_is_iterable(obj, iterable):
    assert is_iterable(obj) is iterable


@pytest.mark.parametrize(
    "obj, length",
    [
        ("string", True),
        (2.4, False),
        (np.array([1, 2, 3]), True),
        (Exception(), False),
    ],
)
def test_has_length(obj, length):
    assert has_length(obj) is length


@pytest.mark.parametrize(
    "val, expected",
    [
        (np.array([1]), 1),
        (np.array(["test"]), "test"),
    ],
)
def test_ndarray_to_scalar(val: NDArray, expected):
    assert ndarray_to_scalar(val) == expected


@pytest.mark.parametrize(
    "val, expected",
    [
        ([1, 2, 3], np.array([1, 2, 3])),
        (np.array([1, 2]), np.array([1, 2])),
        (2.5, np.array(2.5)),
    ],
)
def test_strip_device_array(val, expected):
    assert np.array_equal(strip_device_array(val), expected)
