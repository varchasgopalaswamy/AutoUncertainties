from __future__ import annotations

import inspect

import numpy as np
from pint import UnitRegistry
import pytest

from auto_uncertainties import Uncertainty


@pytest.mark.parametrize(
    ("name", "args", "kwargs"),
    (
        ("sum", (0, np.float64), {}),
        ("mean", (), {"axis": 1}),
        ("std", (1, None, None, 1), {}),
        ("var", (), {"axis": 0}),
        ("max", (), {"axis": 0}),
        ("min", (), {"axis": 1}),
        ("ravel", ("F",), {}),
        ("repeat", (2,), {"axis": 0}),
        ("squeeze", (), {}),
        ("swapaxes", (0, 1), {}),
        ("diagonal", (0, 0, 1), {}),
        ("take", ([1, 0],), {"axis": 1}),
        ("round", (1,), {}),
        ("sort", (), {"axis": 0}),
        ("conjugate", (), {}),
    ),
)
def test_methods_use_numpy_handlers(name, args, kwargs):
    source = Uncertainty(np.array([[3.0, 1.0], [4.0, 2.0]]), np.array([[0.3, 0.1], [0.4, 0.2]]))
    method = getattr(source, name)
    assert not any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in inspect.signature(method).parameters.values()
    )
    actual = method(*args, **kwargs)
    expected = getattr(np, name)(source, *args, **kwargs)
    np.testing.assert_array_equal(actual.value, expected.value)
    np.testing.assert_array_equal(actual.error, expected.error)
    with pytest.raises(TypeError):
        method(unknown_option=True)


def test_reshape_and_transpose_calling_conventions_and_views():
    source = Uncertainty(np.arange(6.0), np.arange(6.0) / 10)
    for result in (source.reshape(2, 3), source.reshape((2, 3))):
        assert np.shares_memory(source.value, result.value)
        assert np.shares_memory(source.error, result.error)
        np.testing.assert_array_equal(result.transpose(1, 0).value, result.T.value)
        np.testing.assert_array_equal(result.transpose((1, 0)).error, result.T.error)
    copied = source.reshape((2, 3), copy=True)
    assert not np.shares_memory(source.value, copied.value)
    assert not np.shares_memory(source.error, copied.error)


def test_method_outputs_are_transactional_and_preserve_identity():
    source = Uncertainty([1, 2], [0.1, 0.2])
    out = Uncertainty([10, 20], [1, 2])
    assert source.clip(1.5, None, out) is out
    np.testing.assert_array_equal(out.value, [1.5, 2])
    np.testing.assert_array_equal(out.error, source.error)
    out.error.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        source.clip(0, 1, out)
    np.testing.assert_array_equal(out.value, [1.5, 2])


def test_overlapping_output_components_are_rejected_without_writes():
    buffer = np.array([10.0, 20.0, 30.0])
    out = Uncertainty(buffer[:2], buffer[1:])
    with pytest.raises(ValueError, match="overlap"):
        np.add(Uncertainty([1, 2], [0.1, 0.2]), 1, out=out)
    np.testing.assert_array_equal(buffer, [10, 20, 30])


def test_output_can_overlap_input_storage():
    source = Uncertainty([1, 2, 3], [0.1, 0.2, 0.3])
    returned = np.multiply(source[:-1], 2, out=source[1:])
    np.testing.assert_array_equal(returned.value, [2, 4])
    np.testing.assert_allclose(source.error, [0.1, 0.2, 0.4])


def test_construction_initializes_once_with_and_without_units():
    calls = []

    class Counted(Uncertainty):
        def __init__(self, value, error=None):
            calls.append(value)
            super().__init__(value, error)

    units = UnitRegistry()
    cases = (
        (1, 0.1),
        ([1, 2], [0.1, 0.2]),
        (np.array([1.0, 2.0]), 0.1),
        (Uncertainty(1, 0.1), 999),
        (2 * units.m, 10 * units.cm),
        ([1 * units.m, 200 * units.cm], [0.1 * units.m, 20 * units.cm]),
    )
    for value, error in cases:
        calls.clear()
        result = Counted(value, error)
        assert len(calls) == 1
        if hasattr(result, "units"):
            assert result.units == units.m
            np.testing.assert_allclose(result.m.error, [0.1, 0.2] if result.m.is_vector else 0.1)


def test_removed_public_exports():
    import auto_uncertainties as package

    for name in (
        "ScalarUncertainty",
        "VectorUncertainty",
        "ScalarDisplay",
        "VectorDisplay",
        "UncertaintyDisplay",
        "P",
        "R",
        "elementwise_value_and_grad",
    ):
        assert name not in package.__all__
        assert not hasattr(package, name)
    with pytest.raises(TypeError):
        Uncertainty(1, 0.1, skip=True)
