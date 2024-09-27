from __future__ import annotations

import warnings

from hypothesis import given, settings
from hypothesis.extra import numpy as hnp
import hypothesis.strategies as st
import numpy as np

import auto_uncertainties

try:
    from pint import DimensionalityError
except ImportError:

    class DimensionalityError(Exception):
        pass


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


UNITS = [None]


def op_test(op, *args, **kwargs):
    with_unc = [a for a in args]
    without_unc = [a.value for a in args if hasattr(a, "_nom")]
    units = kwargs.pop("units", None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            w_ = op(*without_unc, **kwargs)
        except DimensionalityError:
            with_unc = [a.m for a in args]
            without_unc = [a.value.m for a in args if hasattr(a, "_nom")]
            units = None
            w_ = op(*without_unc, **kwargs)
        except TypeError:
            return None
        w = op(*with_unc, **kwargs)
        if units is not None and hasattr(w_, "units"):
            assert w.units == w_.units
            w = w.m
            w_ = w_.m
    if hasattr(w, "_nom"):
        np.testing.assert_almost_equal(w_, w.value, decimal=5)
    else:
        np.testing.assert_almost_equal(w_, w)
    return w


def given_float_3d(ops):
    def inner(func):
        return given(
            unom=hnp.arrays(
                dtype=st.sampled_from([np.float64]),
                shape=(2, 3),
                elements=st.floats(
                    min_value=-10.0,
                    max_value=10.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            ),
            uerr=hnp.arrays(
                dtype=st.sampled_from([np.float64]),
                shape=(2, 3),
                elements=st.floats(
                    min_value=0.0,
                    max_value=10.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            ),
            units=st.sampled_from(UNITS),
            op=st.sampled_from(ops),
        )(func)

    return inner


@given_float_3d(auto_uncertainties.numpy.numpy_wrappers.bcast_same_shape_ufuncs)
@settings(deadline=3000)
def test_same_shape(unom, uerr, units, op):
    vnom = unom / 2
    verr = uerr / 2
    u = auto_uncertainties.Uncertainty(unom, uerr)
    v = auto_uncertainties.Uncertainty(vnom, verr)
    assert u.shape == v.shape
    if units is not None:
        u *= units
        v *= units
    oper = getattr(np, op)
    if op in auto_uncertainties.numpy.numpy_wrappers.unary_bcast_same_shape_ufuncs:
        op_test(oper, u, units=units)
    else:
        op_test(oper, u, v, units=units)


@given_float_3d(auto_uncertainties.numpy.numpy_wrappers.bcast_same_shape_bool_ufuncs)
@settings(deadline=3000)
def test_same_shape_bool(unom, uerr, units, op):
    vnom = unom / 2
    verr = uerr / 2
    u = auto_uncertainties.Uncertainty(unom, uerr)
    v = auto_uncertainties.Uncertainty(vnom, verr)
    if units is not None:
        u *= units
        v *= units
    oper = getattr(np, op)
    if op in auto_uncertainties.numpy.numpy_wrappers.unary_bcast_same_shape_bool_ufuncs:
        w = oper(u)
    else:
        w = oper(u, v)
    assert w.dtype == bool


@given_float_3d(auto_uncertainties.numpy.numpy_wrappers.bcast_nograd_ufuncs)
@settings(deadline=3000)
def test_nograd(unom, uerr, units, op):
    vnom = unom / 2
    verr = uerr / 2
    u = auto_uncertainties.Uncertainty(unom, uerr)
    v = auto_uncertainties.Uncertainty(vnom, verr)
    if units is not None:
        u *= units
        v *= units

    oper = getattr(np, op)
    try:
        w = op_test(oper, u, units=units)
    except TypeError:
        w = op_test(oper, u, v, units=units)
    assert not isinstance(w, auto_uncertainties.Uncertainty)


@given_float_3d(auto_uncertainties.numpy.numpy_wrappers.bcast_apply_to_both_ufuncs)
@settings(deadline=3000)
def test_apply_to_both(unom, uerr, units, op):
    vnom = unom / 2
    verr = uerr / 2
    u = auto_uncertainties.Uncertainty(unom, uerr)
    v = auto_uncertainties.Uncertainty(vnom, verr)
    if units is not None:
        u *= units
        v *= units
    oper = getattr(np, op)
    op_test(oper, u, units=units)


@given_float_3d(auto_uncertainties.numpy.numpy_wrappers.bcast_reduction_unary)
@settings(deadline=3000)
def test_unary_reduction(unom, uerr, units, op):
    vnom = unom / 2
    verr = uerr / 2
    u = auto_uncertainties.Uncertainty(unom, uerr)
    v = auto_uncertainties.Uncertainty(vnom, verr)
    if units is not None:
        u *= units
        v *= units
    oper = getattr(np, op)
    op_test(oper, u, units=units)
