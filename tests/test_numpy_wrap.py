# -*- coding: utf-8 -*-
from __future__ import annotations

import operator
import warnings

import hypothesis.strategies as st
import numpy as np
import pint
import pytest
from hypothesis import given, settings
from hypothesis.extra import numpy as hnp

import auto_uncertainties

unit_registry = pint.UnitRegistry()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    unit_registry.Quantity([])
warnings.filterwarnings("ignore", category=pint.UnitStrippedWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
unit_registry.enable_contexts("boltzmann")
unit_registry.default_format = "0.8g~P"
unit_registry.default_system = "cgs"


UNITS = [None, unit_registry("s")]


def op_test(op, *args, **kwargs):
    with_unc = [a for a in args]
    without_unc = [a.value for a in args if hasattr(a, "_nom")]
    units = kwargs.pop("units", None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        w = op(*with_unc, **kwargs)
        w_ = op(*without_unc, **kwargs)
        if units is not None:
            assert w.units == w_.units
            w = w.m
            w_ = w_.m
    if hasattr(w, "_nom"):
        np.testing.assert_almost_equal(w_, w.value, decimal=5)
    else:
        np.testing.assert_almost_equal(w_, w)
    return w


def given_float_3d(func):
    return given(
        unom=hnp.arrays(
            dtype=st.sampled_from([np.float64]),
            shape=(2, 3, 5),
            unique=True,
            elements=st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        ),
        vnom=hnp.arrays(
            dtype=st.sampled_from([np.float64]),
            shape=(2, 3, 5),
            unique=True,
            elements=st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        ),
        uerr=hnp.arrays(
            dtype=st.sampled_from([np.float64]),
            shape=(2, 3, 5),
            unique=True,
            elements=st.floats(
                min_value=0.1,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        ),
        verr=hnp.arrays(
            dtype=st.sampled_from([np.float64]),
            shape=(2, 3, 5),
            unique=True,
            elements=st.floats(
                min_value=0.1,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        ),
        units=st.sampled_from(UNITS),
    )(func)


@given_float_3d
@settings(deadline=None)
def test_add(unom, uerr, vnom, verr, units):
    u = auto_uncertainties.Uncertainty(unom, uerr)
    v = auto_uncertainties.Uncertainty(vnom, verr)
    if units is not None:
        u *= units
        v *= units

    w = u + v
    if units is not None:
        assert w.units == units
        assert u.units == units
        assert v.units == units
        w = w.m
        u = u.m
        v = v.m
    w_mag = u.value + v.value
    w_err = np.sqrt(u.error**2 + v.error**2)
    np.testing.assert_almost_equal(w_mag, w.value)
    np.testing.assert_almost_equal(w_err, w.error)


@given_float_3d
@settings(deadline=None)
def test_same_shape(unom, uerr, vnom, verr, units):
    u = auto_uncertainties.Uncertainty(unom, uerr)
    v = auto_uncertainties.Uncertainty(vnom, verr)
    if units is not None:
        u *= units
        v *= units
    for op in auto_uncertainties.wrap_numpy.bcast_same_shape_ufuncs:
        oper = getattr(np, op)
        try:
            op_test(oper, u, units=units)
        except TypeError:
            op_test(oper, u, v, units=units)


@given_float_3d
@settings(deadline=None)
def test_same_shape_bool(unom, uerr, vnom, verr, units):
    u = auto_uncertainties.Uncertainty(unom, uerr)
    v = auto_uncertainties.Uncertainty(vnom, verr)
    if units is not None:
        u *= units
        v *= units
    for op in auto_uncertainties.wrap_numpy.bcast_same_shape_bool_ufuncs:
        oper = getattr(np, op)
        try:
            w = oper(u)
        except TypeError:
            w = oper(u, v)
        assert w.dtype == bool


@given_float_3d
@settings(deadline=None)
def test_nograd(unom, uerr, vnom, verr, units):
    u = auto_uncertainties.Uncertainty(unom, uerr)
    v = auto_uncertainties.Uncertainty(vnom, verr)
    if units is not None:
        u *= units
        v *= units

    for op in auto_uncertainties.wrap_numpy.bcast_nograd_ufuncs:
        oper = getattr(np, op)
        try:
            w = op_test(oper, u, units=units)
        except TypeError:
            w = op_test(oper, u, v, units=units)
        assert not isinstance(w, auto_uncertainties.Uncertainty)


@given_float_3d
@settings(deadline=None)
def test_apply_to_both(unom, uerr, vnom, verr, units):
    u = auto_uncertainties.Uncertainty(unom, uerr)
    v = auto_uncertainties.Uncertainty(vnom, verr)
    if units is not None:
        u *= units
        v *= units
    for op in auto_uncertainties.wrap_numpy.bcast_apply_to_both_ufuncs:
        oper = getattr(np, op)
        try:
            w = op_test(oper, u, units=units)
        except TypeError:
            try:
                w = op_test(oper, u, v, units=units)
            except (ValueError, TypeError):
                pass
        except ValueError:
            pass
    for op in auto_uncertainties.wrap_numpy.bcast_apply_to_both_funcs:
        oper = getattr(np, op)
        try:
            w = op_test(oper, u, units=units)
        except TypeError:
            try:
                w = op_test(oper, u, v, units=units)
            except (ValueError, TypeError):
                pass
        except ValueError:
            pass


@given_float_3d
@settings(deadline=None)
def test_selection(unom, uerr, vnom, verr, units):
    u = auto_uncertainties.Uncertainty(unom, uerr)
    v = auto_uncertainties.Uncertainty(vnom, verr)
    if units is not None:
        u *= units
        v *= units
    for op in auto_uncertainties.wrap_numpy.bcast_nograd_ufuncs:
        oper = getattr(np, op)
        try:
            w = op_test(oper, u, units=units)
        except TypeError:
            w = op_test(oper, u, v, units=units)
        assert not isinstance(w, auto_uncertainties.Uncertainty)

    for op in auto_uncertainties.wrap_numpy.bcast_selection_funcs:
        oper = getattr(np, op)
        try:
            w = op_test(oper, u, units=units)
        except TypeError:
            w = op_test(oper, u, v, units=units)
        assert not isinstance(w, auto_uncertainties.Uncertainty)


@given_float_3d
@settings(deadline=None)
def test_unary_reduction(unom, uerr, vnom, verr, units):
    u = auto_uncertainties.Uncertainty(unom, uerr)
    v = auto_uncertainties.Uncertainty(vnom, verr)
    if units is not None:
        u *= units
        v *= units
    for op in auto_uncertainties.wrap_numpy.bcast_reduction_unary:
        oper = getattr(np, op)
        try:
            w = op_test(oper, u, units=units)
        except TypeError:
            w = op_test(oper, u, v, units=units)
