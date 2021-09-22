# -*- coding: utf-8 -*-
import pytest
import pint

unit_registry = pint.UnitRegistry()
import hypothesis.strategies as st
from hypothesis import given, settings
from hypothesis.extra import numpy as hnp
import numpy as np
import operator

import uncert 

def op_test(op,*args,**kwargs):
    with_unc = [a for a in args]
    without_unc = [a._nom for a in args if hasattr(a,'_nom')]
    w = op(*with_unc,**kwargs)
    w_ = op(*without_unc,**kwargs)
    if hasattr(w,'_nom'):
        np.testing.assert_almost_equal(w_,w._nom)
    else:
        np.testing.assert_almost_equal(w_,w)
    return w 


def given_float_3d(func):
    return given(unom=hnp.arrays(
        dtype=st.sampled_from([np.float64]),
        shape=(31, 29, 11),
        elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    ), 
    vnom=hnp.arrays(
        dtype=st.sampled_from([np.float64]),
        shape=(31, 29, 11),
        elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    ),
    uerr=hnp.arrays(
        dtype=st.sampled_from([np.float64]),
        shape=(31, 29, 11),
        elements=st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False)
    ), 
    verr=hnp.arrays(
        dtype=st.sampled_from([np.float64]),
        shape=(31, 29, 11),
        elements=st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False)
    ),
    
    )(func)

def given_float_2d(func):
    return given(unom=hnp.arrays(
        dtype=st.sampled_from([np.float64]),
        shape=(31, 29),
        elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    ), 
    vnom=hnp.arrays(
        dtype=st.sampled_from([np.float64]),
        shape=(31, 29),
        elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    ),
    uerr=hnp.arrays(
        dtype=st.sampled_from([np.float64]),
        shape=(31, 29),
        elements=st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False)
    ), 
    verr=hnp.arrays(
        dtype=st.sampled_from([np.float64]),
        shape=(31, 29),
        elements=st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False)
    ))(func)

@given_float_3d
def test_same_shape(unom,uerr,vnom,verr):
    u = uncert.Uncertainty(unom,uerr)
    v = uncert.Uncertainty(vnom,verr)
    for op in uncert.wrap_numpy.bcast_same_shape_ufuncs:
        oper = getattr(np,op)
        try:
            op_test(oper,u)
        except TypeError:
            op_test(oper,u,v)

@given_float_3d
def test_same_shape(unom,uerr,vnom,verr):
    u = uncert.Uncertainty(unom,uerr)
    v = uncert.Uncertainty(vnom,verr)
    for op in uncert.wrap_numpy.bcast_same_shape_bool_ufuncs:
        oper = getattr(np,op)
        try:
            w = oper(u)
        except TypeError:
            w = oper(u,v)
        assert w.dtype == bool

@given_float_3d
def test_nograd(unom,uerr,vnom,verr):
    u = uncert.Uncertainty(unom,uerr)
    v = uncert.Uncertainty(vnom,verr)
    for op in uncert.wrap_numpy.bcast_nograd_ufuncs:
        oper = getattr(np,op)
        try:
            w = op_test(oper,u)
        except TypeError:
            w = op_test(oper,u,v)
        assert not isinstance(w,uncert.Uncertainty)

@given_float_3d
def test_selection(unom,uerr,vnom,verr):
    u = uncert.Uncertainty(unom,uerr)
    v = uncert.Uncertainty(vnom,verr)
    for op in uncert.wrap_numpy.bcast_nograd_ufuncs:
        oper = getattr(np,op)
        try:
            w = op_test(oper,u)
        except TypeError:
            w = op_test(oper,u,v)
        assert not isinstance(w,uncert.Uncertainty)
    
    for op in uncert.wrap_numpy.bcast_selection_ufuncs:
        oper = getattr(np,op)
        try:
            w = op_test(oper,u)
        except TypeError:
            w = op_test(oper,u,v)
        assert not isinstance(w,uncert.Uncertainty)

@given_float_3d
@settings(deadline=None)
def test_unary_reduction(unom,uerr,vnom,verr):
    u = uncert.Uncertainty(unom,uerr)
    v = uncert.Uncertainty(vnom,verr)
    for op in uncert.wrap_numpy.bcast_reduction_unary:
        oper = getattr(np,op)
        try:
            w = op_test(oper,u)
            w = op_test(oper,u,axis=1)
        except TypeError:
            w = op_test(oper,u,v)
            w = op_test(oper,u,v,axis=1)