from __future__ import annotations

from typing import final

import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
import pytest

from auto_uncertainties import UncertaintyArray, nominal_values
from auto_uncertainties.uncertainty import set_downcast_error
from auto_uncertainties.uncertainty.uncertainty_containers import (
    set_compare_error,
)

set_downcast_error(True)
set_compare_error(1e-4)


class TestUncertaintyArray(base.ExtensionTests):
    def test_setitem_sequence(self, data, box_in_series):
        if box_in_series:
            data = pd.Series(data)
        original = data.copy()

        data[[0, 1]] = [data[1], data[0]]
        if data[0] != original[1]:
            print(data[0], original[1])
            raise AssertionError("Setting with sequence failed")

    def test_invert(self, data, box_in_series):
        if box_in_series:
            data = pd.Series(data)
        with pytest.raises(TypeError):
            data = ~data
            assert data[0] == ~data[0]

    def test_add_series_with_extension_array(self, data):
        ser = pd.Series(data)

        result = ser + data
        expected = pd.Series(data + data)
        tm.assert_series_equal(result, expected)

    def test_contains(self, data, data_missing):
        # GH-37867
        # Tests for membership checks. Membership checks for nan-likes is tricky and
        # the settled on rule is: `nan_like in arr` is True if nan_like is
        # arr.dtype.na_value and arr.isna().any() is True. Else the check returns False.

        na_value = data.dtype.na_value
        # ensure data without missing values
        data = data[~data.isna()]

        # first elements are non-missing
        assert data_missing[0] in data_missing
        assert data[0] in data

        # check the presence of na_value
        assert na_value in data_missing
        assert na_value not in data

        # the data can never contain other nan-likes than na_value
        for na_value_obj in tm.NULL_OBJECTS:
            if na_value_obj is na_value or type(na_value_obj) == type(na_value):
                # type check for e.g. two instances of Decimal("NAN")
                continue
            assert na_value_obj not in data
            assert na_value_obj not in data_missing

    def _compare_other(self, ser: pd.Series, data, op, other):
        if op.__name__ in ["eq", "ne"]:
            # comparison should match point-wise comparisons
            result = op(ser, other)
            expected = ser.combine(other, op)
            expected = self._cast_pointwise_result(op.__name__, ser, other, expected)
            tm.assert_series_equal(result, expected)

        else:
            result = op(ser, other)

            # Didn't error, then should match pointwise behavior
            expected = ser.combine(other, op)
            expected = self._cast_pointwise_result(op.__name__, ser, other, expected)
            tm.assert_series_equal(result, expected)

    def test_compare_scalar(self, data, comparison_op):
        ser = pd.Series(data)
        self._compare_other(ser, data, comparison_op, 0)

    def test_compare_array(self, data, comparison_op):
        ser = pd.Series(data)
        other = pd.Series([data[0]] * len(data), dtype=data.dtype)
        self._compare_other(ser, data, comparison_op, other)

    @final
    def check_opname(self, ser: pd.Series, op_name: str, other):
        op = self.get_op_from_name(op_name)

        self._check_op(ser, op, other, op_name, None)

    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        # ndarray & other series
        op_name = all_arithmetic_operators
        ser = pd.Series(data)
        self.check_opname(
            ser, op_name, pd.Series([ser.iloc[0]] * len(ser), dtype=ser.dtype)
        )

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        # Specify if we expect this reduction to succeed.
        return True

    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool):
        # We perform the same operation on the np.float64 data and check
        #  that the results match. Override if you need to cast to something
        #  other than float64.
        if op_name in UncertaintyArray._supported_reductions:
            res_op = getattr(ser, op_name)
        else:
            with pytest.raises(TypeError):
                res_op = getattr(ser, op_name)

        try:
            alt = ser.astype("float64")
        except (TypeError, ValueError):
            # e.g. Interval can't cast (TypeError), StringArray can't cast
            #  (ValueError), so let's cast to object and do
            #  the reduction pointwise
            alt = ser.astype(object)

        exp_op = getattr(alt, op_name)
        if op_name == "count":
            result = res_op()
            expected = exp_op()
        else:
            result = res_op(skipna=skipna)
            expected = exp_op(skipna=skipna)

        tm.assert_almost_equal(nominal_values(result), nominal_values(expected))
