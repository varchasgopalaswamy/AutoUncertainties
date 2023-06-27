# -*- coding: utf-8 -*-
from __future__ import annotations

import operator

import numpy as np
import pandas as pd
import pytest
from pandas.tests.extension.conftest import as_array  # noqa: F401
from pandas.tests.extension.conftest import as_frame  # noqa: F401
from pandas.tests.extension.conftest import as_series  # noqa: F401
from pandas.tests.extension.conftest import fillna_method  # noqa: F401
from pandas.tests.extension.conftest import groupby_apply_op  # noqa: F401
from pandas.tests.extension.conftest import use_numpy  # noqa: F401

from auto_uncertainties.pandas_compat import UncertaintyArray


@pytest.fixture
def data():
    return UncertaintyArray(
        np.random.random(size=100), np.abs(np.random.random(size=100))
    )


class TestUserInterface(object):
    def test_get_underlying_data(self, data: UncertaintyArray):
        ser = pd.Series(data)
        # this first test creates an array of bool (which is desired, eg for indexing)
        assert all(ser.values == data)
        assert ser.values[23] == data[23]

    def test_arithmetic(self, data: UncertaintyArray):
        ser = pd.Series(data)
        ser2 = ser + ser
        assert all(ser2.values == 2 * data)

    def test_initialisation(self, data: UncertaintyArray):
        # fails with plain array
        # works with UncertaintyArray
        df = pd.DataFrame(
            {
                "x1": pd.Series(data.uncertainty, dtype="Uncertainty[float]"),
                "x2": pd.Series(data.uncertainty.value, dtype="float"),
                "x3": pd.Series(
                    data.uncertainty.value,
                    data.uncertainty.error,
                    dtype="uncertainty[float32]",
                ),
            }
        )

        for col in df.columns:
            assert all(df[col] == df.length)

    def test_df_operations(self):
        # simply a copy of what's in the notebook
        df = pd.DataFrame(
            {
                "torque": pd.Series(
                    [1.0, 2.0, 2.0, 3.0], dtype="Uncertainty[float]"
                ),
                "angular_velocity": UncertaintyArray(
                    [1.0, 2.0, 2.0, 3.0],
                    [1.0, 2.0, 2.0, 3.0],
                    dtype="Uncertainty[float]",
                ),
            }
        )

        df["power"] = df["torque"] * df["angular_velocity"]

        df.power.values
        df.power.values.value
        df.power.values.error

        df.angular_velocity.values


arithmetic_ops = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.pow,
]

comparative_ops = [
    operator.eq,
    operator.le,
    operator.lt,
    operator.ge,
    operator.gt,
]

unit_ops = [
    operator.mul,
    operator.truediv,
]
