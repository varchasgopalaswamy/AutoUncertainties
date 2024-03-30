# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
from pandas.tests.extension import base

from auto_uncertainties.uncertainty import set_downcast_error

set_downcast_error(True)


class TestUncertaintyArray(base.ExtensionTests):
    def test_setitem_sequence(self, data, box_in_series):
        if box_in_series:
            data = pd.Series(data)
        original = data.copy()

        data[[0, 1]] = [data[1], data[0]]
        if data[0] != original[1]:
            print(data[0], original[1])
            raise AssertionError("Setting with sequence failed")
