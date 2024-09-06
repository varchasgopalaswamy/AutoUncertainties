from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st
import numpy as np
import pytest

from auto_uncertainties.display_format import (
    ScalarDisplay,
    VectorDisplay,
    pdg_round,
)


class TestVectorDisplay:
    @staticmethod
    @pytest.mark.parametrize(
        "arr1, arr2, expected",
        [
            (
                np.array([1, 2, 3]),
                np.array([4, 5, 6]),
                "<table><tbody><tr><th>Magnitude</th><td style='text-align:left;'><pre>1, 2, 3</pre></td></tr><tr><th>Error</th><td style='text-align:left;'><pre>4, 5, 6</pre></td></tr></tbody></table>",
            ),
        ],
    )
    def test__repr_html_(arr1, arr2, expected):
        vd = VectorDisplay()
        vd._nom = arr1
        vd._err = arr2
        assert vd._repr_html_() == expected

    @staticmethod
    @pytest.mark.parametrize(
        "arr1, arr2, expected",
        [(np.array([1, 2, 3]), np.array([4, 5, 6]), "$1 \\pm 4, 2 \\pm 5, 3 \\pm 6~$")],
    )
    def test__repr_latex_(arr1, arr2, expected):
        vd = VectorDisplay()
        vd._nom = arr1
        vd._err = arr2
        assert vd._repr_latex_() == expected

    @staticmethod
    @pytest.mark.parametrize(
        "arr1, arr2, expected",
        [(np.array([1, 2, 3]), np.array([4, 5, 6]), "[1 +/- 4, 2 +/- 5, 3 +/- 6]")],
    )
    def test_str_and_repr(arr1, arr2, expected):
        vd = VectorDisplay()
        vd._nom = arr1
        vd._err = arr2
        assert vd.__str__() == expected
        assert vd.__repr__() == expected

    @staticmethod
    @pytest.mark.parametrize(
        "arr1, arr2, fmt, expected",
        [
            (
                np.array([1, 2, 3]),
                np.array([4, 5, 6]),
                "0.2f",
                "[1.00 +/- 4.00, 2.00 +/- 5.00, 3.00 +/- 6.00]",
            )
        ],
    )
    def test___format__(arr1, arr2, fmt, expected):
        vd = VectorDisplay()
        vd._nom = arr1
        vd._err = arr2
        assert vd.__format__(fmt) == expected


class TestScalarDisplay:
    @staticmethod
    @given(
        st.floats(min_value=0, max_value=5.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    def test__repr_html_(f1, f2):
        vd = ScalarDisplay()
        vd._nom = f1
        vd._err = f2
        assert vd._repr_html_() == f"{pdg_round(f1, f2)[0]} ± {pdg_round(f1, f2)[1]}"

        # TODO: Should be improved to get full coverage!

    @staticmethod
    @given(
        st.floats(min_value=0, max_value=5.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    def test__repr_latex_(f1, f2):
        vd = ScalarDisplay()
        vd._nom = f1
        vd._err = f2
        assert (
            vd._repr_latex_() == f"{pdg_round(f1, f2)[0]} \\pm {pdg_round(f1, f2)[1]}"
        )

        # TODO: Should be improved to get full coverage!

    @staticmethod
    @given(
        st.floats(min_value=0, max_value=5.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    def test_str_and_repr(f1, f2):
        vd = ScalarDisplay()
        vd._nom = f1
        vd._err = f2
        assert vd.__str__() == f"{pdg_round(f1, f2)[0]} +/- {pdg_round(f1, f2)[1]}"
        assert vd.__repr__() == f"{pdg_round(f1, f2)[0]} +/- {pdg_round(f1, f2)[1]}"

        # TODO: Should be improved to get full coverage!

    @staticmethod
    @given(
        st.floats(min_value=0, max_value=5.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    def test___format__(f1, f2):
        vd = ScalarDisplay()
        vd._nom = f1
        vd._err = f2
        assert vd.__format__("") == f"{pdg_round(f1, f2)[0]} +/- {pdg_round(f1, f2)[1]}"

        # TODO: Should be improved to get full coverage!


def test_first_digit(): ...
