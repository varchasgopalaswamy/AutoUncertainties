from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st
import numpy as np
import pytest

from auto_uncertainties.display_format import (
    PDG_precision,
    ScalarDisplay,
    VectorDisplay,
    first_digit,
    pdg_round,
    set_display_rounding,
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
        sd = ScalarDisplay()
        sd._nom = f1
        sd._err = f2
        assert sd._repr_html_() == f"{pdg_round(f1, f2)[0]} ± {pdg_round(f1, f2)[1]}"

    @staticmethod
    @given(
        st.floats(min_value=0, max_value=5.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    def test__repr_latex_(f1, f2):
        sd = ScalarDisplay()
        sd._nom = f1
        sd._err = f2
        assert (
            sd._repr_latex_() == f"{pdg_round(f1, f2)[0]} \\pm {pdg_round(f1, f2)[1]}"
        )

    @staticmethod
    @given(
        st.floats(min_value=0, max_value=5.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    def test_str_and_repr(f1, f2):
        sd = ScalarDisplay()
        sd._nom = f1
        sd._err = f2
        assert sd.__str__() == f"{pdg_round(f1, f2)[0]} +/- {pdg_round(f1, f2)[1]}"
        assert sd.__repr__() == f"{pdg_round(f1, f2)[0]} +/- {pdg_round(f1, f2)[1]}"

    @staticmethod
    @given(
        st.floats(min_value=0, max_value=5.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    def test___format__(f1, f2):
        sd = ScalarDisplay()
        sd._nom = f1
        sd._err = f2
        assert sd.__format__("") == f"{pdg_round(f1, f2)[0]} +/- {pdg_round(f1, f2)[1]}"

    @staticmethod
    def test_edge_cases():
        set_display_rounding(True)

        sd = ScalarDisplay()
        sd._nom = 10
        sd._err = -2.5

        assert sd._repr_html_() == "10"
        assert sd._repr_latex_() == "10"
        assert sd.__str__() == "10"
        assert sd.__format__("") == "10"

        set_display_rounding(False)  # reset state for rest of tests


@pytest.mark.parametrize(
    "val, expected",
    [
        (0, 0),
        (1, 0),
        (100, 2),
        (3.2e7, 7),
        (999.99, 2),
        (-1, 0),
        (-10e5, 6),
        (1e-3, -3),
        (39.2e-12, -11),
    ],
)
def test_first_digit(val, expected):
    assert first_digit(val) == expected


@pytest.mark.parametrize(
    "val, expected", [(1, (2, 1)), (0.99, (2, 1.0)), (4.235, (1, 4.235))]
)
def test_PDG_precision(val, expected):
    assert PDG_precision(val) == expected


@pytest.mark.parametrize(
    "val1, val2, expected1, expected2",
    [
        (1, 2, ("1", "2"), ("1.0", "2.0")),
        (0, 0.125, ("0", "0.125"), ("0.0", "0.12")),
        (1, -0.1252, ("1", "-0.1252"), ("1", "")),
    ],
)
def test_pdg_round(val1, val2, expected1, expected2):
    assert pdg_round(val1, val2) == expected1

    set_display_rounding(True)
    assert pdg_round(val1, val2) == expected2

    set_display_rounding(False)  # reset state for rest of tests
