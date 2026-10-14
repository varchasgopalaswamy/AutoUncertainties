from __future__ import annotations

import decimal

from hypothesis import given
from hypothesis import strategies as st
import numpy as np
import pytest

from auto_uncertainties import Uncertainty, configure
from auto_uncertainties.display_format import (
    PDG_precision,
    UncertaintyDisplay,
    first_digit,
    pdg_round,
)


class TestVectorDisplay:
    @staticmethod
    @pytest.mark.parametrize(
        ("arr1", "arr2", "expected"),
        (
            (
                np.array([1, 2, 3]),
                np.array([4, 5, 6]),
                "<table><tbody><tr><th>Magnitude</th><td style='text-align:left;'><pre>1, 2, 3</pre></td></tr><tr><th>Error</th><td style='text-align:left;'><pre>4, 5, 6</pre></td></tr></tbody></table>",
            ),
        ),
    )
    def test__repr_html_(arr1, arr2, expected):
        vd = UncertaintyDisplay()
        vd.nom = arr1
        vd.err = arr2
        assert vd._repr_html_() == expected

    @staticmethod
    @pytest.mark.parametrize(
        ("arr1", "arr2", "expected"),
        ((np.array([1, 2, 3]), np.array([4, 5, 6]), "$1 \\pm 4, 2 \\pm 5, 3 \\pm 6~$"),),
    )
    def test__repr_latex_(arr1, arr2, expected):
        vd = UncertaintyDisplay()
        vd.nom = arr1
        vd.err = arr2
        assert vd._repr_latex_() == expected

    @staticmethod
    @pytest.mark.parametrize(
        ("arr1", "arr2", "expected"),
        ((np.array([1, 2, 3]), np.array([4, 5, 6]), "[1 +/- 4, 2 +/- 5, 3 +/- 6]"),),
    )
    def test_str_and_repr(arr1, arr2, expected):
        vd = UncertaintyDisplay()
        vd.nom = arr1
        vd.err = arr2
        assert vd.__str__() == expected
        assert vd.__repr__() == expected

    @staticmethod
    @pytest.mark.parametrize(
        ("arr1", "arr2", "fmt", "expected"),
        (
            (
                np.array([1, 2, 3]),
                np.array([4, 5, 6]),
                "0.2f",
                "[1.00 +/- 4.00, 2.00 +/- 5.00, 3.00 +/- 6.00]",
            ),
        ),
    )
    def test___format__(arr1, arr2, fmt, expected):
        vd = UncertaintyDisplay()
        vd.nom = arr1
        vd.err = arr2
        assert vd.__format__(fmt) == expected


class TestScalarDisplay:
    @pytest.fixture(autouse=True, params=[False, True])
    def rounding(self, request):
        configure(round_on_display=request.param)

    @staticmethod
    @given(
        st.floats(min_value=0, max_value=5.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    def test__repr_html_(f1, f2):
        sd = UncertaintyDisplay()
        sd.nom = f1
        sd.err = f2
        value, error = pdg_round(f1, f2)
        assert sd._repr_html_() == (f"{value} ± {error}" if error else value)

    @staticmethod
    @given(
        st.floats(min_value=0, max_value=5.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    def test__repr_latex_(f1, f2):
        sd = UncertaintyDisplay()
        sd.nom = f1
        sd.err = f2
        value, error = pdg_round(f1, f2)
        assert sd._repr_latex_() == (f"{value} \\pm {error}" if error else value)

    @staticmethod
    @given(
        st.floats(min_value=0, max_value=5.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    def test_str_and_repr(f1, f2):
        sd = UncertaintyDisplay()
        sd.nom = f1
        sd.err = f2
        value, error = pdg_round(f1, f2)
        expected = f"{value} +/- {error}" if error else value
        assert sd.__str__() == expected
        assert sd.__repr__() == expected

    @staticmethod
    @given(
        st.floats(min_value=0, max_value=5.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    def test___format__(f1, f2):
        sd = UncertaintyDisplay()
        sd.nom = f1
        sd.err = f2
        value, error = pdg_round(f1, f2, return_zero=True)
        assert sd.__format__("g") == f"{value} +/- {error}"

    @staticmethod
    def test_edge_cases():
        configure(round_on_display=True)

        sd = UncertaintyDisplay()
        sd.nom = 10
        sd.err = -2.5

        assert sd._repr_html_() == "10"
        assert sd._repr_latex_() == "10"
        assert sd.__str__() == "10"
        assert sd.__format__("g") == "10 +/- 0"

        configure(round_on_display=False)  # reset state for rest of tests


@pytest.mark.parametrize(
    ("val", "expected"),
    (
        (0, 0),
        (1, 0),
        (100, 2),
        (3.2e7, 7),
        (999.99, 2),
        (-1, 0),
        (-10e5, 6),
        (1e-3, -3),
        (39.2e-12, -11),
    ),
)
def test_first_digit(val, expected):
    assert first_digit(val) == expected


@pytest.mark.parametrize(("val", "expected"), ((1, (2, 1)), (0.99, (2, 1.0)), (4.235, (1, 4.235))))
def test_PDG_precision(val, expected):
    assert PDG_precision(val) == expected


@pytest.mark.parametrize(
    ("val1", "val2", "expected1", "expected2"),
    (
        (1, 2, ("1", "2"), ("1.0", "2.0")),
        (0, 0.125, ("0", "0.125"), ("0.00", "0.12")),
        (1, -0.1252, ("1", "-0.1252"), ("1", "")),
    ),
)
def test_pdg_round(val1, val2, expected1, expected2):
    configure(round_on_display=False)
    assert pdg_round(val1, val2) == expected1

    configure(round_on_display=True)
    assert pdg_round(val1, val2) == expected2

    configure(round_on_display=False)  # reset state for rest of tests


@pytest.mark.parametrize(
    ("value", "error", "expected"),
    (
        # Examples from the PDG introduction, section 5.3.
        (0.827, 0.119, ("0.83", "0.12")),
        (0.827, 0.367, ("0.8", "0.4")),
        (1.234567, 0.0123, ("1.235", "0.012")),
        (1.234567, 0.0354, ("1.235", "0.035")),
        (1.234567, 0.03549, ("1.235", "0.035")),
        (1.234567, 0.0355, ("1.23", "0.04")),
        (1.234567, 0.0949, ("1.23", "0.09")),
        (1.234567, 0.09499, ("1.23", "0.09")),
        (1.234567, 0.095, ("1.23", "0.10")),
        (1.234567, 0.0999, ("1.23", "0.10")),
        (1.234567, 0.001, ("1.2346", "0.0010")),
        (123.456, 12.3, ("123", "12")),
        (123.456, 35.5, ("1.2e+2", "4e+1")),
        (123.456, 95, ("1.2e+2", "1.0e+2")),
        (-1.234567, 0.0123, ("-1.235", "0.012")),
        (0, 0.0123, ("0.000", "0.012")),
        (0.001234, 0.4, ("0.0", "0.4")),
        # Retain trailing zeros and use ties-to-even without double rounding.
        (1.225, 0.125, ("1.22", "0.12")),
        (1.235, 0.125, ("1.24", "0.12")),
    ),
)
def test_pdg_round_reference_cases(value, error, expected):
    configure(round_on_display=True)
    assert pdg_round(value, error) == expected


@pytest.mark.parametrize(
    ("dtype", "expected_value"),
    ((np.float16, "1.234"), (np.float32, "1.235"), (np.float64, "1.235")),
)
def test_pdg_round_numpy_scalars_and_display(dtype, expected_value):
    configure(round_on_display=True)
    value, error = dtype(1.234567), dtype(0.0123)
    assert pdg_round(value, error) == (expected_value, "0.012")
    scalar = Uncertainty(value, error)
    vector = Uncertainty(np.array([value]), np.array([error]))
    assert str(scalar) == f"{expected_value} +/- 0.012"
    assert format(scalar, "g") == f"{expected_value} +/- 0.012"
    assert scalar._repr_html_() == f"{expected_value} ± 0.012"
    assert scalar._repr_latex_() == rf"{expected_value} \pm 0.012"
    assert str(vector) == f"[{expected_value} +/- 0.012]"
    assert format(vector, "g") == f"[{expected_value} +/- 0.012]"
    assert expected_value in vector._repr_html_()
    assert "0.012" in vector._repr_latex_()


def test_pdg_round_explicit_format():
    configure(round_on_display=True)
    assert pdg_round(123.456, 12.3, "f") == ("123", "12")
    assert pdg_round(1.234567, 0.0123, ".3f") == ("1.235", "0.012")


@pytest.mark.parametrize(
    ("value", "error", "expected"),
    (
        (1e-300, 1.23e-302, ("1.000e-300", "1.2e-302")),
        (1e300, 1.23e298, ("1.000e+300", "1.2e+298")),
        (0, 5e-324, ("0e-324", "5e-324")),
        (1e308, 9.99e307, ("1.0e+308", "1.0e+308")),
    ),
)
def test_pdg_round_extreme_magnitudes(value, error, expected):
    configure(round_on_display=True)
    assert pdg_round(value, error) == expected


def test_pdg_round_large_difference_in_magnitudes():
    configure(round_on_display=True)
    value, error = pdg_round(1e308, 5e-324)
    assert decimal.Decimal(value) == decimal.Decimal("1e308")
    assert error == "5e-324"
    assert decimal.Decimal(value).as_tuple().exponent == -324


def test_pdg_round_ignores_external_decimal_precision():
    configure(round_on_display=True)
    with decimal.localcontext() as context:
        context.prec = 2
        assert pdg_round(1.234567, 0.0949) == ("1.23", "0.09")


def test_pdg_round_ignores_external_decimal_traps_and_exponent_limits():
    configure(round_on_display=True)
    with decimal.localcontext() as context:
        context.prec = 2
        context.rounding = decimal.ROUND_UP
        context.Emin = -2
        context.Emax = 2
        for signal in context.traps:
            context.traps[signal] = True
        assert str(Uncertainty(1.2345, 0.012)) == "1.234 +/- 0.012"
        assert pdg_round(1e300, 1.23e298) == ("1.000e+300", "1.2e+298")
        assert pdg_round(1.225, 0.001, ".2f") == ("1.22", "0.00")
        assert context.prec == 2
        assert context.rounding == decimal.ROUND_UP
        assert context.Emax == 2
        assert all(context.traps.values())
        assert not any(context.flags.values())
